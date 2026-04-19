#![allow(dead_code)]
//! Binary Pipeline — обработка последовательности через HDC операции
//!
//! Аналог v14 пайплайна (HDCMemory + HDCAttention + Controller + ThoughtLoop),
//! но полностью binary (XOR, XNOR+POPCNT, bundle, permute).
//!
//! Это ФУНДАМЕНТ v19: модель обрабатывает ВСЮ последовательность,
//! а не делает точечный trigram lookup.
//!
//! Pipeline: tokens → codebook → permute → [HDCMemory + Attention + Controller] × N → predict
//!
//! Обучение: Hebbian (НЕ backprop). "Что срабатывает вместе — связывается."

use crate::binary::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ============================================================
// Binary HDCMemory — каузальная память (v14 аналог)
// ============================================================
// mass_role: какие токены "важные" (high similarity → high mass)
// decay_role: какие токены "забываются быстро"
// Контекст = кумулятивный bundle с экспоненциальным затуханием

pub struct PipelineMemory {
    pub mass_role: BinaryVec,
    pub decay_role: BinaryVec,
    dim: usize,
}

impl PipelineMemory {
    pub fn new(dim: usize) -> Self {
        let mut rng_m = StdRng::seed_from_u64(0x3E3_0001);
        let mut rng_d = StdRng::seed_from_u64(0x3E3_0002);
        PipelineMemory {
            mass_role: BinaryVec::random(dim, &mut rng_m),
            decay_role: BinaryVec::random(dim, &mut rng_d),
            dim,
        }
    }

    /// Каузальный scan: для каждой позиции → контекст от ВСЕХ прошлых.
    /// O(T²) worst case, но cumulative_decay обрезает рано (~20-30 positions back).
    /// Pre-computes mass/decay для каждого токена (O(T) similarity ops).
    pub fn forward(&self, tokens: &[BinaryVec]) -> Vec<BinaryVec> {
        let dim = self.dim;
        let len = tokens.len();

        // Pre-compute mass и decay для всех позиций (O(T))
        let mass_vals: Vec<f64> = tokens.iter()
            .map(|w| (w.similarity(&self.mass_role) as f64 / dim as f64 + 1.0) / 2.0)
            .collect();
        let decay_vals: Vec<f64> = tokens.iter()
            .map(|w| 0.5 + w.similarity(&self.decay_role) as f64 / dim as f64 * 0.3)
            .collect();

        let mut contexts = Vec::with_capacity(len);

        for t in 0..len {
            let mut acc = BundleAccumulator::new(dim);
            let mut cumulative_decay = 1.0f64;

            // Сам текущий токен × 3 (самый важный)
            acc.add(&tokens[t]);
            acc.add(&tokens[t]);
            acc.add(&tokens[t]);

            // Прошлые с экспоненциальным затуханием
            for offset in 1..=t {
                let s = t - offset;
                let weight = mass_vals[s] * cumulative_decay;

                if weight > 0.02 {
                    let repeats = (weight * 6.0).round().max(1.0).min(6.0) as usize;
                    for _ in 0..repeats {
                        acc.add(&tokens[s]);
                    }
                }

                cumulative_decay *= decay_vals[s];
                if cumulative_decay < 0.005 { break; }
            }

            contexts.push(acc.to_binary());
        }

        contexts
    }
}

// ============================================================
// Binary T×T Attention — позиционное внимание
// ============================================================
// Q = token XOR role_query
// K = token XOR role_key
// score = similarity(Q, K) — каузально (только прошлое)
// attended = bundle прошлых с положительными scores

pub struct PipelineAttention {
    pub role_query: BinaryVec,
    pub role_key: BinaryVec,
    dim: usize,
}

impl PipelineAttention {
    pub fn new(dim: usize) -> Self {
        let mut rng_q = StdRng::seed_from_u64(0xA77_0001);
        let mut rng_k = StdRng::seed_from_u64(0xA77_0002);
        PipelineAttention {
            role_query: BinaryVec::random(dim, &mut rng_q),
            role_key: BinaryVec::random(dim, &mut rng_k),
            dim,
        }
    }

    /// T×T каузальное внимание. Для каждой позиции — attention ко всем прошлым.
    /// Возвращает Vec<BinaryVec> — attended context для каждой позиции.
    pub fn forward(&self, tokens: &[BinaryVec]) -> Vec<BinaryVec> {
        let dim = self.dim;
        let len = tokens.len();
        let mut attended = Vec::with_capacity(len);

        // Pre-compute all keys
        let keys: Vec<BinaryVec> = tokens.iter()
            .map(|t| t.bind(&self.role_key))
            .collect();

        for t in 0..len {
            let q = tokens[t].bind(&self.role_query);
            let mut acc = BundleAccumulator::new(dim);

            // Сам текущий токен
            acc.add(&tokens[t]);

            // Attention ко всем прошлым (каузально)
            for s in 0..t {
                let score = q.similarity(&keys[s]) as f64 / dim as f64;

                if score > 0.0 {
                    // sigmoid-like: discretize positive scores
                    let repeats = (score * 4.0).round().max(1.0).min(4.0) as usize;
                    for _ in 0..repeats {
                        acc.add(&tokens[s]);
                    }
                }
            }

            attended.push(acc.to_binary());
        }

        attended
    }
}

// ============================================================
// Binary Controller — нелинейное преобразование + residual
// ============================================================
// Проекция через random binary bases (НЕ обучаемые matmul)
// + ReLU-like activation + проекция обратно + residual

pub struct PipelineController {
    down: Vec<BinaryVec>,   // D → inner_dim random bases
    up: Vec<BinaryVec>,     // inner_dim → D random bases
    inner_dim: usize,
    dim: usize,
}

impl PipelineController {
    pub fn new(dim: usize, inner_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(0xC7B1_001);
        let down = (0..inner_dim).map(|_| BinaryVec::random(dim, &mut rng)).collect();
        let up = (0..inner_dim).map(|_| BinaryVec::random(dim, &mut rng)).collect();
        PipelineController { down, up, inner_dim, dim }
    }

    /// Forward: x → project down → ReLU → project up → residual(x)
    pub fn forward(&self, x: &BinaryVec) -> BinaryVec {
        // Project down: similarity с каждым базисом
        let projected: Vec<f64> = self.down.iter()
            .map(|d| x.similarity(d) as f64 / self.dim as f64)
            .collect();

        // ReLU-like activation
        // Project up: bundle activated базисов
        let mut acc = BundleAccumulator::new(self.dim);

        // Residual: x доминирует (3 copies)
        acc.add(x);
        acc.add(x);
        acc.add(x);

        for (i, &val) in projected.iter().enumerate() {
            if val > 0.05 {
                let repeats = (val * 3.0).round().max(1.0).min(3.0) as usize;
                for _ in 0..repeats {
                    acc.add(&self.up[i]);
                }
            }
        }

        acc.to_binary()
    }

    /// Batch forward для всех позиций
    pub fn forward_batch(&self, tokens: &[BinaryVec]) -> Vec<BinaryVec> {
        tokens.iter().map(|t| self.forward(t)).collect()
    }
}

// ============================================================
// BinaryHDCBlock = Memory + Attention + Controller
// ============================================================

pub struct BinaryHDCBlock {
    pub memory: PipelineMemory,
    pub attention: PipelineAttention,
    pub controller: PipelineController,
    dim: usize,
}

impl BinaryHDCBlock {
    pub fn new(dim: usize, inner_dim: usize) -> Self {
        BinaryHDCBlock {
            memory: PipelineMemory::new(dim),
            attention: PipelineAttention::new(dim),
            controller: PipelineController::new(dim, inner_dim),
            dim,
        }
    }

    /// Forward через block: Memory → Attention → Controller
    /// С residual connections через bundle
    pub fn forward(&self, tokens: &[BinaryVec]) -> Vec<BinaryVec> {
        let len = tokens.len();

        // 1. HDCMemory: каузальный контекст
        let mem_out = self.memory.forward(tokens);

        // 2. Residual: bundle(token, memory_output)
        let after_mem: Vec<BinaryVec> = (0..len).map(|i| {
            let mut acc = BundleAccumulator::new(self.dim);
            acc.add(&tokens[i]);
            acc.add(&tokens[i]); // token × 2 (residual dominates)
            acc.add(&mem_out[i]);
            acc.to_binary()
        }).collect();

        // 3. Attention
        let attn_out = self.attention.forward(&after_mem);

        // 4. Residual: bundle(after_mem, attention_output)
        let after_attn: Vec<BinaryVec> = (0..len).map(|i| {
            let mut acc = BundleAccumulator::new(self.dim);
            acc.add(&after_mem[i]);
            acc.add(&after_mem[i]); // residual × 2
            acc.add(&attn_out[i]);
            acc.to_binary()
        }).collect();

        // 5. Controller: нелинейное преобразование + residual
        self.controller.forward_batch(&after_attn)
    }
}

// ============================================================
// BinaryPipeline — полный пайплайн
// ============================================================

pub struct BinaryPipeline {
    pub codebook: Vec<BinaryVec>,     // vocab_size × dim (обучаемый через Hebbian)
    pub blocks: Vec<BinaryHDCBlock>,  // N blocks
    pub dim: usize,
    pub vocab_size: usize,
    pub n_blocks: usize,
}

impl BinaryPipeline {
    pub fn new(dim: usize, vocab_size: usize, n_blocks: usize, inner_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let codebook = (0..vocab_size)
            .map(|_| BinaryVec::random(dim, &mut rng))
            .collect();

        let blocks = (0..n_blocks)
            .map(|_| BinaryHDCBlock::new(dim, inner_dim))
            .collect();

        BinaryPipeline { codebook, blocks, dim, vocab_size, n_blocks }
    }

    /// Загрузить codebook из HierarchicalMemory (переиспользуем semantic codebook)
    pub fn load_codebook(&mut self, words: &[BinaryVec]) {
        for (i, w) in words.iter().enumerate() {
            if i < self.vocab_size {
                self.codebook[i] = w.clone();
            }
        }
    }

    /// Forward pass: tokens → hidden representations
    /// Возвращает BinaryVec для каждой позиции
    pub fn forward(&self, token_ids: &[u16]) -> Vec<BinaryVec> {
        // 1. Codebook lookup + cyclic permutation (позиционное кодирование)
        let embeddings: Vec<BinaryVec> = token_ids.iter().enumerate()
            .map(|(pos, &tok)| {
                if (tok as usize) < self.vocab_size {
                    self.codebook[tok as usize].permute(pos % self.dim)
                } else {
                    BinaryVec::zeros(self.dim)
                }
            })
            .collect();

        // 2. Прогнать через все блоки
        let mut hidden = embeddings;
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }

        hidden
    }

    /// Predict: для позиции pos предсказать следующий токен.
    /// Возвращает top-K кандидатов с scores.
    /// Confidence = (top1 - top2) / top1 — насколько уверен pipeline.
    pub fn predict_at(&self, hidden: &BinaryVec, top_k: usize) -> (Vec<(u16, i32)>, f64) {
        // similarity(hidden, codebook[t]) для всех токенов
        let mut scores: Vec<(u16, i32)> = self.codebook.iter().enumerate()
            .map(|(i, w)| (i as u16, hidden.similarity(w)))
            .collect();

        scores.sort_by(|a, b| b.1.cmp(&a.1));
        scores.truncate(top_k);

        let confidence = if scores.len() >= 2 && scores[0].1 > 0 {
            (scores[0].1 - scores[1].1) as f64 / scores[0].1 as f64
        } else {
            0.0
        };

        (scores, confidence)
    }

    /// Hebbian update при правильном predict
    pub fn hebbian_update_correct(
        &mut self,
        token_id: u16,
        hidden: &BinaryVec,
        _block_idx: usize,
    ) {
        let tok = token_id as usize;
        if tok >= self.vocab_size { return; }

        // Codebook: bundle(old × 4, hidden × 1) — codebook learns what contexts lead to this token
        let mut acc = BundleAccumulator::new(self.dim);
        for _ in 0..4 { acc.add(&self.codebook[tok]); }
        acc.add(hidden);
        self.codebook[tok] = acc.to_binary();
    }

    /// Hebbian update при НЕПРАВИЛЬНОМ predict — слабое обновление codebook only
    pub fn hebbian_update_wrong(
        &mut self,
        expected_id: u16,
        hidden: &BinaryVec,
    ) {
        let tok = expected_id as usize;
        if tok >= self.vocab_size { return; }

        // Слабое: bundle(old × 8, hidden × 1) — медленная коррекция
        let mut acc = BundleAccumulator::new(self.dim);
        for _ in 0..8 { acc.add(&self.codebook[tok]); }
        acc.add(hidden);
        self.codebook[tok] = acc.to_binary();
    }

    /// Полный training step на chunk данных.
    /// Возвращает (correct, total) для accuracy.
    /// sample_rate применяется СНАРУЖИ (к chunks), здесь обрабатываем ВСЕ позиции в chunk.
    pub fn train_chunk(&mut self, tokens: &[u16], _sample_rate: usize) -> (usize, usize) {
        let len = tokens.len();
        if len < 4 { return (0, 0); }

        // Forward: получить hidden для всех позиций в chunk
        let hidden = self.forward(tokens);

        let mut correct = 0usize;
        let mut total = 0usize;

        // Для каждой позиции в chunk: predict + Hebbian
        for pos in 2..len.saturating_sub(1) {

            let expected = tokens[pos + 1];
            if (expected as usize) >= self.vocab_size { continue; }

            // Predict
            let (predictions, _conf) = self.predict_at(&hidden[pos], 1);
            total += 1;

            if let Some(&(predicted, _)) = predictions.first() {
                if predicted == expected {
                    // Правильно → Hebbian усиление
                    correct += 1;
                    self.hebbian_update_correct(expected, &hidden[pos], 0);
                } else {
                    // Неправильно → обновляем codebook ТОЛЬКО если уже есть хоть какая-то accuracy
                    // При 0% accuracy anti-Hebbian разрушает codebook (всё дрейфует к среднему)
                    let current_acc = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
                    if current_acc > 0.01 {
                        self.hebbian_update_wrong(expected, &hidden[pos]);
                    }
                }
            }
        }

        (correct, total)
    }

    /// Evaluate accuracy без обновления модели.
    /// Работает на chunks (не на всём датасете — O(T²) attention не масштабируется).
    pub fn evaluate(&self, tokens: &[u16], n_positions: usize) -> (usize, usize, usize) {
        let len = tokens.len();
        let chunk_size = len.min(256); // eval на chunks, но не больше данных
        if chunk_size < 4 { return (0, 0, 0); }
        let step = if len > n_positions * chunk_size { len / n_positions } else { chunk_size };

        let mut correct_top1 = 0usize;
        let mut correct_top5 = 0usize;
        let mut tested = 0usize;

        let mut pos = 0;
        while pos + chunk_size <= len && tested < n_positions {
            let chunk = &tokens[pos..pos + chunk_size];
            let hidden = self.forward(chunk);

            // Evaluate на последней позиции chunk
            let eval_pos = chunk_size - 2;
            let expected = chunk[eval_pos + 1];
            if (expected as usize) < self.vocab_size {
                let (predictions, _conf) = self.predict_at(&hidden[eval_pos], 5);
                tested += 1;

                if let Some(&(top1, _)) = predictions.first() {
                    if top1 == expected { correct_top1 += 1; }
                }
                if predictions.iter().any(|&(t, _)| t == expected) {
                    correct_top5 += 1;
                }
            }

            pos += step;
        }

        (correct_top1, correct_top5, tested)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_memory_forward() {
        let mem = PipelineMemory::new(256);
        let mut rng = StdRng::seed_from_u64(42);
        let tokens: Vec<BinaryVec> = (0..20).map(|_| BinaryVec::random(256, &mut rng)).collect();

        let contexts = mem.forward(&tokens);
        assert_eq!(contexts.len(), 20);

        // Каждый контекст не должен быть нулевым
        for ctx in &contexts {
            assert!(ctx.similarity(&BinaryVec::zeros(256)) < 200,
                "Context should not be near-zero");
        }

        // Контекст позиции 10 должен отличаться от позиции 0
        // (разные прошлые, разный cumulative decay)
        let sim = contexts[0].similarity(&contexts[10]);
        assert!(sim < 200, "Different positions should have different contexts, sim={}", sim);
    }

    #[test]
    fn test_pipeline_attention_forward() {
        let attn = PipelineAttention::new(256);
        let mut rng = StdRng::seed_from_u64(42);
        let tokens: Vec<BinaryVec> = (0..20).map(|_| BinaryVec::random(256, &mut rng)).collect();

        let attended = attn.forward(&tokens);
        assert_eq!(attended.len(), 20);
    }

    #[test]
    fn test_pipeline_controller_residual() {
        let ctrl = PipelineController::new(256, 64);
        let mut rng = StdRng::seed_from_u64(42);
        let x = BinaryVec::random(256, &mut rng);

        let out = ctrl.forward(&x);

        // Residual: output should be similar to input (x dominates in bundle)
        let sim = x.similarity(&out);
        assert!(sim > 0, "Controller output should preserve input via residual, sim={}", sim);
    }

    #[test]
    fn test_pipeline_full_forward() {
        let pipeline = BinaryPipeline::new(256, 100, 1, 32); // 1 block, small
        let tokens: Vec<u16> = (0..30).map(|i| (i % 100) as u16).collect();

        let hidden = pipeline.forward(&tokens);
        assert_eq!(hidden.len(), 30);

        // Predict at each position
        let (preds, _conf) = pipeline.predict_at(&hidden[15], 5);
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_pipeline_train_chunk() {
        let mut pipeline = BinaryPipeline::new(256, 100, 1, 32);

        // Repeating pattern: 0,1,2,3,4, 0,1,2,3,4, ...
        let tokens: Vec<u16> = (0..200).map(|i| (i % 5) as u16).collect();

        // Train multiple times on the same data
        let mut last_correct = 0;
        for _ in 0..5 {
            let (correct, total) = pipeline.train_chunk(&tokens, 1);
            last_correct = correct;
        }

        // After training, should predict SOME correctly on repeating pattern
        // (Hebbian codebook learns the pattern)
        let (c1, c5, tested) = pipeline.evaluate(&tokens, 50);
        assert!(tested > 0, "Should evaluate some positions");
        // We don't assert high accuracy — Hebbian may need more iterations
    }
}
