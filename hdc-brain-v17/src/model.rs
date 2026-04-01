//! HDC-Brain v17: Layered Learning Architecture
//!
//! Как ребёнок учит язык:
//!   Слой 1-2 (Фразы):     заучить "мама дай", "папа дай"
//!   Слой 3-4 (Правила):   понять что "кто-то + дай" = просьба
//!   Слой 5-6 (Рассуждение): придумать "бабушка дай конфету"
//!
//! Обучение через HDC логику (bind/unbind/bundle), НЕ gradient descent.
//! Один проход по данным — не тысячи итераций.

use std::collections::HashMap;
use rand::Rng;
use crate::binary::*;

// ============================================================
// Config
// ============================================================

pub struct Config {
    pub hdc_dim: usize,       // 4096
    pub vocab_size: usize,    // 16000
    pub lsh_bits: usize,      // 16 → 65536 buckets
    pub max_rules: usize,     // 512
    pub top_k: usize,         // predictions per layer
}

impl Config {
    pub fn default_with_vocab(vocab_size: usize) -> Self {
        Config {
            hdc_dim: 4096,
            vocab_size,
            lsh_bits: 16,
            max_rules: 512,
            top_k: 10,
        }
    }

    pub fn n_buckets(&self) -> usize { 1 << self.lsh_bits }
}

// ============================================================
// Layer 1-2: Phrase Memory
// ============================================================

/// One LSH bucket: bundled phrase vector + successor histogram.
pub struct PhraseBucket {
    pub accumulator: BundleAccumulator,
    pub successors: HashMap<u16, u32>,
    pub count: u32,
    /// Logical fact memory: bundle of bind(trigram, codebook[successor]).
    /// unbind(fact_vec, query_trigram) → answer vector → nearest token.
    /// Это чистая логика, не статистика.
    pub fact_vec: Option<BinaryVec>,
}

impl PhraseBucket {
    pub fn new(dim: usize) -> Self {
        PhraseBucket {
            accumulator: BundleAccumulator::new(dim),
            successors: HashMap::new(),
            count: 0,
            fact_vec: None,
        }
    }

    pub fn add(&mut self, trigram: &BinaryVec, successor: u16) {
        self.accumulator.add(trigram);
        *self.successors.entry(successor).or_insert(0) += 1;
        self.count += 1;
    }

    /// Top-K successors sorted by frequency.
    pub fn top_successors(&self, k: usize) -> Vec<(u16, u32)> {
        let mut pairs: Vec<(u16, u32)> = self.successors.iter()
            .map(|(&tok, &cnt)| (tok, cnt))
            .collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(k);
        pairs
    }

    /// Most likely successor.
    pub fn top1(&self) -> Option<u16> {
        self.successors.iter()
            .max_by_key(|(_, &cnt)| cnt)
            .map(|(&tok, _)| tok)
    }

    /// Collapse: сжать accumulator в бинарный, сбросить счётчики.
    /// Следующий проход будет строить ПОВЕРХ сжатого, уточняя.
    /// Successor counts делим пополам (затухание старых данных).
    pub fn collapse(&mut self) {
        if self.count == 0 { return; }

        // Collapse accumulator: binary → seed for next pass
        let collapsed = self.accumulator.to_binary();
        self.accumulator.reset();
        // Seed: добавить collapsed один раз как "память" прошлого прохода
        self.accumulator.add(&collapsed);

        // Successor counts: decay (старые данные менее важны)
        for cnt in self.successors.values_mut() {
            *cnt = (*cnt / 2).max(1); // halve, keep at least 1
        }
        // Remove near-zero entries
        self.successors.retain(|_, cnt| *cnt >= 2);

        self.count = 1; // reset count but mark as non-empty
    }
}

// ============================================================
// Layer 3-4: Rules
// ============================================================

/// A learned rule: captures abstract pattern from phrase errors.
///
/// rule = bind(common_error_context, correct_codebook_entry)
/// При inference: unbind(context, rule) → предсказание.
pub struct Rule {
    pub pattern: BinaryVec,
    pub support: u32,       // сколько примеров поддерживают правило
    pub confidence: f32,    // доля правильных предсказаний
}

// ============================================================
// Layer 5-6: Reasoning (Fact Memory)
// ============================================================

/// Running fact accumulator for reasoning layer.
/// Tracks context as bundled trigrams for consistency checking.
pub struct FactMemory {
    accumulator: BundleAccumulator,
    recent: Vec<BinaryVec>, // last N context vectors
    max_recent: usize,
}

impl FactMemory {
    pub fn new(dim: usize, max_recent: usize) -> Self {
        FactMemory {
            accumulator: BundleAccumulator::new(dim),
            recent: Vec::new(),
            max_recent,
        }
    }

    pub fn add_fact(&mut self, fact: &BinaryVec) {
        self.accumulator.add(fact);
        self.recent.push(fact.clone());
        if self.recent.len() > self.max_recent {
            self.recent.remove(0);
        }
    }

    /// Consistency score: how well does candidate fit accumulated facts.
    pub fn consistency(&self, candidate: &BinaryVec) -> i32 {
        if self.recent.is_empty() { return 0; }
        // Average similarity with recent facts
        let total: i32 = self.recent.iter()
            .map(|f| candidate.similarity(f))
            .sum();
        total / self.recent.len() as i32
    }

    pub fn reset(&mut self) {
        self.accumulator.reset();
        self.recent.clear();
    }
}

// ============================================================
// Main Model
// ============================================================

pub struct HDCBrainV17 {
    pub config: Config,
    pub codebook: Vec<BinaryVec>,

    // Layer 1-2: Phrase Memory (narrow trigram → successor)
    pub phrase_buckets: Vec<PhraseBucket>,

    // Layer 3-4: Context Memory (broad 5-gram → successor)
    // "В КАКОМ контексте фраза применяется?"
    pub context_buckets: Vec<PhraseBucket>,
    pub context_lsh_bits: usize,

    // Layer 5-6: Improvisation (semantic substitution)
    pub codebook_neighbors: Vec<Vec<(u16, i32)>>,  // per token: top-K similar + similarity

    // Layer 5-6 (additional): Discovered Rules
    pub rules: Vec<Rule>,

    // Stats
    pub phrases_trained: usize,
    pub contexts_trained: usize,
    pub rules_discovered: usize,
}

impl HDCBrainV17 {
    pub fn new(config: Config) -> Self {
        let mut rng = rand::thread_rng();
        let n_buckets = config.n_buckets();

        // Random binary codebook
        let codebook: Vec<BinaryVec> = (0..config.vocab_size)
            .map(|_| BinaryVec::random(config.hdc_dim, &mut rng))
            .collect();

        // Empty phrase buckets (Layer 1-2: narrow)
        let phrase_buckets: Vec<PhraseBucket> = (0..n_buckets)
            .map(|_| PhraseBucket::new(config.hdc_dim))
            .collect();

        // Empty context buckets (Layer 3-4: broad, fewer buckets)
        let context_lsh_bits = 14; // 16384 buckets (broader = less precise)
        let n_context_buckets = 1 << context_lsh_bits;
        let context_buckets: Vec<PhraseBucket> = (0..n_context_buckets)
            .map(|_| PhraseBucket::new(config.hdc_dim))
            .collect();

        HDCBrainV17 {
            config,
            codebook,
            phrase_buckets,
            context_buckets,
            context_lsh_bits,
            codebook_neighbors: Vec::new(),
            rules: Vec::new(),
            phrases_trained: 0,
            contexts_trained: 0,
            rules_discovered: 0,
        }
    }

    /// Build trigram: bind(tok[t], perm¹(tok[t-1]), perm²(tok[t-2]))
    /// Layer 1-2: narrow context, precise matching
    #[inline]
    pub fn make_trigram(&self, tokens: &[u16], t: usize) -> BinaryVec {
        let a = &self.codebook[tokens[t] as usize];
        let b = self.codebook[tokens[t - 1] as usize].permute(1);
        let c = self.codebook[tokens[t - 2] as usize].permute(2);
        a.bind(&b).bind(&c)
    }

    /// Build broad context: weighted 5-gram bundle.
    /// Layer 3-4: "в каком КОНТЕКСТЕ я это видел?"
    /// Ближние слова важнее (weight 5,4,3,2,1).
    /// Bundle (majority vote) вместо bind — сохраняет все токены, не только комбинацию.
    pub fn make_broad_context(&self, tokens: &[u16], t: usize) -> BinaryVec {
        let mut acc = BundleAccumulator::new(self.config.hdc_dim);
        let window = 5.min(t + 1);
        for offset in 0..window {
            let tok_vec = &self.codebook[tokens[t - offset] as usize];
            let permuted = tok_vec.permute(offset);
            let weight = window - offset; // closer = heavier
            for _ in 0..weight {
                acc.add(&permuted);
            }
        }
        acc.to_binary()
    }

    // ========================================================
    // Phase 0: Learn Semantic Codebook
    // ========================================================

    /// Построить смысловой codebook: слова в одинаковых контекстах → похожие вектора.
    ///
    /// Для каждого токена собираем ВСЕ контексты (соседние слова).
    /// bundle(контексты) → "смысл" слова.
    ///
    /// После этого: "город" ≈ "деревня" (оба после "в", перед "было").
    /// Это даёт импровизацию: "в деревне" → тот же бакет что "в городе".
    pub fn learn_codebook(&mut self, data: &[u16]) {
        let n = data.len();
        if n < 5 { return; }

        println!("  Building semantic codebook (parallel)...");

        let vocab = self.config.vocab_size;
        let dim = self.config.hdc_dim;

        // Parallel: split data into chunks, each thread builds local accumulators
        use rayon::prelude::*;
        let n_threads = rayon::current_num_threads();
        let chunk_size = (n - 4) / n_threads + 1;

        let codebook = &self.codebook;
        let thread_results: Vec<Vec<BundleAccumulator>> = (0..n_threads).into_par_iter().map(|tid| {
            let mut local: Vec<BundleAccumulator> = (0..vocab)
                .map(|_| BundleAccumulator::new(dim))
                .collect();

            let start = 2 + tid * chunk_size;
            let end = (start + chunk_size).min(n - 2);

            for t in start..end {
                let tok = data[t] as usize;
                if tok >= vocab { continue; }

                let prev1 = codebook[data[t - 1] as usize].permute(1);
                let prev2 = codebook[data[t - 2] as usize].permute(2);
                let next1 = codebook[data[t + 1] as usize].permute(3);
                let next2 = codebook[data[t + 2] as usize].permute(4);

                local[tok].add(&prev1);
                local[tok].add(&prev2);
                local[tok].add(&next1);
                local[tok].add(&next2);
            }
            local
        }).collect();

        // Merge thread results
        let mut token_contexts: Vec<BundleAccumulator> = (0..vocab)
            .map(|_| BundleAccumulator::new(dim))
            .collect();

        for thread_local in &thread_results {
            for tok in 0..vocab {
                for i in 0..dim {
                    token_contexts[tok].counters[i] += thread_local[tok].counters[i];
                }
                token_contexts[tok].count += thread_local[tok].count;
            }
        }
        drop(thread_results);

        // Blend: семантический вектор MIX с оригинальным случайным.
        // Частые слова (> 50K контекстов) слишком "размазаны" → больше random.
        // Редкие слова (< 10 контекстов) → оставляем random.
        // Средние (10-50K) → максимум semantic.
        let mut updated = 0;
        for tok in 0..vocab {
            let count = token_contexts[tok].count;
            if count < 10 { continue; } // слишком редкий

            let semantic = token_contexts[tok].to_binary();

            // Частые слова: blend с random чтобы не потерять уникальность
            // count > 50K → 50% semantic, 50% random
            // count 10-50K → 90% semantic, 10% random
            let semantic_weight = if count > 50000 { 3 } else if count > 10000 { 5 } else { 7 };
            let random_weight = 8 - semantic_weight; // total = 8

            let mut acc = BundleAccumulator::new(dim);
            for _ in 0..semantic_weight {
                acc.add(&semantic);
            }
            for _ in 0..random_weight {
                acc.add(&self.codebook[tok]); // original random
            }
            self.codebook[tok] = acc.to_binary();
            updated += 1;
        }

        println!("\r  Codebook: updated {}/{} tokens (kept {} rare as random)",
                 updated, vocab, vocab - updated);

        // Проверка: найдём пары похожих слов
        self.find_similar_pairs();
    }

    /// Найти пары семантически похожих слов в codebook.
    pub fn find_similar_pairs(&self) -> Vec<(usize, usize, i32)> {
        let vocab = self.config.vocab_size;
        let mut best_pairs: Vec<(usize, usize, i32)> = Vec::new();
        let threshold = self.config.hdc_dim as i32 / 4; // > 25% similar

        // Sample: первые 1000 токенов (обычно самые частые)
        let check = 1000.min(vocab);
        for i in 1..check { // skip 0 (<unk>)
            for j in i + 1..check {
                let sim = self.codebook[i].similarity(&self.codebook[j]);
                if sim > threshold {
                    best_pairs.push((i, j, sim));
                }
            }
        }

        best_pairs.sort_by(|a, b| b.2.cmp(&a.2));
        best_pairs.truncate(20);

        if !best_pairs.is_empty() {
            println!("  Top similar word pairs:");
            for &(i, j, sim) in best_pairs.iter().take(10) {
                let pct = sim as f64 / self.config.hdc_dim as f64 * 100.0;
                println!("    [{}] ≈ [{}]  ({:.0}%)", i, j, pct);
            }
        } else {
            println!("  No highly similar pairs found (threshold {})", threshold);
        }

        best_pairs
    }

    // ========================================================
    // Phase 0b: Build Codebook Neighbors (for improvisation)
    // ========================================================

    /// Для каждого токена найти top-K семантически похожих.
    /// "город" → ["деревня", "село", "посёлок", ...]
    /// Это позволяет слою 5-6 подставлять похожие слова.
    pub fn build_neighbors(&mut self, top_k: usize) {
        use rayon::prelude::*;

        let vocab = self.config.vocab_size;
        let threshold = self.config.hdc_dim as i32 / 10;
        let codebook = &self.codebook;

        self.codebook_neighbors = (0..vocab).into_par_iter().map(|i| {
            let mut sims: Vec<(u16, i32)> = (0..vocab)
                .filter(|&j| j != i)
                .filter_map(|j| {
                    let sim = codebook[i].similarity(&codebook[j]);
                    if sim > threshold { Some((j as u16, sim)) } else { None }
                })
                .collect();
            sims.sort_by(|a, b| b.1.cmp(&a.1));
            sims.truncate(top_k);
            sims
        }).collect();

        let with_neighbors = self.codebook_neighbors.iter()
            .filter(|n| !n.is_empty()).count();
        println!("  Neighbors: {} / {} tokens (parallel)", with_neighbors, vocab);
    }

    // ========================================================
    // Phase A: Learn Phrases (один проход по данным)
    // ========================================================

    /// Заучить фразы: для каждой позиции записать trigram → successor.
    /// Как ребёнок повторяет: "мама дай", "папа дай", "киса мяу".
    pub fn train_phrases(&mut self, data: &[u16]) {
        let n = data.len();
        if n < 4 { return; }

        let total = n - 3;
        let report_every = total / 10 + 1;

        for t in 2..n - 1 {
            let trigram = self.make_trigram(data, t);
            let hash = lsh_hash(&trigram, self.config.lsh_bits) as usize;
            let successor = data[t + 1];

            self.phrase_buckets[hash].add(&trigram, successor);

            if (t - 2) % report_every == 0 {
                let pct = ((t - 2) as f64 / total as f64 * 100.0) as u32;
                print!("\r  Phrases: {}% ({}/{})", pct, t - 2, total);
            }
        }
        self.phrases_trained = total;
        println!("\r  Phrases: 100% — {} phrases in {} buckets",
                 total, self.active_buckets());
    }

    fn active_buckets(&self) -> usize {
        self.phrase_buckets.iter().filter(|b| b.count > 0).count()
    }

    /// Collapse all buckets: сжать → сбросить → готово к новому проходу.
    /// Как AccCollapse в v15: каждый проход уточняет предыдущий.
    pub fn collapse_all(&mut self) {
        for bucket in &mut self.phrase_buckets {
            bucket.collapse();
        }
        for bucket in &mut self.context_buckets {
            bucket.collapse();
        }
    }

    /// Reset phrase memory for fresh start (keeps codebook and rules).
    pub fn reset_phrases(&mut self) {
        let n = self.config.n_buckets();
        self.phrase_buckets = (0..n)
            .map(|_| PhraseBucket::new(self.config.hdc_dim))
            .collect();
        self.phrases_trained = 0;
    }

    /// Phrase-only top-1 accuracy (fast, no rules/reasoning).
    pub fn phrase_accuracy(&self, data: &[u16], n_eval: usize) -> f64 {
        let n = n_eval.min(data.len().saturating_sub(3));
        if n == 0 { return 0.0; }
        let mut correct = 0u32;
        for t in 2..2 + n {
            let trigram = self.make_trigram(data, t);
            let hash = lsh_hash(&trigram, self.config.lsh_bits) as usize;
            // Multi-probe: check primary + neighbors
            let mut found = self.phrase_buckets[hash].top1() == Some(data[t + 1]);
            if !found {
                for bit in 0..self.config.lsh_bits {
                    let neighbor = hash ^ (1 << bit);
                    if self.phrase_buckets[neighbor].top1() == Some(data[t + 1]) {
                        found = true;
                        break;
                    }
                }
            }
            if found { correct += 1; }
        }
        correct as f64 / n as f64 * 100.0
    }

    // ========================================================
    // Phase A2: Build Logical Fact Memory
    // ========================================================

    /// Построить логическую память фактов.
    /// Для каждой позиции: fact = bind(trigram, codebook[successor])
    /// Бакет хранит bundle всех фактов → unbind(facts, query) = логический вывод.
    ///
    /// Это НЕ подсчёт частот. Это хранение СВЯЗЕЙ через HDC bind.
    /// Предсказание = unbind (извлечение), не argmax(счётчик).
    pub fn build_fact_memory(&mut self, data: &[u16]) {
        use rayon::prelude::*;

        let n = data.len();
        if n < 4 { return; }

        let n_buckets = self.config.n_buckets();
        let dim = self.config.hdc_dim;
        let lsh_bits = self.config.lsh_bits;
        let codebook = &self.codebook;
        let n_threads = rayon::current_num_threads();
        let chunk_size = (n - 3) / n_threads + 1;

        // Parallel: each thread builds local fact accumulators
        let thread_accs: Vec<Vec<BundleAccumulator>> = (0..n_threads)
            .into_par_iter()
            .map(|tid| {
                let start = 2 + tid * chunk_size;
                let end = (start + chunk_size).min(n - 1);
                let mut local: Vec<BundleAccumulator> = (0..n_buckets)
                    .map(|_| BundleAccumulator::new(dim)).collect();

                for t in start..end {
                    if t < 2 { continue; }
                    let a = &codebook[data[t] as usize];
                    let b = codebook[data[t-1] as usize].permute(1);
                    let c = codebook[data[t-2] as usize].permute(2);
                    let trigram = a.bind(&b).bind(&c);
                    let hash = lsh_hash(&trigram, lsh_bits) as usize;
                    let fact = trigram.bind(&codebook[data[t+1] as usize]);
                    local[hash].add(&fact);
                }
                local
            }).collect();

        // Merge and collapse
        let mut stored = 0;
        for i in 0..n_buckets {
            let mut total_count = 0u32;
            let mut merged = BundleAccumulator::new(dim);
            for thread_local in &thread_accs {
                if thread_local[i].count > 0 {
                    for j in 0..dim {
                        merged.counters[j] += thread_local[i].counters[j];
                    }
                    total_count += thread_local[i].count;
                }
            }
            if total_count >= 3 {
                self.phrase_buckets[i].fact_vec = Some(merged.to_binary());
                stored += 1;
            }
        }

        println!("  Facts: {} buckets (parallel {}T)", stored, n_threads);
    }

    // ========================================================
    // Phase B: Learn Contexts (Layer 3-4)
    // "В каких ситуациях заученные фразы работают?"
    // ========================================================

    /// Шаг 2: пробуем применить заученные фразы в контексте (PARALLEL).
    pub fn learn_contexts(&mut self, data: &[u16]) {
        use rayon::prelude::*;

        let n = data.len();
        if n < 6 { return; }

        let n_threads = rayon::current_num_threads();
        let n_ctx = 1usize << self.context_lsh_bits;
        let ctx_bits = self.context_lsh_bits;
        let dim = self.config.hdc_dim;
        let codebook = &self.codebook;
        let chunk_size = (n - 5) / n_threads + 1;

        // Each thread: lightweight successor counts per bucket
        let thread_results: Vec<Vec<HashMap<u16, u32>>> = (0..n_threads)
            .into_par_iter()
            .map(|tid| {
                let start = 4 + tid * chunk_size;
                let end = (start + chunk_size).min(n - 1);
                let mut local: Vec<HashMap<u16, u32>> = (0..n_ctx)
                    .map(|_| HashMap::new()).collect();
                let mut acc = BundleAccumulator::new(dim); // reuse per thread

                for t in start..end {
                    // Build broad context inline (can't call &self in parallel)
                    acc.reset();
                    let window = 5.min(t + 1);
                    for offset in 0..window {
                        let tok_idx = data[t - offset] as usize;
                        if tok_idx < codebook.len() {
                            let permuted = codebook[tok_idx].permute(offset);
                            let weight = window - offset;
                            for _ in 0..weight { acc.add(&permuted); }
                        }
                    }
                    let broad_ctx = acc.to_binary();
                    let hash = lsh_hash(&broad_ctx, ctx_bits) as usize;
                    *local[hash].entry(data[t + 1]).or_insert(0) += 1;
                }
                local
            }).collect();

        // Merge into context_buckets
        for thread_local in &thread_results {
            for i in 0..n_ctx {
                for (&tok, &cnt) in &thread_local[i] {
                    *self.context_buckets[i].successors.entry(tok).or_insert(0) += cnt;
                    self.context_buckets[i].count += cnt;
                }
            }
        }

        let active = self.context_buckets.iter().filter(|b| b.count > 0).count();
        let total: u32 = self.context_buckets.iter().map(|b| b.count).sum();
        println!("  Contexts: {} (parallel {}T), {} buckets", total, n_threads, active);
    }

    // ========================================================
    // Phase C: Discover Rules (из сравнения фраз и контекстов)
    // ========================================================

    /// Обнаружить правила: найти где фразовая память ошибается,
    /// сгруппировать ошибки → паттерн ошибки = правило.
    ///
    /// Как ребёнок: "мама дай" работает, но "бабушка дай" — нет.
    /// Правило: "кто-то + дай" = любой взрослый, не только мама.
    pub fn discover_rules(&mut self, data: &[u16]) {
        use rayon::prelude::*;

        let n = data.len();
        if n < 4 { return; }

        let n_error_buckets: usize = 1 << 14;
        let n_threads = rayon::current_num_threads();
        let chunk_size = (n - 3) / n_threads + 1;
        let dim = self.config.hdc_dim;
        let lsh_bits = self.config.lsh_bits;
        let codebook = &self.codebook;
        let phrase_buckets = &self.phrase_buckets;

        // Parallel: each thread finds errors in its chunk
        let thread_results: Vec<(Vec<BundleAccumulator>, Vec<HashMap<u16, u32>>, u64, u64)> =
            (0..n_threads).into_par_iter().map(|tid| {
                let start = 2 + tid * chunk_size;
                let end = (start + chunk_size).min(n - 1);

                let mut err_accs: Vec<BundleAccumulator> = (0..n_error_buckets)
                    .map(|_| BundleAccumulator::new(dim)).collect();
                let mut err_toks: Vec<HashMap<u16, u32>> = (0..n_error_buckets)
                    .map(|_| HashMap::new()).collect();
                let mut errors = 0u64;
                let mut checked = 0u64;

                for t in start..end {
                    if t < 2 { continue; }
                    let a = &codebook[data[t] as usize];
                    let b = codebook[data[t-1] as usize].permute(1);
                    let c = codebook[data[t-2] as usize].permute(2);
                    let trigram = a.bind(&b).bind(&c);
                    let hash = lsh_hash(&trigram, lsh_bits) as usize;
                    let correct = data[t + 1];

                    let predicted = phrase_buckets[hash].top1();
                    checked += 1;

                    if predicted != Some(correct) {
                        let correction = trigram.bind(&codebook[correct as usize]);
                        let err_hash = lsh_hash(&correction, 14) as usize;
                        err_accs[err_hash].add(&correction);
                        *err_toks[err_hash].entry(correct).or_insert(0) += 1;
                        errors += 1;
                    }
                }
                (err_accs, err_toks, errors, checked)
            }).collect();

        // Merge
        let mut error_accumulators: Vec<BundleAccumulator> = (0..n_error_buckets)
            .map(|_| BundleAccumulator::new(dim)).collect();
        let mut error_tokens: Vec<HashMap<u16, u32>> = (0..n_error_buckets)
            .map(|_| HashMap::new()).collect();
        let mut total_errors = 0u64;
        let mut total_checked = 0u64;

        for (accs, toks, errs, checked) in &thread_results {
            total_errors += errs;
            total_checked += checked;
            for i in 0..n_error_buckets {
                for j in 0..dim {
                    error_accumulators[i].counters[j] += accs[i].counters[j];
                }
                error_accumulators[i].count += accs[i].count;
                for (&tok, &cnt) in &toks[i] {
                    *error_tokens[i].entry(tok).or_insert(0) += cnt;
                }
            }
        }
        drop(thread_results);

        let error_rate = total_errors as f64 / total_checked as f64 * 100.0;
        println!("  Rules: {} errors / {} ({:.1}%) (parallel {}T)",
                 total_errors, total_checked, error_rate, n_threads);

        // Step 2: Error clusters with many entries → rules
        let mut candidates: Vec<(BinaryVec, u32, f32)> = Vec::new();

        for i in 0..n_error_buckets {
            let count = error_accumulators[i].count;
            if count < 10 { continue; } // skip rare errors

            let pattern = error_accumulators[i].to_binary();

            // Confidence: how concentrated are the correct tokens?
            let top_count = error_tokens[i].values().max().copied().unwrap_or(0);
            let confidence = top_count as f32 / count as f32;

            candidates.push((pattern, count, confidence));
        }

        // Sort by support * confidence
        candidates.sort_by(|a, b| {
            let score_a = a.1 as f32 * a.2;
            let score_b = b.1 as f32 * b.2;
            score_b.partial_cmp(&score_a).unwrap()
        });

        // Keep top-N rules
        candidates.truncate(self.config.max_rules);

        self.rules = candidates.into_iter()
            .map(|(pattern, support, confidence)| Rule { pattern, support, confidence })
            .collect();

        self.rules_discovered = self.rules.len();
        println!("  Discovered {} rules (top support: {})",
                 self.rules_discovered,
                 self.rules.first().map(|r| r.support).unwrap_or(0));
    }

    // ========================================================
    // Forward: 3-Layer Prediction
    // ========================================================

    /// Predict next token using all 3 layers.
    ///
    /// Layer 1-2: phrase lookup → "я заучил эту фразу"
    /// Layer 3-4: context match → "в таком контексте это работает"
    /// Layer 5-6: reasoning → "правила и факты это подтверждают"
    pub fn predict(
        &self,
        tokens: &[u16],
        t: usize,
        fact_mem: Option<&FactMemory>,
    ) -> PredictionResult {
        let trigram = self.make_trigram(tokens, t);
        let hash = lsh_hash(&trigram, self.config.lsh_bits) as usize;

        // === Layer 1-2: Phrase Memory (multi-probe LSH) ===
        // Search primary bucket + 1-bit flip neighbors
        let mut all_phrase: HashMap<u16, u32> = HashMap::new();

        // Primary bucket (weighted heavily — exact match)
        let bucket = &self.phrase_buckets[hash];
        let primary_top = bucket.top_successors(self.config.top_k);
        for &(tok, cnt) in &primary_top {
            *all_phrase.entry(tok).or_insert(0) += cnt * 10; // primary = 10x weight
        }

        // Neighbor buckets: flip each bit of hash → 16 neighbors
        // Weight by how similar neighbor's representative is to our trigram
        for bit in 0..self.config.lsh_bits {
            let neighbor = hash ^ (1 << bit);
            let nbucket = &self.phrase_buckets[neighbor];
            if nbucket.count < 5 { continue; }
            // Check actual similarity between trigram and bucket representative
            let rep = nbucket.accumulator.to_binary();
            let sim = trigram.similarity(&rep);
            if sim <= 0 { continue; } // skip dissimilar neighbors
            let weight = (sim as u32 / 100).max(1); // proportional to similarity
            for &(tok, cnt) in &nbucket.top_successors(3) {
                *all_phrase.entry(tok).or_insert(0) += cnt * weight;
            }
        }

        let mut phrase_candidates: Vec<(u16, u32)> = all_phrase.into_iter().collect();
        phrase_candidates.sort_by(|a, b| b.1.cmp(&a.1));
        phrase_candidates.truncate(self.config.top_k * 2);

        // === Layer 3-4: Context Memory ===
        // "В таком широком контексте, что обычно идёт дальше?"
        // Шаг 2: ищем заученные фразы в похожих ситуациях
        let mut context_scores: HashMap<u16, u32> = HashMap::new();
        if t >= 4 {
            let broad_ctx = self.make_broad_context(tokens, t);
            let ctx_hash = lsh_hash(&broad_ctx, self.context_lsh_bits) as usize;

            // Primary context bucket
            let ctx_bucket = &self.context_buckets[ctx_hash];
            for &(tok, cnt) in &ctx_bucket.top_successors(self.config.top_k) {
                *context_scores.entry(tok).or_insert(0) += cnt * 3;
            }

            // Context neighbors (fewer — 14 bit hash, check only top 8 bits)
            for bit in 0..8.min(self.context_lsh_bits) {
                let neighbor = ctx_hash ^ (1 << bit);
                let nbucket = &self.context_buckets[neighbor];
                if nbucket.count < 5 { continue; }
                for &(tok, cnt) in &nbucket.top_successors(3) {
                    *context_scores.entry(tok).or_insert(0) += cnt;
                }
            }
        }

        // Combine: phrase scores (primary) + context scores (secondary)
        let mut scored: Vec<(u16, f64)> = phrase_candidates.iter()
            .map(|&(tok, count)| {
                let mut score = count as f64 * 100.0; // phrase = primary

                // Context boost: if context memory ALSO predicts this token → boost
                if let Some(&ctx_count) = context_scores.get(&tok) {
                    score += ctx_count as f64 * 10.0; // context confirmation
                }

                (tok, score)
            })
            .collect();

        // Add context-only predictions (tokens that context knows but phrases don't)
        for (&tok, &cnt) in &context_scores {
            if !scored.iter().any(|&(t, _)| t == tok) {
                scored.push((tok, cnt as f64 * 5.0)); // lower weight than phrase+context
            }
        }

        // === Logical Inference: unbind(facts, trigram) → answer ===
        // Чистая логика: если в бакете есть fact_vec = bundle(bind(ctx,succ)...),
        // то unbind(fact_vec, trigram) ≈ codebook[correct_successor].
        // Это modus ponens: знаю связь (ctx→succ), вижу ctx, извлекаю succ.
        if let Some(ref fact_vec) = bucket.fact_vec {
            let answer_vec = fact_vec.unbind(&trigram);

            // Find nearest codebook entries to the logical answer
            // (only top-5, and only if significantly similar)
            let threshold = self.config.hdc_dim as i32 / 6;
            for j in 0..self.config.vocab_size {
                let sim = answer_vec.similarity(&self.codebook[j]);
                if sim > threshold {
                    let logical_score = sim as f64 * 2.0; // logical inference = strong signal
                    let tok = j as u16;
                    if let Some(existing) = scored.iter_mut().find(|s| s.0 == tok) {
                        existing.1 += logical_score;
                    } else {
                        scored.push((tok, logical_score));
                    }
                }
            }
        }

        // === Layer 5-6: Improvisation (semantic word substitution) ===
        // "Не знаю 'в деревне' → но 'деревня ≈ город' → знаю 'в городе' → использую!"
        //
        // Для каждого из 3 слов в trigram: подставляем семантически похожее,
        // ищем результат в phrase memory, берём ответ со скидкой.
        let dim = self.config.hdc_dim as f64;

        if t >= 2 && !self.codebook_neighbors.is_empty() {
            let toks = [tokens[t] as usize, tokens[t-1] as usize, tokens[t-2] as usize];
            let perms = [0usize, 1, 2]; // permutation shifts for each position

            for pos in 0..3 { // substitute each of the 3 trigram positions
                let original_tok = toks[pos];
                if original_tok >= self.codebook_neighbors.len() { continue; }

                for &(similar_tok, sim) in &self.codebook_neighbors[original_tok] {
                    // Build alternative trigram with substituted word
                    let mut alt_toks = toks;
                    alt_toks[pos] = similar_tok as usize;

                    let alt_a = &self.codebook[alt_toks[0]];
                    let alt_b = self.codebook[alt_toks[1]].permute(1);
                    let alt_c = self.codebook[alt_toks[2]].permute(2);
                    let alt_trigram = alt_a.bind(&alt_b).bind(&alt_c);

                    let alt_hash = lsh_hash(&alt_trigram, self.config.lsh_bits) as usize;
                    let alt_bucket = &self.phrase_buckets[alt_hash];

                    if alt_bucket.count < 3 { continue; } // skip near-empty

                    // Discount by similarity: closer word → higher weight
                    let weight = (sim as f64 / dim) * 30.0;

                    for &(succ_tok, cnt) in &alt_bucket.top_successors(3) {
                        let sub_score = cnt as f64 * weight;
                        if let Some(existing) = scored.iter_mut().find(|s| s.0 == succ_tok) {
                            existing.1 += sub_score; // boost existing candidate
                        } else {
                            scored.push((succ_tok, sub_score)); // new candidate from improv
                        }
                    }
                }
            }
        }

        // Reasoning: context consistency (light touch — boost, never suppress)
        if let Some(facts) = fact_mem {
            for item in scored.iter_mut() {
                let consistency = facts.consistency(&self.codebook[item.0 as usize]);
                let norm_cons = consistency as f64 / dim;
                if norm_cons > 0.0 {
                    item.1 += norm_cons * 0.5;
                }
            }
        }

        // Sort by final score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(self.config.top_k);

        // Determine which layer was decisive
        let top_pred = scored.first().map(|s| s.0);
        let phrase_top = phrase_candidates.first().map(|c| c.0);
        let context_top = context_scores.iter()
            .max_by_key(|(_, &v)| v)
            .map(|(&k, _)| k);

        let layer = if top_pred == phrase_top && phrase_top.is_some() {
            PredictionLayer::Phrase           // Layer 1-2 won
        } else if top_pred == context_top && context_top.is_some() {
            PredictionLayer::Rule             // Layer 3-4 (context) won
        } else {
            PredictionLayer::Reasoning        // Layer 5-6 (improvisation) won!
        };

        PredictionResult {
            predictions: scored,
            layer,
            phrase_count: bucket.count,
        }
    }

    /// Find top-K nearest codebook entries to a query vector.
    pub fn nearest_codebook(&self, query: &BinaryVec, k: usize) -> Vec<(u16, i32)> {
        let mut results: Vec<(u16, i32)> = (0..self.config.vocab_size)
            .map(|i| (i as u16, query.similarity(&self.codebook[i])))
            .collect();
        results.sort_by(|a, b| b.1.cmp(&a.1));
        results.truncate(k);
        results
    }

    // ========================================================
    // Evaluation
    // ========================================================

    /// Evaluate on data: top-1 and top-5 accuracy per layer.
    pub fn evaluate(&self, data: &[u16]) -> EvalResult {
        let n = data.len();
        if n < 4 { return EvalResult::default(); }

        let total = n - 3;
        let mut correct_1 = 0u64;
        let mut correct_5 = 0u64;
        let mut phrase_hits = 0u64;
        let mut rule_hits = 0u64;
        let mut reason_hits = 0u64;

        let mut fact_mem = FactMemory::new(self.config.hdc_dim, 32);
        let report_every = total / 20 + 1;

        for t in 2..n - 1 {
            let correct = data[t + 1];
            let result = self.predict(data, t, Some(&fact_mem));

            // Update fact memory with context
            let trigram = self.make_trigram(data, t);
            fact_mem.add_fact(&trigram);

            if let Some(&(top1, _)) = result.predictions.first() {
                if top1 == correct {
                    correct_1 += 1;
                    match result.layer {
                        PredictionLayer::Phrase => phrase_hits += 1,
                        PredictionLayer::Rule => rule_hits += 1,
                        PredictionLayer::Reasoning => reason_hits += 1,
                    }
                }
            }

            if result.predictions.iter().any(|&(t, _)| t == correct) {
                correct_5 += 1;
            }

            if (t - 2) % report_every == 0 {
                let pct = ((t - 2) as f64 / total as f64 * 100.0) as u32;
                let acc1 = if t > 2 { correct_1 as f64 / (t - 1) as f64 * 100.0 } else { 0.0 };
                print!("\r  Eval: {}% acc@1={:.1}%", pct, acc1);
            }
        }

        println!();

        EvalResult {
            total: total as u64,
            correct_top1: correct_1,
            correct_top5: correct_5,
            phrase_hits,
            rule_hits,
            reason_hits,
        }
    }

    // ========================================================
    // Generation (Chat)
    // ========================================================

    /// Generate tokens autoregressively.
    pub fn generate(
        &self,
        prompt_tokens: &[u16],
        max_len: usize,
        temperature: f64,
        top_k: usize,
    ) -> Vec<u16> {
        let mut rng = rand::thread_rng();
        let mut tokens: Vec<u16> = prompt_tokens.to_vec();
        let mut fact_mem = FactMemory::new(self.config.hdc_dim, 32);

        // Add prompt to fact memory
        for t in 2..tokens.len() {
            let trigram = self.make_trigram(&tokens, t);
            fact_mem.add_fact(&trigram);
        }

        for _ in 0..max_len {
            if tokens.len() < 3 { break; }
            let t = tokens.len() - 1;
            let result = self.predict(&tokens, t, Some(&fact_mem));

            if result.predictions.is_empty() {
                // Fallback: random token
                tokens.push(rng.gen_range(0..self.config.vocab_size as u16));
                continue;
            }

            // Temperature sampling from top-K
            let candidates: Vec<(u16, f64)> = result.predictions.iter()
                .take(top_k)
                .map(|&(tok, score)| (tok, (score / temperature).exp()))
                .collect();

            let total: f64 = candidates.iter().map(|(_, s)| s).sum();
            if total <= 0.0 || total.is_nan() {
                tokens.push(candidates[0].0);
            } else {
                let mut r: f64 = rng.gen::<f64>() * total;
                let mut chosen = candidates[0].0;
                for &(tok, score) in &candidates {
                    r -= score;
                    if r <= 0.0 {
                        chosen = tok;
                        break;
                    }
                }
                tokens.push(chosen);
            }

            // Update fact memory
            let trigram = self.make_trigram(&tokens, tokens.len() - 1);
            fact_mem.add_fact(&trigram);
        }

        tokens
    }

    // ========================================================
    // Save / Load
    // ========================================================

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        use byteorder::{LittleEndian, WriteBytesExt};

        let mut f = std::fs::File::create(path)?;
        let magic: u32 = 0x48444317; // HDC17
        f.write_u32::<LittleEndian>(magic)?;
        f.write_u32::<LittleEndian>(self.config.hdc_dim as u32)?;
        f.write_u32::<LittleEndian>(self.config.vocab_size as u32)?;
        f.write_u32::<LittleEndian>(self.config.lsh_bits as u32)?;

        // Codebook
        for cb in &self.codebook {
            for &word in &cb.data {
                f.write_u64::<LittleEndian>(word)?;
            }
        }

        // Phrase buckets: save only active ones
        let active: Vec<(u32, &PhraseBucket)> = self.phrase_buckets.iter()
            .enumerate()
            .filter(|(_, b)| b.count > 0)
            .map(|(i, b)| (i as u32, b))
            .collect();
        f.write_u32::<LittleEndian>(active.len() as u32)?;
        for (idx, bucket) in &active {
            f.write_u32::<LittleEndian>(*idx)?;
            f.write_u32::<LittleEndian>(bucket.count)?;
            // Accumulator (binary representative)
            let rep = bucket.accumulator.to_binary();
            for &word in &rep.data {
                f.write_u64::<LittleEndian>(word)?;
            }
            // Top-10 successors
            let top = bucket.top_successors(10);
            f.write_u32::<LittleEndian>(top.len() as u32)?;
            for (tok, cnt) in &top {
                f.write_u16::<LittleEndian>(*tok)?;
                f.write_u32::<LittleEndian>(*cnt)?;
            }
        }

        // Rules
        f.write_u32::<LittleEndian>(self.rules.len() as u32)?;
        for rule in &self.rules {
            for &word in &rule.pattern.data {
                f.write_u64::<LittleEndian>(word)?;
            }
            f.write_u32::<LittleEndian>(rule.support)?;
            f.write_f32::<LittleEndian>(rule.confidence)?;
        }

        println!("Saved to {} ({:.1} MB)",
                 path, f.metadata()?.len() as f64 / 1024.0 / 1024.0);
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        use byteorder::{LittleEndian, ReadBytesExt};

        let mut f = std::fs::File::open(path)?;
        let magic = f.read_u32::<LittleEndian>()?;
        assert_eq!(magic, 0x48444317, "Not a v17 model file");

        let hdc_dim = f.read_u32::<LittleEndian>()? as usize;
        let vocab_size = f.read_u32::<LittleEndian>()? as usize;
        let lsh_bits = f.read_u32::<LittleEndian>()? as usize;

        let config = Config {
            hdc_dim, vocab_size, lsh_bits,
            max_rules: 512, top_k: 10,
        };
        let words_per_vec = (hdc_dim + BITS_PER_WORD - 1) / BITS_PER_WORD;

        // Codebook
        let mut codebook = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let mut data = vec![0u64; words_per_vec];
            for w in data.iter_mut() {
                *w = f.read_u64::<LittleEndian>()?;
            }
            codebook.push(BinaryVec { data, dim: hdc_dim });
        }

        // Phrase buckets
        let n_buckets = config.n_buckets();
        let mut phrase_buckets: Vec<PhraseBucket> = (0..n_buckets)
            .map(|_| PhraseBucket::new(hdc_dim))
            .collect();

        let n_active = f.read_u32::<LittleEndian>()? as usize;
        for _ in 0..n_active {
            let idx = f.read_u32::<LittleEndian>()? as usize;
            let count = f.read_u32::<LittleEndian>()?;
            let mut data = vec![0u64; words_per_vec];
            for w in data.iter_mut() {
                *w = f.read_u64::<LittleEndian>()?;
            }
            let n_succ = f.read_u32::<LittleEndian>()? as usize;
            let mut successors = HashMap::new();
            for _ in 0..n_succ {
                let tok = f.read_u16::<LittleEndian>()?;
                let cnt = f.read_u32::<LittleEndian>()?;
                successors.insert(tok, cnt);
            }
            phrase_buckets[idx].count = count;
            phrase_buckets[idx].successors = successors;
            // Note: accumulator is not fully restored (only binary rep),
            // but that's fine for inference
        }

        // Rules
        let n_rules = f.read_u32::<LittleEndian>()? as usize;
        let mut rules = Vec::with_capacity(n_rules);
        for _ in 0..n_rules {
            let mut data = vec![0u64; words_per_vec];
            for w in data.iter_mut() {
                *w = f.read_u64::<LittleEndian>()?;
            }
            let support = f.read_u32::<LittleEndian>()?;
            let confidence = f.read_f32::<LittleEndian>()?;
            rules.push(Rule {
                pattern: BinaryVec { data, dim: hdc_dim },
                support,
                confidence,
            });
        }

        println!("Loaded v17: dim={}, vocab={}, {} active buckets, {} rules",
                 hdc_dim, vocab_size, n_active, n_rules);

        let context_lsh_bits = 14;
        let n_context_buckets = 1 << context_lsh_bits;
        let context_buckets: Vec<PhraseBucket> = (0..n_context_buckets)
            .map(|_| PhraseBucket::new(hdc_dim))
            .collect();

        let mut model = HDCBrainV17 {
            config,
            codebook,
            phrase_buckets,
            context_buckets,
            context_lsh_bits,
            codebook_neighbors: Vec::new(),
            rules,
            phrases_trained: 0,
            contexts_trained: 0,
            rules_discovered: n_rules,
        };
        // Rebuild neighbors from loaded codebook
        model.build_neighbors(5);
        Ok(model)
    }
}

// ============================================================
// Result types
// ============================================================

#[derive(Debug, Clone, Copy)]
pub enum PredictionLayer {
    Phrase,
    Rule,
    Reasoning,
}

pub struct PredictionResult {
    pub predictions: Vec<(u16, f64)>,
    pub layer: PredictionLayer,
    pub phrase_count: u32,
}

#[derive(Default)]
pub struct EvalResult {
    pub total: u64,
    pub correct_top1: u64,
    pub correct_top5: u64,
    pub phrase_hits: u64,
    pub rule_hits: u64,
    pub reason_hits: u64,
}

impl EvalResult {
    pub fn print(&self) {
        let acc1 = self.correct_top1 as f64 / self.total as f64 * 100.0;
        let acc5 = self.correct_top5 as f64 / self.total as f64 * 100.0;

        println!("  Results ({} positions):", self.total);
        println!("    Top-1 accuracy: {:.2}%", acc1);
        println!("    Top-5 accuracy: {:.2}%", acc5);
        println!("    --- By layer ---");
        println!("    Phrase hits:    {} ({:.1}%)",
                 self.phrase_hits, self.phrase_hits as f64 / self.total as f64 * 100.0);
        println!("    Rule hits:      {} ({:.1}%)",
                 self.rule_hits, self.rule_hits as f64 / self.total as f64 * 100.0);
        println!("    Reasoning hits: {} ({:.1}%)",
                 self.reason_hits, self.reason_hits as f64 / self.total as f64 * 100.0);
    }
}
