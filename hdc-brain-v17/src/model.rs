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
}

impl PhraseBucket {
    pub fn new(dim: usize) -> Self {
        PhraseBucket {
            accumulator: BundleAccumulator::new(dim),
            successors: HashMap::new(),
            count: 0,
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

    // Layer 1-2: Phrase Memory (LSH buckets)
    pub phrase_buckets: Vec<PhraseBucket>,

    // Layer 3-4: Discovered Rules
    pub rules: Vec<Rule>,

    // Stats
    pub phrases_trained: usize,
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

        // Empty phrase buckets
        let phrase_buckets: Vec<PhraseBucket> = (0..n_buckets)
            .map(|_| PhraseBucket::new(config.hdc_dim))
            .collect();

        HDCBrainV17 {
            config,
            codebook,
            phrase_buckets,
            rules: Vec::new(),
            phrases_trained: 0,
            rules_discovered: 0,
        }
    }

    /// Build trigram: bind(tok[t], perm¹(tok[t-1]), perm²(tok[t-2]))
    #[inline]
    pub fn make_trigram(&self, tokens: &[u16], t: usize) -> BinaryVec {
        let a = &self.codebook[tokens[t] as usize];
        let b = self.codebook[tokens[t - 1] as usize].permute(1);
        let c = self.codebook[tokens[t - 2] as usize].permute(2);
        a.bind(&b).bind(&c)
    }

    // ========================================================
    // Phase 1: Learn Phrases (один проход по данным)
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

    // ========================================================
    // Phase 2: Discover Rules (из ошибок фразовой памяти)
    // ========================================================

    /// Обнаружить правила: найти где фразовая память ошибается,
    /// сгруппировать ошибки → паттерн ошибки = правило.
    ///
    /// Как ребёнок: "мама дай" работает, но "бабушка дай" — нет.
    /// Правило: "кто-то + дай" = любой взрослый, не только мама.
    pub fn discover_rules(&mut self, data: &[u16]) {
        let n = data.len();
        if n < 4 { return; }

        // Step 1: Find positions where phrase memory is wrong
        // For each error, compute correction = bind(trigram, codebook[correct])
        // LSH-hash corrections → error clusters
        let n_error_buckets: usize = 1 << 14; // 16384
        let mut error_accumulators: Vec<BundleAccumulator> = (0..n_error_buckets)
            .map(|_| BundleAccumulator::new(self.config.hdc_dim))
            .collect();
        let mut error_tokens: Vec<HashMap<u16, u32>> = (0..n_error_buckets)
            .map(|_| HashMap::new())
            .collect();

        let mut total_errors = 0u64;
        let mut total_checked = 0u64;
        let report_every = (n - 3) / 10 + 1;

        for t in 2..n - 1 {
            let trigram = self.make_trigram(data, t);
            let hash = lsh_hash(&trigram, self.config.lsh_bits) as usize;
            let correct = data[t + 1];

            // What does phrase memory predict?
            let predicted = self.phrase_buckets[hash].top1();
            total_checked += 1;

            if predicted != Some(correct) {
                // Error! Compute correction vector
                let correction = trigram.bind(&self.codebook[correct as usize]);
                let err_hash = lsh_hash(&correction, 14) as usize;
                error_accumulators[err_hash].add(&correction);
                *error_tokens[err_hash].entry(correct).or_insert(0) += 1;
                total_errors += 1;
            }

            if (t - 2) % report_every == 0 {
                let pct = ((t - 2) as f64 / (n - 3) as f64 * 100.0) as u32;
                print!("\r  Rules: {}% errors={}", pct, total_errors);
            }
        }

        let error_rate = total_errors as f64 / total_checked as f64 * 100.0;
        println!("\r  Rule discovery: {} errors / {} checked ({:.1}%)",
                 total_errors, total_checked, error_rate);

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
    /// Layer 1-2: phrase lookup → "я видел эту фразу"
    /// Layer 3-4: rule application → "тут работает правило"
    /// Layer 5-6: reasoning → "в контексте это логично"
    pub fn predict(
        &self,
        tokens: &[u16],
        t: usize,
        fact_mem: Option<&FactMemory>,
    ) -> PredictionResult {
        let trigram = self.make_trigram(tokens, t);
        let hash = lsh_hash(&trigram, self.config.lsh_bits) as usize;

        // === Layer 1-2: Phrase Memory ===
        let bucket = &self.phrase_buckets[hash];
        let phrase_candidates = bucket.top_successors(self.config.top_k);

        // === Layer 3-4: Rule Application ===
        // For each candidate, check how well rules support it
        let mut scored: Vec<(u16, f64)> = phrase_candidates.iter()
            .map(|&(tok, count)| {
                let mut score = count as f64;

                // Apply each rule: unbind(trigram, rule) → how similar to codebook[tok]?
                for rule in &self.rules {
                    let conclusion = trigram.unbind(&rule.pattern);
                    let sim = conclusion.similarity(&self.codebook[tok as usize]);
                    // Positive sim → rule supports this token
                    score += sim as f64 * rule.confidence as f64 * 0.01;
                }

                (tok, score)
            })
            .collect();

        // Also check: do any rules predict tokens NOT in phrase candidates?
        // (This is the "improvisation" part — rules can suggest new tokens)
        for rule in self.rules.iter().take(20) { // top-20 rules only
            let conclusion = trigram.unbind(&rule.pattern);
            // Find nearest codebook entry
            let (best_tok, best_sim) = self.nearest_codebook(&conclusion, 1)[0];
            if best_sim > self.config.hdc_dim as i32 / 8 { // significant similarity
                if !scored.iter().any(|&(t, _)| t == best_tok) {
                    scored.push((best_tok, best_sim as f64 * rule.confidence as f64 * 0.005));
                }
            }
        }

        // === Layer 5-6: Reasoning (context consistency) ===
        if let Some(facts) = fact_mem {
            for item in scored.iter_mut() {
                let consistency = facts.consistency(&self.codebook[item.0 as usize]);
                // Boost tokens consistent with recent context
                item.1 += consistency as f64 * 0.1;
            }
        }

        // Sort by final score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(self.config.top_k);

        // Determine which layer was decisive
        let layer = if bucket.count > 0 && !phrase_candidates.is_empty() {
            if scored[0].0 == phrase_candidates[0].0 {
                PredictionLayer::Phrase
            } else {
                PredictionLayer::Rule
            }
        } else if !self.rules.is_empty() {
            PredictionLayer::Rule
        } else {
            PredictionLayer::Reasoning
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

        Ok(HDCBrainV17 {
            config,
            codebook,
            phrase_buckets,
            rules,
            phrases_trained: 0,
            rules_discovered: n_rules,
        })
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
