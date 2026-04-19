//! HDC-Brain v18: Conscious HDC Model — Worm-Driven Learning
//!
//! П.2: Каскадный predict через Working Memory
//! Обучение = черви (Collector/Analyst/Skeptic/Explorer)
//! Working Memory = временная область куда черви складывают находки

use crate::memory::*;
use crate::language::*;
use crate::worms::*;
use crate::working_memory::*;
use crate::worm_mind::*;
use crate::logger::*;
use std::collections::HashMap;

pub struct Config {
    pub hdc_dim: usize,
    pub vocab_size: usize,
    pub codebook_window: usize,
    pub chunk_size: usize,
    pub sample_rate: usize,
    pub worm_checks: usize,
    pub n_passes: usize,
}

impl Config {
    pub fn default_with_vocab(vocab_size: usize) -> Self {
        Config {
            hdc_dim: 4096,
            vocab_size,
            codebook_window: 3,
            chunk_size: 100_000,
            sample_rate: 1,
            worm_checks: 10_000,
            n_passes: 1,
        }
    }
}

pub struct HDCBrainV18 {
    pub config: Config,
    pub memory: HierarchicalMemory,
    pub lang: LogicLanguage,
    pub attn_roles: crate::worm_mind::AttentionRoles,
    pub hdc_memory: crate::worm_mind::HDCMemory,
    pub background: crate::worm_mind::BackgroundMind,
}

impl HDCBrainV18 {
    pub fn new(config: Config) -> Self {
        let dim = config.hdc_dim;
        HDCBrainV18 {
            attn_roles: crate::worm_mind::AttentionRoles::new(dim),
            hdc_memory: crate::worm_mind::HDCMemory::new(dim),
            memory: HierarchicalMemory::new(dim, config.vocab_size),
            lang: LogicLanguage::new(dim),
            background: crate::worm_mind::BackgroundMind::new(),
            config,
        }
    }

    // ============================================================
    // Phase 0: Semantic Codebook
    // ============================================================

    pub fn learn_codebook(&mut self, tokens: &[u16]) {
        println!("Phase 0: Building semantic codebook...");
        let start = std::time::Instant::now();
        self.memory.build_semantic_codebook(tokens, self.config.codebook_window);
        println!("  Codebook built in {:.1}s ({} words, dim={})",
            start.elapsed().as_secs_f64(), self.config.vocab_size, self.config.hdc_dim);
    }

    pub fn show_neighbors(&self, vocab: &[String], sample_tokens: &[u16]) {
        println!("\n  Semantic neighbors:");
        for &tok in sample_tokens.iter().take(5) {
            if (tok as usize) >= self.config.vocab_size { continue; }
            let word = &vocab[tok as usize];
            let neighbors = self.memory.nearest_words(&self.memory.words[tok as usize], 6);
            let neighbor_words: Vec<String> = neighbors.iter()
                .filter(|(t, _)| *t != tok)
                .take(5)
                .map(|(t, s)| format!("{}({})", &vocab[*t as usize], s))
                .collect();
            println!("    \"{}\" → [{}]", word, neighbor_words.join(", "));
        }
    }

    // ============================================================
    // Phase 1: Worm Training
    // ============================================================

    pub fn train(&mut self, tokens: &[u16], vocab: Option<&[String]>, log: Option<&Logger>) {
        let n_passes = self.config.n_passes;
        let start = std::time::Instant::now();

        let msg = format!("Worm Training ({} passes), chunk={}, sample_rate={}, worm_checks={}",
            n_passes, self.config.chunk_size, self.config.sample_rate, self.config.worm_checks);
        if let Some(l) = log { l.section(&msg); } else { println!("{}", msg); }

        let orch = WormOrchestrator::new(
            self.config.chunk_size,
            self.config.sample_rate,
            self.config.worm_checks,
        );

        for pass in 0..n_passes {
            let pass_msg = format!("--- Pass {}/{} ---", pass + 1, n_passes);
            if let Some(l) = log { l.log(&pass_msg); } else { println!("{}", pass_msg); }

            let report = orch.run_pass(
                &mut self.memory, &self.lang, tokens, vocab, log,
            );

            let report_str = report.to_string();
            if let Some(l) = log { l.log(&report_str); } else { report.print(); }

            let eval_msg = "Evaluating with WormMind...";
            if let Some(l) = log { l.log(eval_msg); } else { println!("{}", eval_msg); }

            let eval = self.evaluate_mind(tokens, 5000);

            if let Some(l) = log {
                l.metric_pct("Top-1", eval.correct_top1 as f64 / eval.tested.max(1) as f64 * 100.0);
                l.metric_pct("Top-5", eval.correct_top5 as f64 / eval.tested.max(1) as f64 * 100.0);
                l.metric("avg_reasoning", eval.avg_reasoning);
                l.metric("avg_confidence", eval.avg_confidence);
                l.metric("background_updates", eval.total_bg_updates as f64);
                l.metric_pct("no_trigram_match", eval.pct_no_trigram);
                l.metric("avg_uncertain", eval.avg_uncertain);
                l.metric("avg_contradictions", eval.avg_contradictions);
                l.metric("need_more_data", eval.total_need_more as f64);
                l.metric("relations", eval.relations_count as f64);
                l.metric("bg_relations_created", eval.bg_relations_created as f64);
                l.metric("influence_relations_boosted", eval.relations_boosted as f64);
                l.metric("influence_cross_worm", eval.cross_worm_overrides as f64);
                l.metric("influence_causal_extra", eval.causal_extra_tokens as f64);
                l.metric("influence_awareness", eval.awareness_adaptations as f64);
                l.metric("influence_evidence_reranks", eval.evidence_reranks as f64);
            } else {
                eval.print();
            }

            let stats = format!("{} | bigram_idx={} | token_idx={} | time={:.1}s",
                self.memory.stats(),
                self.memory.facts.bigram_index.len(),
                self.memory.facts.token_index.len(),
                start.elapsed().as_secs_f64());
            if let Some(l) = log { l.log(&stats); } else { println!("  {}", stats); }

            // Обновить самопознание из eval результатов
            let t = eval.tested.max(1) as f64;
            self.background.self_knowledge.observed_accuracy = eval.correct_top1 as f64 / t;
            self.background.self_knowledge.avg_confidence = eval.avg_confidence;
            self.background.self_knowledge.total_predicts += eval.tested as u64;

            // Самопознание: вывод self-description
            let self_desc = self.background.self_knowledge.describe();
            if let Some(l) = log { l.log(&self_desc); } else { println!("  {}", self_desc); }

            // САМОМОДИФИКАЦИЯ: черви анализируют свою accuracy и адаптируют параметры
            let acc = eval.correct_top1 as f64 / eval.tested.max(1) as f64;
            if let Some(modify_msg) = self.attn_roles.awareness.self_modify(acc, eval.avg_confidence) {
                if let Some(l) = log { l.log(&modify_msg); } else { println!("  {}", modify_msg); }
            }
        }

        let done = format!("Training Complete ({:.1}s) | {}", start.elapsed().as_secs_f64(), self.memory.stats());
        if let Some(l) = log { l.section(&done); } else { println!("{}", done); }
    }

    // ============================================================
    // П.2: Каскадный Predict с fallback
    // ============================================================

    /// Каскадное предсказание:
    ///   1. Trigram (3 tok) — точное совпадение, вес 1.0
    ///   2. Bigram (2 tok)  — fallback по биграмме, вес 0.5
    ///   3. Unigram (1 tok) — fallback по униграмме, вес 0.2
    ///   4. Abstractions     — тематические подсказки, вес 0.1
    pub fn predict(&self, t0: u16, t1: u16, t2: u16) -> Vec<(u16, f64)> {
        let mut scores: HashMap<u16, f64> = HashMap::new();
        let dim = self.config.hdc_dim;

        // === 1. Trigram (вес 1.0) ===
        let trigram = self.memory.make_trigram_query(t0, t1, t2);
        let top_facts = self.memory.facts.find_top_k(&trigram, 20);
        let mut trigram_found = false;
        for (idx, sim) in &top_facts {
            if *sim <= 0 { continue; }
            trigram_found = true;
            let entry = &self.memory.facts.entries[*idx];
            let sim_weight = *sim as f64 / dim as f64;
            for (tok, cnt) in entry.successors.iter() {
                *scores.entry(*tok).or_insert(0.0) +=
                    sim_weight * (*cnt as f64) / entry.total_count.max(1) as f64;
            }
        }

        // === 2. Bigram fallback (вес 0.5) ===
        let bigram_indices = self.memory.facts.find_by_bigram(t1, t2, 50);
        for idx in &bigram_indices {
            let entry = &self.memory.facts.entries[*idx];
            for (tok, cnt) in entry.successors.iter() {
                *scores.entry(*tok).or_insert(0.0) +=
                    0.5 * (*cnt as f64) / entry.total_count.max(1) as f64;
            }
        }

        // Также проверить bigram rules от Explorer
        if !self.memory.rules.entries.is_empty() {
            let bigram_query = self.memory.make_bigram_query(t1, t2);
            let top_rules = self.memory.rules.find_top_k(&bigram_query, 10);
            for (idx, sim) in &top_rules {
                if *sim <= 0 { continue; }
                let entry = &self.memory.rules.entries[*idx];
                let sim_weight = *sim as f64 / dim as f64;
                for (tok, cnt) in entry.successors.iter() {
                    *scores.entry(*tok).or_insert(0.0) +=
                        sim_weight * 0.4 * (*cnt as f64) / entry.total_count.max(1) as f64;
                }
            }
        }

        // === 3. Unigram fallback (вес 0.2) — если триграм не нашёл ===
        if !trigram_found {
            let unigram_indices = self.memory.facts.find_by_token(t2, 30);
            for idx in &unigram_indices {
                let entry = &self.memory.facts.entries[*idx];
                for (tok, cnt) in entry.successors.iter() {
                    *scores.entry(*tok).or_insert(0.0) +=
                        0.2 * (*cnt as f64) / entry.total_count.max(1) as f64;
                }
            }
        }

        // === 4. Abstractions (вес 0.1) ===
        if !self.memory.abstractions.entries.is_empty() {
            let top_wide = self.memory.abstractions.find_top_k(&trigram, 5);
            for (idx, sim) in &top_wide {
                if *sim <= 0 { continue; }
                let entry = &self.memory.abstractions.entries[*idx];
                let sim_weight = *sim as f64 / dim as f64;
                for (tok, cnt) in entry.successors.iter() {
                    *scores.entry(*tok).or_insert(0.0) +=
                        sim_weight * 0.1 * (*cnt as f64) / entry.total_count.max(1) as f64;
                }
            }
        }

        let mut result: Vec<(u16, f64)> = scores.into_iter().collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result.truncate(10);
        result
    }

    // ============================================================
    // Evaluation
    // ============================================================

    /// Evaluation через Working Memory — с расширенными метриками
    pub fn evaluate_wm(&self, tokens: &[u16], n_positions: usize) -> EvalResultWM {
        let total = tokens.len().saturating_sub(4);
        let step = if total > n_positions { total / n_positions } else { 1 };

        let mut correct_top1 = 0u32;
        let mut correct_top5 = 0u32;
        let mut tested = 0u32;
        let mut total_evidence = 0usize;
        let mut total_sources = 0usize;
        let mut total_confidence = 0.0f64;

        for i in (2..tokens.len().saturating_sub(1)).step_by(step) {
            if tested >= n_positions as u32 { break; }

            let t0 = tokens[i - 2];
            let t1 = tokens[i - 1];
            let t2 = tokens[i];
            let expected = tokens[i + 1];

            if t0 as usize >= self.config.vocab_size
                || t1 as usize >= self.config.vocab_size
                || t2 as usize >= self.config.vocab_size
                || expected as usize >= self.config.vocab_size
            { continue; }

            let wm_result = WorkingMemory::full_pipeline(
                &self.memory, tokens, i, self.config.vocab_size, self.config.hdc_dim,
            );
            tested += 1;
            total_evidence += wm_result.total_evidence;
            total_sources += wm_result.sources_used.len();
            total_confidence += wm_result.confidence;

            if let Some((top1_tok, _)) = wm_result.predictions.first() {
                if *top1_tok == expected { correct_top1 += 1; }
            }
            if wm_result.predictions.iter().take(5).any(|(t, _)| *t == expected) {
                correct_top5 += 1;
            }
        }

        let t = tested.max(1) as f64;
        EvalResultWM {
            tested,
            correct_top1,
            correct_top5,
            avg_evidence: total_evidence as f64 / t,
            avg_sources: total_sources as f64 / t,
            avg_confidence: total_confidence / t,
        }
    }

    /// Evaluation через WormMind during training: codebook=false (degrades!), bg loop active
    pub fn evaluate_mind(&mut self, tokens: &[u16], n_positions: usize) -> EvalResultMind {
        self.evaluate_mind_inner(tokens, n_positions, false, true)
    }

    /// Readonly evaluation: no codebook, no relations, no background loop
    pub fn evaluate_mind_readonly(&mut self, tokens: &[u16], n_positions: usize) -> EvalResultMind {
        self.evaluate_mind_inner(tokens, n_positions, false, false)
    }

    fn evaluate_mind_inner(&mut self, tokens: &[u16], n_positions: usize, learn_codebook: bool, allow_background: bool) -> EvalResultMind {
        let total = tokens.len().saturating_sub(4);
        let step = if total > n_positions { total / n_positions } else { 1 };

        let mut correct_top1 = 0u32;
        let mut correct_top5 = 0u32;
        let mut tested = 0u32;
        let mut total_reasoning = 0usize;
        let mut total_thoughts = 0usize;
        let mut total_confidence = 0.0f64;
        let mut total_bg_updates = 0usize;
        let mut total_no_trigram = 0usize;
        // Influence aggregation
        let mut total_relations_boosted = 0usize;
        let mut total_cross_worm = 0usize;
        let mut total_causal_extra = 0usize;
        let mut total_awareness_adapt = 0usize;
        let mut total_evidence_reranks = 0usize;
        let mut total_uncertain = 0usize;
        let mut total_contradictions = 0usize;
        let mut total_need_more = 0usize;
        let mut bg_relations_created = 0usize;

        for i in (2..tokens.len().saturating_sub(1)).step_by(step) {
            if tested >= n_positions as u32 { break; }

            let expected = tokens[i + 1];
            if (tokens[i - 2] as usize) >= self.config.vocab_size
                || (tokens[i - 1] as usize) >= self.config.vocab_size
                || (tokens[i] as usize) >= self.config.vocab_size
                || (expected as usize) >= self.config.vocab_size
            { continue; }

            let result = WormMind::think(
                &mut self.memory, &self.lang, &mut self.attn_roles, &mut self.hdc_memory,
                tokens, i, self.config.vocab_size, Some(expected), learn_codebook,
            );
            tested += 1;
            total_reasoning += result.reasoning_depth;
            total_thoughts += result.thoughts_exchanged;
            total_confidence += result.confidence;
            total_bg_updates += result.background_updates;

            if let Some((top1_tok, _)) = result.predictions.first() {
                if *top1_tok == expected { correct_top1 += 1; }
            }
            if result.predictions.iter().take(5).any(|(t, _)| *t == expected) {
                correct_top5 += 1;
            }

            // Агрегация influence (аудит реального влияния)
            total_relations_boosted += result.influence.relations_boosted;
            total_cross_worm += result.influence.cross_worm_overrides;
            total_causal_extra += result.influence.causal_extra_tokens;
            total_awareness_adapt += result.influence.awareness_adaptations;
            total_evidence_reranks += result.influence.evidence_reranks;

            // Агрегация диагностики
            total_no_trigram += if result.diagnostic.no_trigram_match { 1 } else { 0 };
            total_uncertain += result.diagnostic.uncertain_areas;
            total_contradictions += result.diagnostic.contradictions;
            total_need_more += result.diagnostic.need_more_data.len();

            // Background Loop: каждые 1000 predict'ов (не в readonly mode)
            if allow_background && tested % 1000 == 0 && tested > 0 {
                let bg_result = self.background.run_cycle(&mut self.memory, &self.lang, None);
                bg_relations_created += bg_result.relations_created;
                total_bg_updates += bg_result.relations_created;
                // Логируем сообщения от червей
                for msg in &bg_result.messages {
                    match msg {
                        crate::worm_mind::WormMessage::Discovery(s) => eprintln!("  🔬 {}", s),
                        crate::worm_mind::WormMessage::NeedData(s) => eprintln!("  📊 {}", s),
                        crate::worm_mind::WormMessage::Confused(s) => eprintln!("  ❓ {}", s),
                        crate::worm_mind::WormMessage::Ask(s) => eprintln!("  💬 {}", s),
                        crate::worm_mind::WormMessage::Report(s) => eprintln!("  📋 {}", s),
                    }
                }
            }
        }

        let t = tested.max(1) as f64;
        EvalResultMind {
            tested,
            correct_top1,
            correct_top5,
            avg_reasoning: total_reasoning as f64 / t,
            avg_thoughts: total_thoughts as f64 / t,
            avg_confidence: total_confidence / t,
            total_bg_updates,
            pct_no_trigram: total_no_trigram as f64 / t * 100.0,
            avg_uncertain: total_uncertain as f64 / t,
            avg_contradictions: total_contradictions as f64 / t,
            total_need_more,
            relations_count: self.memory.relations.len(),
            bg_relations_created,
            relations_boosted: total_relations_boosted,
            cross_worm_overrides: total_cross_worm,
            causal_extra_tokens: total_causal_extra,
            awareness_adaptations: total_awareness_adapt,
            evidence_reranks: total_evidence_reranks,
        }
    }
}

pub struct EvalResultWM {
    pub tested: u32,
    pub correct_top1: u32,
    pub correct_top5: u32,
    pub avg_evidence: f64,
    pub avg_sources: f64,
    pub avg_confidence: f64,
}

impl EvalResultWM {
    pub fn print(&self) {
        let t = self.tested.max(1) as f64;
        println!("  Results ({} positions):", self.tested);
        println!("    Top-1: {:.2}%", self.correct_top1 as f64 / t * 100.0);
        println!("    Top-5: {:.2}%", self.correct_top5 as f64 / t * 100.0);
        println!("    Working Memory:");
        println!("      avg evidence: {:.1}", self.avg_evidence);
        println!("      avg sources:  {:.1}", self.avg_sources);
        println!("      avg confidence: {:.3}", self.avg_confidence);
    }
}

pub struct EvalResultMind {
    pub tested: u32,
    pub correct_top1: u32,
    pub correct_top5: u32,
    pub avg_reasoning: f64,
    pub avg_thoughts: f64,
    pub avg_confidence: f64,
    pub total_bg_updates: usize,
    // Самодиагностика
    pub pct_no_trigram: f64,
    pub avg_uncertain: f64,
    pub avg_contradictions: f64,
    pub total_need_more: usize,
    // Relations & Background
    pub relations_count: usize,
    pub bg_relations_created: usize,
    // Influence audit — PROOF that features are not decoration
    pub relations_boosted: usize,
    pub cross_worm_overrides: usize,
    pub causal_extra_tokens: usize,
    pub awareness_adaptations: usize,
    pub evidence_reranks: usize,
}

impl EvalResultMind {
    pub fn print(&self) {
        let t = self.tested.max(1) as f64;
        println!("  Results ({} positions):", self.tested);
        println!("    Top-1: {:.2}%", self.correct_top1 as f64 / t * 100.0);
        println!("    Top-5: {:.2}%", self.correct_top5 as f64 / t * 100.0);
        println!("    WormMind:");
        println!("      avg reasoning steps: {:.1}", self.avg_reasoning);
        println!("      avg thoughts:        {:.1}", self.avg_thoughts);
        println!("      avg confidence:       {:.3}", self.avg_confidence);
        println!("      background updates:   {}", self.total_bg_updates);
        println!("    Self-Diagnostic:");
        println!("      no trigram match:     {:.1}%", self.pct_no_trigram);
        println!("      avg uncertain:        {:.1}", self.avg_uncertain);
        println!("      avg contradictions:   {:.1}", self.avg_contradictions);
        println!("      need more data:       {}", self.total_need_more);
        println!("    Relations & Background:");
        println!("      relations total:      {}", self.relations_count);
        println!("      bg relations created: {}", self.bg_relations_created);
        println!("    Influence Audit (proof of non-decoration):");
        println!("      relations boosted:    {}", self.relations_boosted);
        println!("      cross-worm overrides: {}", self.cross_worm_overrides);
        println!("      causal extra tokens:  {}", self.causal_extra_tokens);
        println!("      awareness adaptations:{}", self.awareness_adaptations);
        println!("      evidence reranks:     {}", self.evidence_reranks);
    }
}
