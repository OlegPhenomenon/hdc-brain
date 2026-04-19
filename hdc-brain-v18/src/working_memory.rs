//! Working Memory — временная рабочая область для формирования ответа
//!
//! Как short-term memory у человека:
//! 1. Черви получают контекст
//! 2. Каждый червь добавляет находки (Evidence) в рабочую память
//! 3. Самопроверка: конфликтные улики проверяются повторно
//! 4. Финализация: рабочая память выдаёт ранжированный ответ
//!
//! Working Memory НЕ хранится между predict-вызовами — создаётся заново каждый раз.

use crate::binary::*;
use crate::memory::*;
use std::collections::HashMap;

/// Одна улика — факт найденный червём
#[derive(Clone, Debug)]
pub struct Evidence {
    pub token: u16,
    pub raw_score: f64,       // чистый скор без base_weight (cnt/total * sim_weight)
    pub source: EvidenceSource,
    pub verified: bool,
    pub confidence: f64,      // 0.0..1.0 — уверенность червя в этой улике
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EvidenceSource {
    Trigram,        // точное совпадение триграммы
    Bigram,         // fallback по биграмме
    BigramRule,     // правило Explorer-а
    Unigram,        // fallback по униграмме
    Rule,           // medium context rule
    Abstraction,    // широкий контекст / wide context
    Context,        // attention/context bundle
    Semantic,       // семантический поиск по контекстному вектору
    Verification,   // добавлено при самопроверке
}

impl EvidenceSource {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Trigram => "trigram",
            Self::Bigram => "bigram",
            Self::BigramRule => "bigram_rule",
            Self::Unigram => "unigram",
            Self::Rule => "rule",
            Self::Abstraction => "abstraction",
            Self::Context => "context",
            Self::Semantic => "semantic",
            Self::Verification => "verification",
        }
    }

    /// Базовый вес источника — применяется ОДИН РАЗ в finalize()
    pub fn base_weight(&self) -> f64 {
        match self {
            Self::Trigram => 1.0,
            Self::Bigram => 0.5,
            Self::BigramRule => 0.4,
            Self::Unigram => 0.2,
            Self::Rule => 0.3,
            Self::Abstraction => 0.15,
            Self::Context => 0.3,
            Self::Semantic => 0.25,
            Self::Verification => 0.5,
        }
    }
}

/// Рабочая память — собирает улики от червей и формирует ответ
pub struct WorkingMemory {
    pub evidence: Vec<Evidence>,
    pub dim: usize,
    pub context_tokens: Vec<u16>,  // контекст для которого формируем ответ
    verification_done: bool,
}

/// Результат работы рабочей памяти
pub struct WorkingMemoryResult {
    pub predictions: Vec<(u16, f64)>,
    pub total_evidence: usize,
    pub verified_evidence: usize,
    pub sources_used: Vec<&'static str>,
    pub confidence: f64,
}

impl WorkingMemory {
    pub fn new(dim: usize, context: Vec<u16>) -> Self {
        WorkingMemory {
            evidence: Vec::with_capacity(256),
            dim,
            context_tokens: context,
            verification_done: false,
        }
    }

    // ============================================================
    // Phase 1: Collector — сбор улик из всех уровней памяти
    // ============================================================

    /// Collector: ищем кандидатов в facts по триграмме.
    /// raw_score = sim_weight * (cnt / total) — чистая вероятность без base_weight.
    pub fn collect_from_trigram(&mut self, memory: &HierarchicalMemory, t0: u16, t1: u16, t2: u16) {
        let trigram = memory.make_trigram_query(t0, t1, t2);
        let top_facts = memory.facts.find_top_k(&trigram, 20);

        for (idx, sim) in &top_facts {
            if *sim <= 0 { continue; }
            let entry = &memory.facts.entries[*idx];
            let sim_weight = *sim as f64 / self.dim as f64;
            for (&tok, &cnt) in &entry.successors {
                let raw = sim_weight * (cnt as f64) / entry.total_count.max(1) as f64;
                self.add_evidence(tok, raw, EvidenceSource::Trigram, sim_weight);
            }
        }
    }

    /// Collector: ищем кандидатов по биграмме (fallback).
    /// raw_score = cnt / total (без 0.5 множителя — base_weight применится в finalize).
    pub fn collect_from_bigram(&mut self, memory: &HierarchicalMemory, t1: u16, t2: u16) {
        let bigram_indices = memory.facts.find_by_bigram(t1, t2, 30);
        for idx in &bigram_indices {
            let entry = &memory.facts.entries[*idx];
            // Пропускаем слабые факты (total_count=1 — один раз видели, шум)
            if entry.total_count < 2 { continue; }
            for (&tok, &cnt) in &entry.successors {
                let raw = (cnt as f64) / entry.total_count.max(1) as f64;
                self.add_evidence(tok, raw, EvidenceSource::Bigram, 0.6);
            }
        }
    }

    /// Collector: bigram rules от Explorer.
    pub fn collect_from_bigram_rules(&mut self, memory: &HierarchicalMemory, t1: u16, t2: u16) {
        if memory.rules.entries.is_empty() { return; }
        let bigram_query = memory.make_bigram_query(t1, t2);
        let top_rules = memory.rules.find_top_k(&bigram_query, 10);
        for (idx, sim) in &top_rules {
            if *sim <= 0 { continue; }
            let entry = &memory.rules.entries[*idx];
            let sim_weight = *sim as f64 / self.dim as f64;
            for (&tok, &cnt) in &entry.successors {
                let raw = sim_weight * (cnt as f64) / entry.total_count.max(1) as f64;
                self.add_evidence(tok, raw, EvidenceSource::BigramRule, sim_weight);
            }
        }
    }

    /// Collector: unigram fallback (только если trigram мало нашёл).
    pub fn collect_from_unigram(&mut self, memory: &HierarchicalMemory, t2: u16) {
        let trigram_count = self.evidence.iter()
            .filter(|e| e.source == EvidenceSource::Trigram)
            .count();
        if trigram_count > 5 { return; }

        let unigram_indices = memory.facts.find_by_token(t2, 20);
        for idx in &unigram_indices {
            let entry = &memory.facts.entries[*idx];
            if entry.total_count < 2 { continue; }
            for (&tok, &cnt) in &entry.successors {
                let raw = (cnt as f64) / entry.total_count.max(1) as f64;
                self.add_evidence(tok, raw, EvidenceSource::Unigram, 0.3);
            }
        }
    }

    /// Collector: medium context rules (10 tok window).
    pub fn collect_from_rules(&mut self, memory: &HierarchicalMemory, tokens: &[u16], pos: usize) {
        if pos < 10 || memory.rules.entries.is_empty() { return; }
        let med_ctx = memory.make_context_bundle(tokens, pos, 10);
        let top_med = memory.rules.find_top_k(&med_ctx, 10);
        for (idx, sim) in &top_med {
            if *sim <= 0 { continue; }
            let entry = &memory.rules.entries[*idx];
            let sim_weight = *sim as f64 / self.dim as f64;
            for (&tok, &cnt) in &entry.successors {
                let raw = sim_weight * (cnt as f64) / entry.total_count.max(1) as f64;
                self.add_evidence(tok, raw, EvidenceSource::Rule, sim_weight);
            }
        }
    }

    /// Collector: abstractions — trigram search + wide context (50 tok) search.
    pub fn collect_from_abstractions(&mut self, memory: &HierarchicalMemory, tokens: &[u16], pos: usize, t0: u16, t1: u16, t2: u16) {
        if memory.abstractions.entries.is_empty() { return; }

        // Search abstractions by trigram
        let trigram = memory.make_trigram_query(t0, t1, t2);
        let top_abs = memory.abstractions.find_top_k(&trigram, 5);
        for (idx, sim) in &top_abs {
            if *sim <= 0 { continue; }
            let entry = &memory.abstractions.entries[*idx];
            let sim_weight = *sim as f64 / self.dim as f64;
            for (&tok, &cnt) in &entry.successors {
                let raw = sim_weight * (cnt as f64) / entry.total_count.max(1) as f64;
                self.add_evidence(tok, raw, EvidenceSource::Abstraction, sim_weight);
            }
        }

        // Wide context search (50 tok) — ранее был потерян!
        if pos >= 50 {
            let wide_ctx = memory.make_context_bundle(tokens, pos, 50);
            let top_wide = memory.abstractions.find_top_k(&wide_ctx, 5);
            for (idx, sim) in &top_wide {
                if *sim <= 0 { continue; }
                let entry = &memory.abstractions.entries[*idx];
                let sim_weight = *sim as f64 / self.dim as f64;
                for (&tok, &cnt) in &entry.successors {
                    let raw = sim_weight * 0.5 * (cnt as f64) / entry.total_count.max(1) as f64;
                    self.add_evidence(tok, raw, EvidenceSource::Abstraction, sim_weight * 0.5);
                }
            }
        }
    }

    /// Collector: семантический fallback — ищем по контексту в facts когда мало точных совпадений.
    pub fn collect_semantic(&mut self, memory: &HierarchicalMemory, tokens: &[u16], pos: usize) {
        // Только если не хватает evidence из точных источников
        let strong_evidence = self.evidence.iter()
            .filter(|e| matches!(e.source, EvidenceSource::Trigram | EvidenceSource::Bigram))
            .count();
        if strong_evidence > 10 { return; }
        if pos < 5 { return; }

        // Собираем семантический вектор контекста из последних 5 токенов
        let ctx = memory.make_context_bundle(tokens, pos, 5);
        let top_semantic = memory.facts.find_top_k(&ctx, 15);

        for (idx, sim) in &top_semantic {
            if *sim <= 0 { continue; }
            let entry = &memory.facts.entries[*idx];
            let sim_weight = *sim as f64 / self.dim as f64;
            for (&tok, &cnt) in &entry.successors {
                let raw = sim_weight * (cnt as f64) / entry.total_count.max(1) as f64;
                self.add_evidence(tok, raw, EvidenceSource::Semantic, sim_weight);
            }
        }
    }

    // ============================================================
    // Phase 2: Analyst — attention context re-ranking (оптимизированный)
    // ============================================================

    /// Analyst: контекстное переранжирование через attention.
    /// Оптимизация: использует POPCNT через XOR вместо bit-level loop.
    pub fn analyze_with_context(&mut self, memory: &HierarchicalMemory, tokens: &[u16], pos: usize, vocab_size: usize) {
        let context_window = 32.min(pos);
        if context_window < 3 || self.evidence.is_empty() { return; }

        // Строим контекстный вектор через BundleAccumulator (уже оптимизирован)
        let mut acc = BundleAccumulator::new(self.dim);
        let start = pos - context_window;
        for j in start..=pos {
            let tok = tokens[j] as usize;
            if tok >= vocab_size { continue; }
            let recency = if pos - j <= 3 { 3 } else if pos - j <= 10 { 2 } else { 1 };
            let word = &memory.words[tok];
            for _ in 0..recency {
                acc.add(word);
            }
        }
        let context_vec = acc.to_binary();

        // Собираем уникальные токены-кандидаты
        let candidates: Vec<u16> = {
            let mut seen = std::collections::HashSet::new();
            self.evidence.iter()
                .filter(|e| (e.token as usize) < vocab_size)
                .filter(|e| seen.insert(e.token))
                .map(|e| e.token)
                .collect()
        };

        // Оценка через similarity() — использует POPCNT, O(dim/64) вместо O(dim)
        let max_sim = self.dim as f64;
        for tok in candidates {
            let word = &memory.words[tok as usize];
            let sim = context_vec.similarity(word) as f64;
            // Нормализуем: sim ∈ [-dim, dim] → [-1, 1]
            let normalized = sim / max_sim;
            // Добавляем как evidence (может быть отрицательным — штраф за анти-корреляцию)
            self.add_evidence(tok, normalized, EvidenceSource::Context, normalized.abs().min(1.0));
        }
    }

    // ============================================================
    // Phase 3: Skeptic — самопроверка (verification)
    // ============================================================

    /// Skeptic: самопроверка top кандидатов.
    /// - Multi-source confirmation: 3+ источника → бонус
    /// - Cross-check: кандидат как successor в фактах с context_token = last
    pub fn verify(&mut self, memory: &HierarchicalMemory) {
        if self.verification_done { return; }
        self.verification_done = true;

        // Агрегируем по токенам
        let mut token_scores: HashMap<u16, f64> = HashMap::new();
        let mut token_sources: HashMap<u16, std::collections::HashSet<EvidenceSource>> = HashMap::new();

        for ev in &self.evidence {
            *token_scores.entry(ev.token).or_insert(0.0) += ev.raw_score * ev.source.base_weight();
            token_sources.entry(ev.token).or_default().insert(ev.source.clone());
        }

        // Top-10 кандидатов для проверки
        let mut ranked: Vec<(u16, f64)> = token_scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut verification_evidence: Vec<Evidence> = Vec::new();

        for (tok, _score) in ranked.iter().take(10) {
            let sources = token_sources.get(tok).unwrap();
            let source_count = sources.len();

            // Multi-source confirmation
            if source_count >= 3 {
                verification_evidence.push(Evidence {
                    token: *tok,
                    raw_score: 0.3 * source_count as f64 / 5.0,
                    source: EvidenceSource::Verification,
                    verified: true,
                    confidence: 0.9,
                });
            } else if source_count >= 2 {
                verification_evidence.push(Evidence {
                    token: *tok,
                    raw_score: 0.15,
                    source: EvidenceSource::Verification,
                    verified: true,
                    confidence: 0.7,
                });
            }

            // Cross-check: ищем факты где LAST контекстный токен → candidate как successor
            // Это правильная проверка: "после X часто идёт Y?"
            if !self.context_tokens.is_empty() {
                let last = *self.context_tokens.last().unwrap();
                let facts_with_last = memory.facts.find_by_token(last, 50);
                let mut confirm_count = 0u32;
                for &idx in &facts_with_last {
                    if let Some(&cnt) = memory.facts.entries[idx].successors.get(tok) {
                        confirm_count += cnt;
                    }
                }
                if confirm_count >= 3 {
                    verification_evidence.push(Evidence {
                        token: *tok,
                        raw_score: (confirm_count as f64 / 20.0).min(0.5),
                        source: EvidenceSource::Verification,
                        verified: true,
                        confidence: 0.8,
                    });
                }
            }
        }

        for ev in self.evidence.iter_mut() {
            ev.verified = true;
        }

        self.evidence.extend(verification_evidence);
    }

    // ============================================================
    // Finalize — формируем итоговый ответ
    // ============================================================

    /// Финализация: base_weight применяется ОДИН РАЗ здесь.
    /// final_score = Σ (raw_score × base_weight × confidence)
    pub fn finalize(&self) -> WorkingMemoryResult {
        let mut scores: HashMap<u16, f64> = HashMap::new();
        let mut sources_seen: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
        let mut verified_count = 0usize;

        for ev in &self.evidence {
            // base_weight применяется ТОЛЬКО ЗДЕСЬ — не в collect_*
            let final_score = ev.raw_score * ev.source.base_weight() * ev.confidence;
            *scores.entry(ev.token).or_insert(0.0) += final_score;
            sources_seen.insert(ev.source.name());
            if ev.verified { verified_count += 1; }
        }

        let mut predictions: Vec<(u16, f64)> = scores.into_iter().collect();
        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        predictions.truncate(10);

        // Confidence = separation between top-1 and top-2
        let confidence = if predictions.len() >= 2 {
            let top = predictions[0].1;
            let second = predictions[1].1;
            if top > 0.0 { (top - second) / top } else { 0.0 }
        } else if predictions.len() == 1 {
            1.0
        } else {
            0.0
        };

        WorkingMemoryResult {
            predictions,
            total_evidence: self.evidence.len(),
            verified_evidence: verified_count,
            sources_used: sources_seen.into_iter().collect(),
            confidence,
        }
    }

    // ============================================================
    // Internal
    // ============================================================

    fn add_evidence(&mut self, token: u16, raw_score: f64, source: EvidenceSource, confidence: f64) {
        self.evidence.push(Evidence {
            token,
            raw_score,
            source,
            verified: false,
            confidence: confidence.clamp(0.0, 1.0),
        });
    }

    /// Полный pipeline: collect → analyze → verify → finalize
    pub fn full_pipeline(
        memory: &HierarchicalMemory,
        tokens: &[u16],
        pos: usize,
        vocab_size: usize,
        dim: usize,
    ) -> WorkingMemoryResult {
        if pos < 2 || pos >= tokens.len() {
            return WorkingMemoryResult {
                predictions: vec![],
                total_evidence: 0,
                verified_evidence: 0,
                sources_used: vec![],
                confidence: 0.0,
            };
        }

        let t0 = tokens[pos - 2];
        let t1 = tokens[pos - 1];
        let t2 = tokens[pos];

        let context = if pos >= 5 {
            tokens[pos - 5..=pos].to_vec()
        } else {
            tokens[..=pos].to_vec()
        };

        let mut wm = WorkingMemory::new(dim, context);

        // Phase 1: Collector — сбор улик из всех уровней
        wm.collect_from_trigram(memory, t0, t1, t2);
        wm.collect_from_bigram(memory, t1, t2);
        wm.collect_from_bigram_rules(memory, t1, t2);
        wm.collect_from_unigram(memory, t2);
        wm.collect_from_rules(memory, tokens, pos);
        wm.collect_from_abstractions(memory, tokens, pos, t0, t1, t2);
        wm.collect_semantic(memory, tokens, pos);

        // Phase 2: Analyst — attention context re-ranking
        wm.analyze_with_context(memory, tokens, pos, vocab_size);

        // Phase 3: Skeptic — самопроверка
        wm.verify(memory);

        // Phase 4: Finalize
        wm.finalize()
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_working_memory_basic() {
        let mut wm = WorkingMemory::new(256, vec![1, 2, 3]);
        wm.add_evidence(42, 0.8, EvidenceSource::Trigram, 0.9);
        wm.add_evidence(42, 0.3, EvidenceSource::Bigram, 0.5);
        wm.add_evidence(7, 0.2, EvidenceSource::Unigram, 0.3);

        let result = wm.finalize();
        assert!(!result.predictions.is_empty());
        assert_eq!(result.predictions[0].0, 42); // token 42 should be top
        assert_eq!(result.total_evidence, 3);
    }

    #[test]
    fn test_no_double_weighting() {
        // Verify base_weight applied only once
        let mut wm = WorkingMemory::new(256, vec![1, 2, 3]);
        // raw_score=1.0 for trigram, base_weight=1.0, confidence=1.0
        wm.add_evidence(42, 1.0, EvidenceSource::Trigram, 1.0);

        let result = wm.finalize();
        // final = 1.0 * 1.0 * 1.0 = 1.0 (not 1.0 * 1.0 * 1.0 * 1.0)
        assert!((result.predictions[0].1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_working_memory_verification() {
        let memory = HierarchicalMemory::new(256, 100);
        let mut wm = WorkingMemory::new(256, vec![1, 2, 3]);

        wm.add_evidence(42, 0.8, EvidenceSource::Trigram, 0.9);
        wm.add_evidence(42, 0.5, EvidenceSource::Bigram, 0.7);
        wm.add_evidence(42, 0.3, EvidenceSource::Rule, 0.5);
        wm.add_evidence(7, 0.9, EvidenceSource::Trigram, 0.9);

        wm.verify(&memory);

        let result = wm.finalize();
        assert!(result.verified_evidence > 0);
        assert_eq!(result.predictions[0].0, 42);
    }

    #[test]
    fn test_full_pipeline() {
        let memory = HierarchicalMemory::new(256, 100);
        let tokens: Vec<u16> = (0..200).map(|i| (i % 100) as u16).collect();

        let result = WorkingMemory::full_pipeline(&memory, &tokens, 50, 100, 256);
        assert_eq!(result.total_evidence, 0);
    }

    #[test]
    fn test_pipeline_with_data() {
        use crate::language::LogicLanguage;
        use crate::worms::*;

        let mut memory = HierarchicalMemory::new(256, 100);
        let lang = LogicLanguage::new(256);
        let tokens: Vec<u16> = (0..500).map(|i| (i % 100) as u16).collect();

        CollectorWorm::think(&mut memory, &lang, &[], &tokens, 1);

        let result = WorkingMemory::full_pipeline(&memory, &tokens, 50, 100, 256);
        assert!(result.total_evidence > 0);
        assert!(!result.predictions.is_empty());
    }
}
