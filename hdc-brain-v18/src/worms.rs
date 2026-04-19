//! Worm Engine — черви КАК алгоритм обучения
//!
//! П.1: Analyst ищет соседей по общим токенам (не LSH мусор)
//! П.3: Explorer создаёт bigram rules из паттернов
//! П.4: Curriculum — Collector фокусируется на MISSING областях
//!
//! Черви думают на языке, обмениваются мыслями, действуют инструментами.

use crate::binary::*;
use crate::memory::*;
use crate::language::*;
use crate::logger::*;
use std::collections::HashMap;
use std::io::Write;

fn flush() { std::io::stdout().flush().ok(); }

// ============================================================
// WormOutput
// ============================================================

pub struct WormOutput {
    #[allow(dead_code)]
    pub name: &'static str,
    pub thoughts: Vec<Thought>,
    pub trace: Vec<String>,
    pub stored: usize,
    pub strengthened: usize,
    pub weakened: usize,
    pub asked: usize,
    pub scanned: usize,
}

impl WormOutput {
    fn new(name: &'static str) -> Self {
        WormOutput {
            name, thoughts: Vec::new(), trace: Vec::new(),
            stored: 0, strengthened: 0, weakened: 0, asked: 0, scanned: 0,
        }
    }
    fn trace(&mut self, s: &str) { self.trace.push(s.to_string()); }
    fn think(&mut self, t: Thought) { self.thoughts.push(t); }
}

// ============================================================
// CollectorWorm — П.4: curriculum learning
// ============================================================

pub struct CollectorWorm;

impl CollectorWorm {
    pub fn think(
        memory: &mut HierarchicalMemory,
        lang: &LogicLanguage,
        inbox: &[Thought],
        data: &[u16],
        sample_rate: usize,
    ) -> WormOutput {
        let mut out = WormOutput::new("Collector");
        let vs = memory.vocab_size;
        let med_rate = sample_rate.max(1) * 5;
        let wide_rate = sample_rate.max(1) * 20;

        // П.4 CURRICULUM: собрать MISSING токены от Explorer
        let mut missing_tokens: std::collections::HashSet<u16> = std::collections::HashSet::new();
        for t in inbox.iter().filter(|t| t.kind == Opcode::Missing) {
            // Извлечь ближайший токен из MISSING вектора
            let (nearest_tok, _) = memory.nearest_word(&lang.extract_ask_content(&t.vec));
            missing_tokens.insert(nearest_tok);
        }
        let has_curriculum = !missing_tokens.is_empty();

        for i in 2..data.len().saturating_sub(1) {
            let (t0, t1, t2, succ) = (data[i - 2], data[i - 1], data[i], data[i + 1]);
            if t0 as usize >= vs || t1 as usize >= vs
               || t2 as usize >= vs || succ as usize >= vs { continue; }

            // П.4: Если есть curriculum — sample_rate=1 для MISSING, обычный для остальных
            if has_curriculum {
                let is_missing_area = missing_tokens.contains(&t0)
                    || missing_tokens.contains(&t1)
                    || missing_tokens.contains(&t2);
                if !is_missing_area && sample_rate > 1 && i % sample_rate != 0 {
                    continue;
                }
            } else if sample_rate > 1 && i % sample_rate != 0 {
                continue;
            }

            out.scanned += 1;

            let query = memory.make_trigram_query(t0, t1, t2);

            match memory.facts.find_matching(&query) {
                Some(idx) => {
                    // STRONG мысли только для каждого 50-го — иначе миллионы бесполезных мыслей
                    if out.strengthened % 50 == 0 {
                        out.think(Thought::new(
                            lang.strong(&query), Opcode::Strong, 1, Opcode::SourceData,
                        ));
                    }
                    memory.facts.entries[idx].add_successor(succ);
                    out.strengthened += 1;
                }
                None => {
                    out.think(Thought::new(
                        lang.novel(&query), Opcode::Novel, 2, Opcode::SourceData,
                    ));
                    let mut entry = MemoryEntry::with_context(query, 0, vec![t0, t1, t2]);
                    entry.add_successor(succ);
                    memory.facts.store(entry);
                    out.stored += 1;
                }
            }

            // Medium context — сохраняем последние 3 токена как context_tokens
            // чтобы rules были доступны через token_index и bigram_index
            if i >= 10 && i % med_rate == 0 {
                let ctx = memory.make_context_bundle(data, i, 10);
                match memory.rules.find_matching(&ctx) {
                    Some(idx) => {
                        memory.rules.entries[idx].add_successor(succ);
                        out.strengthened += 1;
                    }
                    None => {
                        // Ключевые токены из окна: последние 3 + succ для индексации
                        let rule_ctx = vec![t0, t1, t2];
                        let mut entry = MemoryEntry::with_context(ctx, 0, rule_ctx);
                        entry.add_successor(succ);
                        memory.rules.store(entry);
                        out.stored += 1;
                    }
                }
            }

            // Wide context — сохраняем последние 5 токенов как context_tokens
            if i >= 50 && i % wide_rate == 0 {
                let ctx = memory.make_context_bundle(data, i, 50);
                match memory.abstractions.find_matching(&ctx) {
                    Some(idx) => {
                        memory.abstractions.entries[idx].add_successor(succ);
                        out.strengthened += 1;
                    }
                    None => {
                        // Берём 5 ключевых токенов из окна для индексации
                        let abs_start = if i >= 5 { i - 5 } else { 0 };
                        let abs_ctx: Vec<u16> = data[abs_start..=i].iter()
                            .filter(|&&tok| (tok as usize) < vs)
                            .cloned()
                            .collect();
                        let mut entry = MemoryEntry::with_context(ctx, 0, abs_ctx);
                        entry.add_successor(succ);
                        memory.abstractions.store(entry);
                        out.stored += 1;
                    }
                }
            }
        }

        out.trace(&format!(
            "DATA: {} scanned → {} novel, {} strengthened{}",
            out.scanned, out.stored, out.strengthened,
            if has_curriculum { format!(" (curriculum: {} MISSING tokens)", missing_tokens.len()) } else { String::new() },
        ));

        out
    }
}

// ============================================================
// AnalystWorm — П.1: поиск по общим токенам, не LSH
// ============================================================

pub struct AnalystWorm;

impl AnalystWorm {
    pub fn think(
        memory: &mut HierarchicalMemory,
        lang: &LogicLanguage,
        inbox: &[Thought],
        max_checks: usize,
        _vocab: Option<&[String]>,
    ) -> WormOutput {
        let mut out = WormOutput::new("Analyst");
        let n = memory.facts.len();
        if n < 10 { return out; }

        let novel_count = inbox.iter().filter(|t| t.kind == Opcode::Novel).count();
        let effective_checks = if novel_count > 100 {
            (max_checks * 2).min(n)
        } else {
            max_checks.min(n)
        };

        let step = (n / effective_checks).max(1);
        let mut correct = 0u32;
        let mut conflicts = 0u32;

        for i in (0..n).step_by(step) {
            if out.scanned >= effective_checks { break; }
            out.scanned += 1;

            let entry_vec = memory.facts.entries[i].vec.clone();
            let my_top = match memory.facts.entries[i].top1() {
                Some((tok, _)) => tok,
                None => continue,
            };
            let ctx_tokens = memory.facts.entries[i].context_tokens.clone();

            // П.1: FIND соседей по ОБЩИМ ТОКЕНАМ (не LSH!)
            // Приоритет: bigram совпадение > shared tokens > LSH fallback
            let neighbors = if ctx_tokens.len() >= 2 {
                // Сначала ищем по биграмме (точнее: последние 2 токена контекста)
                let last2 = (ctx_tokens[ctx_tokens.len() - 2], ctx_tokens[ctx_tokens.len() - 1]);
                let bigram_n = memory.facts.find_by_bigram(last2.0, last2.1, 5);
                let mut combined: Vec<(usize, usize)> = bigram_n.iter()
                    .filter(|&&idx| idx != i)
                    .map(|&idx| (idx, 3)) // bigram match = 3 "shared tokens" weight
                    .collect();

                // Дополняем shared tokens
                let shared_n = memory.facts.find_by_shared_tokens(&ctx_tokens, i, 10);
                for (idx, shared) in shared_n {
                    if !combined.iter().any(|(j, _)| *j == idx) {
                        combined.push((idx, shared));
                    }
                }
                combined
            } else {
                memory.facts.find_top_k(&entry_vec, 5)
                    .into_iter().map(|(idx, sim)| (idx, sim.max(0) as usize)).collect()
            };

            // COMPARE: консенсус соседей
            let mut votes: HashMap<u16, u32> = HashMap::new();
            for &(j, shared) in &neighbors {
                if shared < 1 { continue; }
                if let Some((tok, cnt)) = memory.facts.entries[j].top1() {
                    let weight = (shared as u32) * cnt.min(10);
                    *votes.entry(tok).or_insert(0) += weight;
                }
            }

            let consensus = votes.iter()
                .max_by_key(|(_, &v)| v)
                .map(|(&t, &c)| (t, c));

            match consensus {
                Some((con_tok, _)) if con_tok == my_top => {
                    correct += 1;
                    out.think(Thought::new(
                        lang.strong(&entry_vec), Opcode::Strong, 1, Opcode::SourceAnalyst,
                    ));
                    memory.facts.entries[i].confirm_successor(my_top, 1);
                    out.strengthened += 1;
                }
                Some((con_tok, con_weight)) if con_weight >= 5 => {
                    conflicts += 1;
                    let my_vec = memory.word_vec(my_top).clone();
                    let con_vec = memory.word_vec(con_tok).clone();
                    out.think(Thought::new(
                        lang.conflict(&my_vec, &con_vec),
                        Opcode::Conflict, 4, Opcode::SourceAnalyst,
                    ));
                    memory.facts.entries[i].confirm_successor(con_tok, 2);
                    out.strengthened += 1;
                }
                Some((_, _)) => {
                    conflicts += 1;
                    out.think(Thought::new(
                        lang.ask(&entry_vec), Opcode::Ask, 5, Opcode::SourceAnalyst,
                    ));
                    out.asked += 1;
                }
                None => {
                    out.think(Thought::new(
                        lang.weak(&entry_vec), Opcode::Weak, 3, Opcode::SourceAnalyst,
                    ));
                }
            }
        }

        let total = correct + conflicts;
        let accuracy = if total > 0 { correct as f64 / total as f64 * 100.0 } else { 0.0 };
        out.trace(&format!(
            "CHECK: {:.1}% agree ({}/{}), {} conflicts, {} asked, {} neighbors_by_tokens",
            accuracy, correct, total, conflicts, out.asked,
            if !memory.facts.token_index.is_empty() { "YES" } else { "NO" },
        ));

        out
    }
}

// ============================================================
// SkepticWorm
// ============================================================

pub struct SkepticWorm;

impl SkepticWorm {
    pub fn think(
        memory: &mut HierarchicalMemory,
        lang: &LogicLanguage,
        inbox: &[Thought],
        max_checks: usize,
        _vocab: Option<&[String]>,
    ) -> WormOutput {
        let mut out = WormOutput::new("Skeptic");

        let conflicts: Vec<&Thought> = inbox.iter()
            .filter(|t| t.kind == Opcode::Conflict)
            .collect();

        out.trace(&format!(
            "Inbox: {} CONFLICT, {} ASK from others",
            conflicts.len(),
            inbox.iter().filter(|t| t.kind == Opcode::Ask).count(),
        ));

        for thought in conflicts.iter().take(max_checks) {
            out.scanned += 1;

            let conflict_content = lang.extract_ask_content(&thought.vec);

            // Попробовать rules
            if let Some((rule_idx, sim)) = memory.rules.find_nearest(&conflict_content) {
                if sim > (memory.dim as i32 / 4) {
                    if let Some((rule_tok, _)) = memory.rules.entries[rule_idx].top1() {
                        out.think(Thought::new(
                            lang.why(&conflict_content),
                            Opcode::Why, 3, Opcode::SourceSkeptic,
                        ));
                        if let Some((fact_idx, _)) = memory.facts.find_nearest(&conflict_content) {
                            memory.facts.entries[fact_idx].confirm_successor(rule_tok, 2);
                            out.strengthened += 1;
                        }
                        continue;
                    }
                }
            }

            // Глубокий поиск
            let neighbors = memory.facts.find_top_k(&conflict_content, 10);
            let mut deep_votes: HashMap<u16, u32> = HashMap::new();
            for &(idx, sim) in &neighbors {
                if sim <= 0 { continue; }
                if let Some((tok, cnt)) = memory.facts.entries[idx].top1() {
                    *deep_votes.entry(tok).or_insert(0) += cnt.min(5);
                }
            }

            if let Some((&best_tok, &best_cnt)) = deep_votes.iter().max_by_key(|(_, &v)| v) {
                if best_cnt >= 5 {
                    out.think(Thought::new(
                        lang.strong(&memory.word_vec(best_tok).clone()),
                        Opcode::Strong, 3, Opcode::SourceSkeptic,
                    ));
                    if let Some((fact_idx, _)) = memory.facts.find_nearest(&conflict_content) {
                        memory.facts.entries[fact_idx].confirm_successor(best_tok, 3);
                        out.strengthened += 1;
                    }
                } else {
                    out.think(Thought::new(
                        lang.ask(&conflict_content),
                        Opcode::Ask, 7, Opcode::SourceSkeptic,
                    ));
                    out.asked += 1;
                }
            }
        }

        // Self-scan: ambiguous facts
        let n = memory.facts.len();
        if n >= 10 {
            let step = (n / max_checks.max(1)).max(1);
            for i in (0..n).step_by(step) {
                if out.scanned >= max_checks * 2 { break; }
                let total = memory.facts.entries[i].total_count;
                if total < 3 { continue; }
                let top = memory.facts.entries[i].top_successors(2);
                if top.len() < 2 { continue; }
                let dominance = top[0].1 as f64 / total as f64;
                if dominance >= 0.5 { continue; }
                out.scanned += 1;
                let entry_vec = memory.facts.entries[i].vec.clone();
                out.think(Thought::new(
                    lang.weak(&entry_vec), Opcode::Weak, 3, Opcode::SourceSkeptic,
                ));
                out.weakened += 1;
            }
        }

        out.trace(&format!(
            "RESOLVE: {} strengthened, {} asked, {} weak found",
            out.strengthened, out.asked, out.weakened,
        ));

        out
    }
}

// ============================================================
// ExplorerWorm — П.3: bigram rules + П.4: MISSING for curriculum
// ============================================================

pub struct ExplorerWorm;

impl ExplorerWorm {
    pub fn think(
        memory: &mut HierarchicalMemory,
        lang: &LogicLanguage,
        inbox: &[Thought],
        max_ops: usize,
        _vocab: Option<&[String]>,
    ) -> WormOutput {
        let mut out = WormOutput::new("Explorer");
        let n = memory.facts.len();
        if n < 10 { return out; }

        let strong_count = inbox.iter()
            .filter(|t| t.kind == Opcode::Strong
                && (t.source == Opcode::SourceAnalyst || t.source == Opcode::SourceSkeptic))
            .count();
        out.trace(&format!("Inbox: {} confirmed STRONG", strong_count));

        // === 1. Карта покрытия ===
        let step = (n / max_ops).max(1);
        let mut token_coverage: HashMap<u16, (u32, u32)> = HashMap::new();

        for i in (0..n).step_by(step) {
            if out.scanned >= max_ops { break; }
            out.scanned += 1;
            for &tok in &memory.facts.entries[i].context_tokens {
                let stat = token_coverage.entry(tok).or_insert((0, 0));
                stat.0 += 1;
                stat.1 += memory.facts.entries[i].total_count;
            }
        }

        let mut coverage: Vec<(u16, u32, u32)> = token_coverage.iter()
            .map(|(&t, &(c, conf))| (t, c, conf))
            .collect();
        coverage.sort_by(|a, b| b.2.cmp(&a.2));

        let strong_areas = coverage.iter().filter(|(_, _, c)| *c > 10).count();
        let weak_areas = coverage.iter().filter(|(_, c, _)| *c <= 2).count();

        // === 2. П.4: MISSING → мысли для curriculum ===
        let weak_tokens: Vec<u16> = coverage.iter()
            .filter(|(_, c, _)| *c <= 2)
            .map(|(tok, _, _)| *tok)
            .take(50) // До 50 MISSING областей
            .collect();

        for &tok in weak_tokens.iter().take(20) {
            let word_vec = memory.word_vec(tok).clone();
            out.think(Thought::new(
                lang.missing(&word_vec), Opcode::Missing, 3, Opcode::SourceExplorer,
            ));
            out.think(Thought::new(
                lang.ask(&word_vec), Opcode::Ask, 2, Opcode::SourceExplorer,
            ));
            out.asked += 1;
        }

        // === 3. П.3: Bigram rules — группировка фактов по биграмме ===
        let mut bigram_successors: HashMap<(u16, u16), HashMap<u16, u32>> = HashMap::new();

        for i in (0..n).step_by(step.max(1) * 3) {
            let ctx = &memory.facts.entries[i].context_tokens;
            if ctx.len() < 3 { continue; }
            // Последние 2 токена контекста = биграмма
            let bigram = (ctx[ctx.len() - 2], ctx[ctx.len() - 1]);
            if let Some((succ_tok, cnt)) = memory.facts.entries[i].top1() {
                let entry = bigram_successors.entry(bigram).or_default();
                *entry.entry(succ_tok).or_insert(0) += cnt;
            }
        }

        // Создать bigram rules для частых биграмм
        // Сортируем по total для детерминированности
        let mut bigram_sorted: Vec<_> = bigram_successors.iter().collect();
        bigram_sorted.sort_by(|a, b| {
            let ta: u32 = a.1.values().sum();
            let tb: u32 = b.1.values().sum();
            tb.cmp(&ta)
        });

        let mut bigram_rules_created = 0;
        for ((t0, t1), successors) in bigram_sorted {
            let total: u32 = successors.values().sum();
            // FIX: принимаем и 1-successor правила (они самые сильные!)
            if total < 3 { continue; }
            if bigram_rules_created >= 200 { break; }

            // Создать rule вектор из биграммы
            let rule_vec = memory.make_bigram_query(*t0, *t1);

            // Проверить дубликат
            if memory.rules.find_matching(&rule_vec).is_some() { continue; }

            let mut entry = MemoryEntry::with_context(rule_vec.clone(), 3, vec![*t0, *t1]);
            for (&tok, &cnt) in successors {
                entry.confirm_successor(tok, cnt);
            }
            memory.rules.store(entry);
            bigram_rules_created += 1;
            out.stored += 1;

            out.think(Thought::new(
                lang.store(&rule_vec, Opcode::LevelRules),
                Opcode::Store, 2, Opcode::SourceExplorer,
            ));
        }

        // === 4. Обобщения из групп фактов ===
        let mut successor_groups: HashMap<u16, Vec<usize>> = HashMap::new();
        for i in (0..n).step_by(step.max(1) * 5) {
            if let Some((top_tok, _)) = memory.facts.entries[i].top1() {
                successor_groups.entry(top_tok).or_default().push(i);
            }
        }

        // Сортируем по размеру группы (детерминированный порядок)
        let mut sorted_groups: Vec<_> = successor_groups.into_iter().collect();
        sorted_groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        let mut abstractions_created = 0;
        for (succ_tok, indices) in &sorted_groups {
            if indices.len() < 5 || abstractions_created >= 50 { break; }
            let mut acc = BundleAccumulator::new(memory.dim);

            // FIX: собираем context_tokens из составляющих фактов
            let mut ctx_freq: HashMap<u16, u32> = HashMap::new();
            for &idx in indices.iter().take(20) {
                acc.add(&memory.facts.entries[idx].vec);
                for &tok in &memory.facts.entries[idx].context_tokens {
                    *ctx_freq.entry(tok).or_insert(0) += 1;
                }
            }
            let abstract_vec = acc.to_binary();
            if memory.abstractions.find_matching(&abstract_vec).is_some() { continue; }

            // Берём top-5 самых частых context_tokens для абстракции
            let mut ctx_sorted: Vec<_> = ctx_freq.into_iter().collect();
            ctx_sorted.sort_by(|a, b| b.1.cmp(&a.1));
            let abstract_ctx: Vec<u16> = ctx_sorted.iter().take(5).map(|(t, _)| *t).collect();

            let mut entry = MemoryEntry::with_context(abstract_vec, 3, abstract_ctx);
            entry.confirm_successor(*succ_tok, indices.len() as u32);
            memory.abstractions.store(entry);
            abstractions_created += 1;
            out.stored += 1;
        }

        // === 5. Meta (ограничено до 100 записей) ===
        if coverage.len() >= 3 && memory.meta.len() < 100 {
            let mut meta_acc = BundleAccumulator::new(memory.dim);
            for &(tok, _, _) in coverage.iter().take(10) {
                meta_acc.add(memory.word_vec(tok));
            }
            let profile = meta_acc.to_binary();
            // Проверяем дубликат
            if memory.meta.find_matching(&profile).is_none() {
                let mut meta_entry = MemoryEntry::with_context(profile, 3, vec![]);
                for &(tok, _, conf) in coverage.iter().take(10) {
                    meta_entry.confirm_successor(tok, conf.min(100));
                }
                memory.meta.store(meta_entry);
                out.stored += 1;
            }
        }

        out.trace(&format!(
            "MAP: {} tokens ({} strong, {} weak), {} bigram_rules, {} abstractions, {} MISSING sent",
            coverage.len(), strong_areas, weak_areas,
            bigram_rules_created, abstractions_created, weak_tokens.len().min(20),
        ));

        out
    }
}

// ============================================================
// WormOrchestrator
// ============================================================

pub struct WormOrchestrator {
    pub chunk_size: usize,
    pub sample_rate: usize,
    pub max_checks: usize,
}

impl WormOrchestrator {
    pub fn new(chunk_size: usize, sample_rate: usize, max_checks: usize) -> Self {
        WormOrchestrator { chunk_size, sample_rate, max_checks }
    }

    pub fn run_pass(
        &self,
        memory: &mut HierarchicalMemory,
        lang: &LogicLanguage,
        tokens: &[u16],
        vocab: Option<&[String]>,
        log: Option<&Logger>,
    ) -> TrainReport {
        let mut report = TrainReport::new();
        let total = tokens.len();
        let mut shared_thoughts: Vec<Thought> = Vec::new();
        let mut chunk_count = 0u32;

        let emit = |msg: &str| {
            if let Some(l) = log { l.log(msg); } else { println!("{}", msg); flush(); }
        };

        let mut pos = 0;
        while pos < total.saturating_sub(3) {
            let end = (pos + self.chunk_size).min(total);
            let chunk = &tokens[pos..end];
            chunk_count += 1;

            let collector_out = CollectorWorm::think(
                memory, lang, &shared_thoughts, chunk, self.sample_rate,
            );
            report.add(&collector_out);
            shared_thoughts.extend(collector_out.thoughts);

            if chunk_count % 10 == 0 {
                let analyst_out = AnalystWorm::think(
                    memory, lang, &shared_thoughts, self.max_checks, vocab,
                );
                report.add(&analyst_out);
                shared_thoughts.extend(analyst_out.thoughts);
            }

            if chunk_count % 20 == 0 {
                let skeptic_out = SkepticWorm::think(
                    memory, lang, &shared_thoughts, self.max_checks / 2, vocab,
                );
                report.add(&skeptic_out);
                shared_thoughts.extend(skeptic_out.thoughts);
            }

            if chunk_count % 50 == 0 {
                let explorer_out = ExplorerWorm::think(
                    memory, lang, &shared_thoughts, self.max_checks / 2, vocab,
                );
                report.add(&explorer_out);
                shared_thoughts.extend(explorer_out.thoughts);
            }

            if shared_thoughts.len() > 2000 {
                shared_thoughts.sort_by(|a, b| b.priority.cmp(&a.priority));
                shared_thoughts.truncate(500);
            }

            if chunk_count % 50 == 0 {
                let pct = pos as f64 / total as f64 * 100.0;
                emit(&format!("  {:.0}% | {} | thoughts={}", pct, memory.stats(), shared_thoughts.len()));
            }

            pos = end;
        }

        emit("Final worm cycle...");

        // Логируем общий обмен мыслями перед финальным циклом
        if let Some(l) = log {
            let mut kind_counts: HashMap<Opcode, usize> = HashMap::new();
            for t in &shared_thoughts {
                *kind_counts.entry(t.kind).or_insert(0) += 1;
            }
            let summary: Vec<(String, usize)> = kind_counts.iter()
                .map(|(k, &c)| (format!("{:?}", k), c))
                .collect();
            l.worm_thoughts("SharedThoughts", &summary);
        }

        let analyst_out = AnalystWorm::think(
            memory, lang, &shared_thoughts, self.max_checks * 3, vocab,
        );
        for line in &analyst_out.trace { emit(&format!("  [Analyst] {}", line)); }
        report.add(&analyst_out);
        shared_thoughts.extend(analyst_out.thoughts);

        let skeptic_out = SkepticWorm::think(
            memory, lang, &shared_thoughts, self.max_checks * 2, vocab,
        );
        for line in &skeptic_out.trace { emit(&format!("  [Skeptic] {}", line)); }
        report.add(&skeptic_out);
        shared_thoughts.extend(skeptic_out.thoughts);

        let explorer_out = ExplorerWorm::think(
            memory, lang, &shared_thoughts, self.max_checks * 2, vocab,
        );
        for line in &explorer_out.trace { emit(&format!("  [Explorer] {}", line)); }
        report.add(&explorer_out);

        // Teacher
        let asks: Vec<Thought> = shared_thoughts.iter()
            .filter(|t| t.kind == Opcode::Ask)
            .cloned()
            .collect();

        if !asks.is_empty() {
            let n_help = asks.len().min(self.max_checks);
            let (confirmed, taught) = teacher_respond(memory, lang, &asks, n_help, tokens);
            emit(&format!("  [Teacher] confirmed={}, taught new={}", confirmed, taught));
            report.teacher_confirmed += confirmed;
            report.teacher_taught += taught;
        }

        // Итоговая статистика мыслей
        if let Some(l) = log {
            let total_thoughts = shared_thoughts.len();
            let strong = shared_thoughts.iter().filter(|t| t.kind == Opcode::Strong).count();
            let conflicts = shared_thoughts.iter().filter(|t| t.kind == Opcode::Conflict).count();
            let asks_n = shared_thoughts.iter().filter(|t| t.kind == Opcode::Ask).count();
            let missing = shared_thoughts.iter().filter(|t| t.kind == Opcode::Missing).count();
            l.log(&format!("  Thought summary: total={}, STRONG={}, CONFLICT={}, ASK={}, MISSING={}",
                total_thoughts, strong, conflicts, asks_n, missing));
        }

        report
    }
}

// ============================================================
// Teacher
// ============================================================

fn teacher_respond(
    memory: &mut HierarchicalMemory,
    lang: &LogicLanguage,
    asks: &[Thought],
    n_help: usize,
    tokens: &[u16],
) -> (u32, u32) {
    let mut confirmed = 0u32;
    let mut taught = 0u32;

    for ask in asks.iter().take(n_help) {
        let question = lang.extract_ask_content(&ask.vec);

        if let Some((fact_idx, _)) = memory.facts.find_nearest(&question) {
            let ctx = memory.facts.entries[fact_idx].context_tokens.clone();
            if ctx.len() >= 3 {
                if let Some((succ, count)) = teacher_lookup(tokens, ctx[0], ctx[1], ctx[2]) {
                    memory.facts.entries[fact_idx].confirm_successor(succ, count.min(5));
                    confirmed += 1;
                }
            }
        } else {
            let (nearest_tok, _) = memory.nearest_word(&question);
            let vs = memory.vocab_size;
            for i in 2..tokens.len().saturating_sub(1).min(100_000) {
                if tokens[i] == nearest_tok {
                    let (t0, t1, t2, succ) = (tokens[i - 2], tokens[i - 1], tokens[i], tokens[i + 1]);
                    if t0 as usize >= vs || t1 as usize >= vs || succ as usize >= vs { continue; }
                    let query = memory.make_trigram_query(t0, t1, t2);
                    if memory.facts.find_matching(&query).is_none() {
                        let mut entry = MemoryEntry::with_context(query, 0, vec![t0, t1, t2]);
                        entry.add_successor(succ);
                        memory.facts.store(entry);
                        taught += 1;
                        break;
                    }
                }
            }
        }
    }

    (confirmed, taught)
}

fn teacher_lookup(tokens: &[u16], t0: u16, t1: u16, t2: u16) -> Option<(u16, u32)> {
    let mut counts: HashMap<u16, u32> = HashMap::new();
    let step = (tokens.len() / 500_000).max(1);
    for i in (2..tokens.len().saturating_sub(1)).step_by(step) {
        if tokens[i - 2] == t0 && tokens[i - 1] == t1 && tokens[i] == t2 {
            *counts.entry(tokens[i + 1]).or_insert(0) += 1;
        }
    }
    counts.into_iter().max_by_key(|(_, c)| *c)
}

// ============================================================
// TrainReport
// ============================================================

pub struct TrainReport {
    pub total_stored: usize,
    pub total_strengthened: usize,
    pub total_weakened: usize,
    pub total_asked: usize,
    pub total_scanned: usize,
    pub teacher_confirmed: u32,
    pub teacher_taught: u32,
}

impl TrainReport {
    fn new() -> Self {
        TrainReport {
            total_stored: 0, total_strengthened: 0, total_weakened: 0,
            total_asked: 0, total_scanned: 0, teacher_confirmed: 0, teacher_taught: 0,
        }
    }
    fn add(&mut self, out: &WormOutput) {
        self.total_stored += out.stored;
        self.total_strengthened += out.strengthened;
        self.total_weakened += out.weakened;
        self.total_asked += out.asked;
        self.total_scanned += out.scanned;
    }
    pub fn print(&self) {
        println!("{}", self.to_string());
    }

    pub fn to_string(&self) -> String {
        format!(
            "Worm Report: scanned={}, stored={}, strengthened={}, weakened={}, asked={}, teacher={}c/{}t",
            self.total_scanned, self.total_stored, self.total_strengthened,
            self.total_weakened, self.total_asked,
            self.teacher_confirmed, self.teacher_taught,
        )
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collector_stores_facts() {
        let mut memory = HierarchicalMemory::new(256, 100);
        let lang = LogicLanguage::new(256);
        let data: Vec<u16> = (0..200).map(|i| (i % 100) as u16).collect();
        let out = CollectorWorm::think(&mut memory, &lang, &[], &data, 1);
        assert!(out.stored > 0);
        assert!(memory.facts.len() > 0);
    }

    #[test]
    fn test_analyst_uses_token_index() {
        let mut memory = HierarchicalMemory::new(256, 100);
        let lang = LogicLanguage::new(256);
        let data: Vec<u16> = (0..500).map(|i| (i % 100) as u16).collect();
        CollectorWorm::think(&mut memory, &lang, &[], &data, 1);

        // Token index should be populated
        assert!(!memory.facts.token_index.is_empty());

        let out = AnalystWorm::think(&mut memory, &lang, &[], 100, None);
        assert!(out.scanned > 0);
        // With token-based neighbors, should have better agreement
        assert!(!out.thoughts.is_empty());
    }

    #[test]
    fn test_explorer_creates_bigram_rules() {
        let mut memory = HierarchicalMemory::new(256, 100);
        let lang = LogicLanguage::new(256);
        // Repeating patterns to create bigram rules
        let data: Vec<u16> = (0..1000).map(|i| (i % 50) as u16).collect();
        CollectorWorm::think(&mut memory, &lang, &[], &data, 1);

        let rules_before = memory.rules.len();
        let out = ExplorerWorm::think(&mut memory, &lang, &[], 500, None);

        // Should create bigram rules
        assert!(memory.rules.len() > rules_before || out.stored > 0);
    }

    #[test]
    fn test_curriculum_collector() {
        let mut memory = HierarchicalMemory::new(256, 100);
        let lang = LogicLanguage::new(256);

        // Create MISSING thoughts
        let missing_thought = Thought::new(
            lang.missing(&memory.word_vec(42).clone()),
            Opcode::Missing, 3, Opcode::SourceExplorer,
        );
        let inbox = vec![missing_thought];

        let data: Vec<u16> = (0..200).map(|i| (i % 100) as u16).collect();
        let out = CollectorWorm::think(&mut memory, &lang, &inbox, &data, 10);

        // With curriculum, should scan more around token 42
        assert!(out.scanned > 0);
    }

    #[test]
    fn test_full_orchestrator() {
        let mut memory = HierarchicalMemory::new(256, 100);
        let lang = LogicLanguage::new(256);
        let data: Vec<u16> = (0..1000).map(|i| (i % 100) as u16).collect();
        let orch = WormOrchestrator::new(500, 1, 100);
        let report = orch.run_pass(&mut memory, &lang, &data, None, None);
        assert!(report.total_stored > 0);
        assert!(memory.facts.len() > 0);
    }
}
