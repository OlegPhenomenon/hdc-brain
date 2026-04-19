//! Hierarchical Memory System — 5 уровней знаний + token index
//!
//! Уровень 0: СЛОВА     — codebook (токен → вектор)
//! Уровень 1: ФАКТЫ     — bind-пары из данных (наблюдения)
//! Уровень 2: ПРАВИЛА   — логические выводы из фактов (IF → THEN)
//! Уровень 3: АБСТРАКЦИИ — категории, обобщения (bundle групп)
//! Уровень 4: МЕТА      — знание о своих знаниях (самосознание)
//!
//! Token Index: token → list of fact indices (для Analyst и fallback predict)
//! Bigram Index: (t0,t1) → list of fact indices (для каскадного predict)

use crate::binary::*;
use crate::language::Opcode;
use rand::Rng;
use std::collections::HashMap;

// ============================================================
// Memory Entry — единица знания на любом уровне
// ============================================================

#[derive(Clone)]
pub struct MemoryEntry {
    pub vec: BinaryVec,
    pub successors: HashMap<u16, u32>,
    pub total_count: u32,
    #[allow(dead_code)]
    pub source: u8,
    pub context_tokens: Vec<u16>,
}

impl MemoryEntry {
    #[allow(dead_code)]
    pub fn new(vec: BinaryVec, source: u8) -> Self {
        MemoryEntry {
            vec, successors: HashMap::new(), total_count: 0,
            source, context_tokens: Vec::new(),
        }
    }

    pub fn with_context(vec: BinaryVec, source: u8, context_tokens: Vec<u16>) -> Self {
        MemoryEntry {
            vec, successors: HashMap::new(), total_count: 0,
            source, context_tokens,
        }
    }

    pub fn add_successor(&mut self, token: u16) {
        *self.successors.entry(token).or_insert(0) += 1;
        self.total_count += 1;
    }

    pub fn confirm_successor(&mut self, token: u16, times: u32) {
        *self.successors.entry(token).or_insert(0) += times;
        self.total_count += times;
    }

    pub fn top_successors(&self, k: usize) -> Vec<(u16, u32)> {
        let mut pairs: Vec<(u16, u32)> = self.successors.iter()
            .map(|(&tok, &cnt)| (tok, cnt))
            .collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(k);
        pairs
    }

    pub fn top1(&self) -> Option<(u16, u32)> {
        self.successors.iter()
            .max_by_key(|(_, &cnt)| cnt)
            .map(|(&tok, &cnt)| (tok, cnt))
    }

}

// ============================================================
// Memory Level — один уровень иерархии
// ============================================================

pub struct MemoryLevel {
    #[allow(dead_code)]
    pub name: &'static str,
    pub entries: Vec<MemoryEntry>,
    pub lsh_index: HashMap<u32, Vec<usize>>,
    pub lsh_bits: usize,
    #[allow(dead_code)]
    pub dim: usize,
    /// Token index: token → list of entry indices containing that token in context
    pub token_index: HashMap<u16, Vec<usize>>,
    /// Bigram index: (t0, t1) → list of entry indices
    pub bigram_index: HashMap<(u16, u16), Vec<usize>>,
}

impl MemoryLevel {
    pub fn new(name: &'static str, dim: usize, lsh_bits: usize) -> Self {
        MemoryLevel {
            name,
            entries: Vec::new(),
            lsh_index: HashMap::new(),
            lsh_bits,
            dim,
            token_index: HashMap::new(),
            bigram_index: HashMap::new(),
        }
    }

    /// Добавить запись + обновить все индексы.
    pub fn store(&mut self, entry: MemoryEntry) -> usize {
        let idx = self.entries.len();
        let hash = lsh_hash(&entry.vec, self.lsh_bits);
        self.lsh_index.entry(hash).or_default().push(idx);

        // Token index
        for &tok in &entry.context_tokens {
            self.token_index.entry(tok).or_default().push(idx);
        }

        // Bigram index (из context_tokens) — без дубликатов
        if entry.context_tokens.len() >= 2 {
            let ct = &entry.context_tokens;
            for w in ct.windows(2) {
                self.bigram_index.entry((w[0], w[1])).or_default().push(idx);
            }
        }

        self.entries.push(entry);
        idx
    }

    /// Найти факты с общими context_tokens (для Analyst).
    /// Возвращает до max_results записей, у которых >= min_common токенов совпадают.
    pub fn find_by_shared_tokens(&self, tokens: &[u16], exclude_idx: usize, max_results: usize) -> Vec<(usize, usize)> {
        let mut candidate_counts: HashMap<usize, usize> = HashMap::new();

        for &tok in tokens {
            if let Some(indices) = self.token_index.get(&tok) {
                // Пропустить слишком частые токены (неинформативные)
                if indices.len() > 5000 { continue; }
                for &idx in indices {
                    if idx != exclude_idx {
                        *candidate_counts.entry(idx).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut results: Vec<(usize, usize)> = candidate_counts.into_iter().collect();
        results.sort_by(|a, b| b.1.cmp(&a.1));
        results.truncate(max_results);
        results
    }

    /// Найти факты по биграмме (t0, t1). Для каскадного predict.
    pub fn find_by_bigram(&self, t0: u16, t1: u16, max_results: usize) -> Vec<usize> {
        match self.bigram_index.get(&(t0, t1)) {
            Some(indices) => indices.iter().take(max_results).cloned().collect(),
            None => Vec::new(),
        }
    }

    /// Найти факты по униграмме (token). Для каскадного predict.
    /// Пропускает слишком частые токены (> 10K записей — неинформативные).
    pub fn find_by_token(&self, tok: u16, max_results: usize) -> Vec<usize> {
        match self.token_index.get(&tok) {
            Some(indices) if indices.len() <= 10_000 =>
                indices.iter().take(max_results).cloned().collect(),
            _ => Vec::new(),
        }
    }

    pub fn find_matching(&self, query: &BinaryVec) -> Option<usize> {
        let hash = lsh_hash(query, self.lsh_bits);
        let threshold = query.dim as i32 / 3;
        if let Some(indices) = self.lsh_index.get(&hash) {
            for &idx in indices {
                if self.entries[idx].vec.similarity(query) > threshold {
                    return Some(idx);
                }
            }
        }
        None
    }

    pub fn find_exact(&self, query: &BinaryVec) -> Option<(usize, i32)> {
        let hash = lsh_hash(query, self.lsh_bits);
        self.find_in_bucket(hash, query)
    }

    pub fn find_nearest(&self, query: &BinaryVec) -> Option<(usize, i32)> {
        let hash = lsh_hash(query, self.lsh_bits);
        let mut best_idx = None;
        let mut best_sim = i32::MIN;

        if let Some((idx, sim)) = self.find_in_bucket(hash, query) {
            if sim > best_sim { best_sim = sim; best_idx = Some(idx); }
        }

        for bit in 0..self.lsh_bits.min(16) {
            let neighbor_hash = hash ^ (1 << bit);
            if let Some((idx, sim)) = self.find_in_bucket(neighbor_hash, query) {
                if sim > best_sim { best_sim = sim; best_idx = Some(idx); }
            }
        }

        best_idx.map(|idx| (idx, best_sim))
    }

    pub fn find_top_k(&self, query: &BinaryVec, k: usize) -> Vec<(usize, i32)> {
        let hash = lsh_hash(query, self.lsh_bits);
        let mut candidates: Vec<(usize, i32)> = Vec::new();

        let mut hashes = vec![hash];
        for bit in 0..self.lsh_bits.min(16) {
            hashes.push(hash ^ (1 << bit));
        }

        for h in &hashes {
            if let Some(indices) = self.lsh_index.get(h) {
                for &idx in indices {
                    let sim = self.entries[idx].vec.similarity(query);
                    candidates.push((idx, sim));
                }
            }
        }

        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        candidates.truncate(k);
        candidates
    }

    pub fn len(&self) -> usize { self.entries.len() }

    fn find_in_bucket(&self, hash: u32, query: &BinaryVec) -> Option<(usize, i32)> {
        let indices = self.lsh_index.get(&hash)?;
        let mut best_idx = None;
        let mut best_sim = i32::MIN;
        for &idx in indices {
            let sim = self.entries[idx].vec.similarity(query);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(idx);
            }
        }
        best_idx.map(|idx| (idx, best_sim))
    }
}

// ============================================================
// Relations — семантические связи (уровень 5)
// bind(subject, bind(role, object)) → один BinaryVec
// ============================================================

/// Источник связи
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum RelationSource {
    Data,
    Inference,
    Operator,
}

/// Семантическая связь: subject --role--> object
#[derive(Clone)]
pub struct RelationEntry {
    pub vec: BinaryVec,          // bind(subject, bind(role, object))
    pub subject: u16,            // токен субъекта
    pub role: Opcode,            // тип связи
    pub object: u16,             // токен объекта
    pub confidence: u32,         // сколько раз подтверждено
    #[allow(dead_code)]
    pub source: RelationSource,
}

/// Хранилище семантических связей с LSH индексом
#[allow(dead_code)]
pub struct RelationMemory {
    pub entries: Vec<RelationEntry>,
    lsh_index: HashMap<u32, Vec<usize>>,
    subject_index: HashMap<u16, Vec<usize>>,
    object_index: HashMap<u16, Vec<usize>>,
    subject_role_index: HashMap<(u16, u8), Vec<usize>>,
    lsh_bits: usize,
    dim: usize,
    pub max_entries: usize,
}

#[allow(dead_code)]
impl RelationMemory {
    pub fn new(dim: usize, max_entries: usize) -> Self {
        RelationMemory {
            entries: Vec::new(),
            lsh_index: HashMap::new(),
            subject_index: HashMap::new(),
            object_index: HashMap::new(),
            subject_role_index: HashMap::new(),
            lsh_bits: 14,
            dim,
            max_entries,
        }
    }

    /// Добавить связь. Если уже существует (subject, role, object) — увеличить confidence.
    pub fn store(&mut self, entry: RelationEntry) -> usize {
        // Дедупликация: ищем существующую (subject, role, object)
        let role_idx = entry.role.index() as u8;
        if let Some(indices) = self.subject_role_index.get(&(entry.subject, role_idx)) {
            for &idx in indices {
                if self.entries[idx].object == entry.object {
                    self.entries[idx].confidence += entry.confidence;
                    return idx;
                }
            }
        }

        if self.entries.len() >= self.max_entries {
            return usize::MAX; // не добавляем
        }

        let idx = self.entries.len();
        let hash = lsh_hash(&entry.vec, self.lsh_bits);
        self.lsh_index.entry(hash).or_default().push(idx);
        self.subject_index.entry(entry.subject).or_default().push(idx);
        self.object_index.entry(entry.object).or_default().push(idx);
        self.subject_role_index.entry((entry.subject, role_idx)).or_default().push(idx);
        self.entries.push(entry);
        idx
    }

    /// Найти связи по субъекту
    pub fn find_by_subject(&self, subject: u16, max_results: usize) -> Vec<usize> {
        match self.subject_index.get(&subject) {
            Some(indices) => indices.iter().take(max_results).cloned().collect(),
            None => Vec::new(),
        }
    }

    /// Найти связи по объекту (обратный поиск)
    pub fn find_by_object(&self, object: u16, max_results: usize) -> Vec<usize> {
        match self.object_index.get(&object) {
            Some(indices) => indices.iter().take(max_results).cloned().collect(),
            None => Vec::new(),
        }
    }

    /// QUERY через HDC: unbind(stored_relation, query) → найти ближайшее.
    /// query = bind(subject, role_vec) → ищет object.
    /// Для Relations используем unbind + nearest_word для точного извлечения.
    pub fn query_hdc(&self, query: &BinaryVec, _words: &[BinaryVec], top_k: usize) -> Vec<(u16, i32)> {
        if self.entries.is_empty() { return Vec::new(); }

        let mut candidates: Vec<(usize, i32)> = Vec::new();

        // Для малого числа записей (<10K) — полный скан эффективнее LSH
        if self.entries.len() < 10_000 {
            for (idx, entry) in self.entries.iter().enumerate() {
                let sim = entry.vec.similarity(query);
                if sim > 0 {
                    candidates.push((idx, sim));
                }
            }
        } else {
            // LSH поиск + соседи
            let hash = lsh_hash(query, self.lsh_bits);
            let mut hashes = vec![hash];
            for bit in 0..self.lsh_bits.min(14) {
                hashes.push(hash ^ (1 << bit));
            }
            for h in &hashes {
                if let Some(indices) = self.lsh_index.get(h) {
                    for &idx in indices {
                        let sim = self.entries[idx].vec.similarity(query);
                        if sim > 0 {
                            candidates.push((idx, sim));
                        }
                    }
                }
            }
        }

        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        candidates.truncate(top_k);

        candidates.iter()
            .map(|&(idx, sim)| (self.entries[idx].object, sim))
            .collect()
    }

    /// Найти связи по subject+role
    pub fn find_by_subject_role(&self, subject: u16, role: Opcode, max_results: usize) -> Vec<usize> {
        let role_idx = role.index() as u8;
        match self.subject_role_index.get(&(subject, role_idx)) {
            Some(indices) => indices.iter().take(max_results).cloned().collect(),
            None => Vec::new(),
        }
    }

    pub fn len(&self) -> usize { self.entries.len() }
}

// ============================================================
// HierarchicalMemory
// ============================================================

pub struct HierarchicalMemory {
    pub words: Vec<BinaryVec>,
    pub facts: MemoryLevel,
    pub rules: MemoryLevel,
    pub abstractions: MemoryLevel,
    pub meta: MemoryLevel,
    pub relations: RelationMemory,
    pub dim: usize,
    pub vocab_size: usize,
}

impl HierarchicalMemory {
    pub fn new(dim: usize, vocab_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let words: Vec<BinaryVec> = (0..vocab_size)
            .map(|_| BinaryVec::random(dim, &mut rng))
            .collect();

        HierarchicalMemory {
            words,
            facts: MemoryLevel::new("facts", dim, 16),
            rules: MemoryLevel::new("rules", dim, 14),
            abstractions: MemoryLevel::new("abstractions", dim, 12),
            meta: MemoryLevel::new("meta", dim, 10),
            relations: RelationMemory::new(dim, 500_000),
            dim,
            vocab_size,
        }
    }

    #[inline]
    pub fn word_vec(&self, token: u16) -> &BinaryVec {
        &self.words[token as usize]
    }

    pub fn make_trigram_query(&self, t0: u16, t1: u16, t2: u16) -> BinaryVec {
        self.words[t0 as usize]
            .bind(&self.words[t1 as usize].permute(1))
            .bind(&self.words[t2 as usize].permute(2))
    }

    /// Bigram query: два токена → вектор
    pub fn make_bigram_query(&self, t0: u16, t1: u16) -> BinaryVec {
        self.words[t0 as usize]
            .bind(&self.words[t1 as usize].permute(1))
    }

    pub fn make_context_bundle(&self, tokens: &[u16], end_pos: usize, window: usize) -> BinaryVec {
        let mut acc = BundleAccumulator::new(self.dim);
        let start = if end_pos >= window { end_pos - window } else { 0 };

        for i in start..end_pos {
            let tok = tokens[i] as usize;
            if tok >= self.vocab_size { continue; }
            let offset = end_pos - i;
            let positioned = self.words[tok].permute(offset % self.dim);
            let repeats = if offset <= 3 { 3 } else if offset <= 10 { 2 } else { 1 };
            for _ in 0..repeats {
                acc.add(&positioned);
            }
        }

        acc.to_binary()
    }

    pub fn nearest_word(&self, query: &BinaryVec) -> (u16, i32) {
        let mut best_token = 0u16;
        let mut best_sim = i32::MIN;
        for (i, w) in self.words.iter().enumerate() {
            let sim = query.similarity(w);
            if sim > best_sim {
                best_sim = sim;
                best_token = i as u16;
            }
        }
        (best_token, best_sim)
    }

    pub fn nearest_words(&self, query: &BinaryVec, k: usize) -> Vec<(u16, i32)> {
        let mut pairs: Vec<(u16, i32)> = self.words.iter().enumerate()
            .map(|(i, w)| (i as u16, query.similarity(w)))
            .collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(k);
        pairs
    }

    pub fn build_semantic_codebook(&mut self, tokens: &[u16], window: usize) {
        use rayon::prelude::*;

        let dim = self.dim;
        let vocab_size = self.vocab_size;

        let mut context_accumulators: Vec<BundleAccumulator> = (0..vocab_size)
            .map(|_| BundleAccumulator::new(dim))
            .collect();
        let mut token_counts = vec![0u32; vocab_size];

        for i in window..tokens.len().saturating_sub(window) {
            let tok = tokens[i] as usize;
            if tok >= vocab_size { continue; }
            token_counts[tok] += 1;

            for offset in 1..=window {
                if i >= offset {
                    let left = tokens[i - offset] as usize;
                    if left < vocab_size {
                        context_accumulators[tok].add(&self.words[left]);
                    }
                }
                if i + offset < tokens.len() {
                    let right = tokens[i + offset] as usize;
                    if right < vocab_size {
                        context_accumulators[tok].add(&self.words[right]);
                    }
                }
            }
        }

        let max_count = *token_counts.iter().max().unwrap_or(&1) as f64;

        let new_words: Vec<BinaryVec> = (0..vocab_size)
            .into_par_iter()
            .map(|i| {
                if context_accumulators[i].count < 5 {
                    return self.words[i].clone();
                }

                let semantic = context_accumulators[i].to_binary();
                let freq_ratio = token_counts[i] as f64 / max_count;
                let semantic_weight = 1.0 - freq_ratio * 0.75;

                let mut rng = rand::thread_rng();
                let mut blended = BinaryVec::zeros(dim);
                for bit in 0..dim {
                    let use_semantic = rng.gen::<f64>() < semantic_weight;
                    if use_semantic {
                        blended.set_bit(bit, semantic.get_bit(bit));
                    } else {
                        blended.set_bit(bit, self.words[i].get_bit(bit));
                    }
                }
                blended
            })
            .collect();

        self.words = new_words;
    }

    /// Сохранить codebook (words) в файл
    pub fn save_codebook(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        // Header: dim, vocab_size
        f.write_all(&(self.dim as u32).to_le_bytes())?;
        f.write_all(&(self.vocab_size as u32).to_le_bytes())?;
        // Words: каждый word = dim/64 u64 values
        for word in &self.words {
            for &val in &word.data {
                f.write_all(&val.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Загрузить codebook из файла. Возвращает true если успешно.
    pub fn load_codebook(&mut self, path: &str) -> bool {
        use std::io::Read;
        let mut f = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let mut buf4 = [0u8; 4];
        if f.read_exact(&mut buf4).is_err() { return false; }
        let dim = u32::from_le_bytes(buf4) as usize;
        if f.read_exact(&mut buf4).is_err() { return false; }
        let vs = u32::from_le_bytes(buf4) as usize;

        if dim != self.dim || vs != self.vocab_size { return false; }

        let n_u64 = dim / 64;
        let mut buf8 = [0u8; 8];
        for i in 0..vs {
            let mut data = vec![0u64; n_u64];
            for j in 0..n_u64 {
                if f.read_exact(&mut buf8).is_err() { return false; }
                data[j] = u64::from_le_bytes(buf8);
            }
            self.words[i] = BinaryVec { data, dim };
        }
        true
    }

    /// Hebbian codebook update (v19-plan):
    /// При правильном predict: codebook[token] = bundle(old × inertia, context)
    /// При неправильном predict: codebook[expected] = bundle(old × inertia, context) (слабее)
    /// Инерция 8:1 — codebook меняется медленно.
    pub fn hebbian_codebook_update(&mut self, token: u16, context: &BinaryVec, correct: bool) {
        let tok = token as usize;
        if tok >= self.vocab_size { return; }

        let dim = self.dim;
        let inertia = if correct { 8 } else { 12 }; // anti-Hebbian слабее

        let mut acc = BundleAccumulator::new(dim);
        for _ in 0..inertia {
            acc.add(&self.words[tok]); // old × inertia
        }
        acc.add(context); // new context × 1

        self.words[tok] = acc.to_binary();
    }

    /// Создать HDC вектор для связи: bind(subject_vec, bind(role_vec, object_vec))
    pub fn make_relation_vec(&self, subject: u16, role_vec: &BinaryVec, object: u16) -> BinaryVec {
        let subj_vec = &self.words[subject as usize];
        let obj_vec = &self.words[object as usize];
        subj_vec.bind(&role_vec.bind(obj_vec))
    }

    /// Создать query для поиска объекта: bind(subject_vec, role_vec)
    #[allow(dead_code)]
    pub fn make_relation_query(&self, subject: u16, role_vec: &BinaryVec) -> BinaryVec {
        self.words[subject as usize].bind(role_vec)
    }

    pub fn stats(&self) -> String {
        format!(
            "Memory: words={}, facts={}, rules={}, abstractions={}, meta={}, relations={}",
            self.vocab_size,
            self.facts.len(),
            self.rules.len(),
            self.abstractions.len(),
            self.meta.len(),
            self.relations.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_memory_creation() {
        let mem = HierarchicalMemory::new(256, 100);
        assert_eq!(mem.words.len(), 100);
        assert_eq!(mem.facts.len(), 0);
    }

    #[test]
    fn test_fact_store_and_find() {
        let mem = HierarchicalMemory::new(256, 100);
        let mut facts = MemoryLevel::new("test", 256, 8);

        let entry = MemoryEntry::new(mem.words[0].clone(), 0);
        facts.store(entry);

        let result = facts.find_exact(&mem.words[0]);
        assert!(result.is_some());
        let (idx, sim) = result.unwrap();
        assert_eq!(idx, 0);
        assert_eq!(sim, 256);
    }

    #[test]
    fn test_store_and_find_with_successor() {
        let mem = HierarchicalMemory::new(256, 100);
        let mut facts = MemoryLevel::new("test", 256, 8);

        let trigram = mem.make_trigram_query(10, 20, 30);
        let mut entry = MemoryEntry::with_context(trigram.clone(), 0, vec![10, 20, 30]);
        entry.add_successor(42);
        entry.add_successor(42);
        entry.add_successor(7);
        facts.store(entry);

        let result = facts.find_exact(&trigram);
        assert!(result.is_some());
        let (idx, _) = result.unwrap();
        let top = facts.entries[idx].top_successors(2);
        assert_eq!(top[0].0, 42);
        assert_eq!(top[0].1, 2);
    }

    #[test]
    fn test_find_matching() {
        let mem = HierarchicalMemory::new(256, 100);
        let mut facts = MemoryLevel::new("test", 256, 8);

        let trigram = mem.make_trigram_query(10, 20, 30);
        let entry = MemoryEntry::with_context(trigram.clone(), 0, vec![10, 20, 30]);
        facts.store(entry);

        assert!(facts.find_matching(&trigram).is_some());
    }

    #[test]
    fn test_token_index() {
        let mem = HierarchicalMemory::new(256, 100);
        let mut facts = MemoryLevel::new("test", 256, 8);

        let trigram = mem.make_trigram_query(10, 20, 30);
        let entry = MemoryEntry::with_context(trigram, 0, vec![10, 20, 30]);
        facts.store(entry);

        // Token 20 should find this entry
        let results = facts.find_by_shared_tokens(&[20], 999, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0); // index 0
    }

    #[test]
    fn test_relation_store_and_find() {
        use crate::language::{LogicLanguage, Opcode};
        let mem = HierarchicalMemory::new(256, 100);
        let lang = LogicLanguage::new(256);
        let mut relations = RelationMemory::new(256, 1000);

        let role_vec = lang.op(Opcode::Similar);
        let rel_vec = mem.make_relation_vec(10, role_vec, 20);
        let entry = RelationEntry {
            vec: rel_vec,
            subject: 10,
            role: Opcode::Similar,
            object: 20,
            confidence: 1,
            source: RelationSource::Data,
        };
        let idx = relations.store(entry);
        assert_eq!(idx, 0);
        assert_eq!(relations.len(), 1);

        // Дедупликация: та же связь → увеличить confidence
        let rel_vec2 = mem.make_relation_vec(10, role_vec, 20);
        let entry2 = RelationEntry {
            vec: rel_vec2,
            subject: 10,
            role: Opcode::Similar,
            object: 20,
            confidence: 1,
            source: RelationSource::Data,
        };
        let idx2 = relations.store(entry2);
        assert_eq!(idx2, 0); // тот же индекс
        assert_eq!(relations.len(), 1); // не добавлена новая
        assert_eq!(relations.entries[0].confidence, 2); // confidence++

        // Поиск по subject
        let found = relations.find_by_subject(10, 10);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], 0);

        // Поиск по object
        let found_obj = relations.find_by_object(20, 10);
        assert_eq!(found_obj.len(), 1);
    }

    #[test]
    fn test_relation_subject_role_query() {
        use crate::language::{LogicLanguage, Opcode};
        let mem = HierarchicalMemory::new(4096, 100);
        let lang = LogicLanguage::new(4096);
        let mut relations = RelationMemory::new(4096, 1000);

        // Создать связь: 10 --Similar--> 20
        let role_vec = lang.op(Opcode::Similar);
        let rel_vec = mem.make_relation_vec(10, role_vec, 20);
        relations.store(RelationEntry {
            vec: rel_vec,
            subject: 10,
            role: Opcode::Similar,
            object: 20,
            confidence: 5,
            source: RelationSource::Data,
        });

        // Поиск через индекс: subject=10, role=Similar → должен найти object=20
        let results = relations.find_by_subject_role(10, Opcode::Similar, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(relations.entries[results[0]].object, 20);

        // HDC unbind: unbind(relation_vec, bind(subject, role)) → recovered ≈ object_vec
        let query = mem.make_relation_query(10, role_vec);
        let rel_entry = &relations.entries[0];
        let recovered = rel_entry.vec.unbind(&query);
        // recovered должен быть похож на word[20]
        let sim = recovered.similarity(&mem.words[20]);
        assert!(sim > (4096i32 / 3), "HDC unbind should recover object. sim={}", sim);
    }

    #[test]
    fn test_bigram_index() {
        let mem = HierarchicalMemory::new(256, 100);
        let mut facts = MemoryLevel::new("test", 256, 8);

        let trigram = mem.make_trigram_query(10, 20, 30);
        let entry = MemoryEntry::with_context(trigram, 0, vec![10, 20, 30]);
        facts.store(entry);

        let results = facts.find_by_bigram(10, 20, 10);
        assert!(!results.is_empty());
    }
}
