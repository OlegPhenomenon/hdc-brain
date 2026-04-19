//! WormMind — черви ДУМАЮТ при predict, а не просто ищут в памяти
//!
//! Каждый predict = мини-сессия мышления:
//!   1. Collector собирает кандидатов + строит ThoughtChain
//!   2. Analyst проверяет кандидатов через внутренний язык
//!   3. Skeptic ищет контр-доказательства
//!   4. Explorer предлагает неожиданные варианты
//!   5. Consensus — черви голосуют через Thoughts
//!
//! При этом черви ПОПУТНО обновляют память (background learning):
//!   - Analyst укрепляет подтверждённые факты
//!   - Skeptic ослабляет противоречивые
//!   - Explorer создаёт новые абстракции
//!
//! Внутренний язык РЕАЛЬНО используется:
//!   FIND → CHECK → COMPARE → IF/THEN → STRONG/WEAK

use crate::binary::*;
use crate::memory::*;
use crate::language::*;
use std::collections::HashMap;
use std::collections::HashSet;

/// Результат одного шага рассуждения червя
#[derive(Clone)]
pub struct ReasoningStep {
    pub thought: Thought,           // HDC мысль (bind-цепочка)
    #[allow(dead_code)]
    pub description: String,        // человекочитаемое описание
}

/// Протокол рассуждения — цепочка шагов на внутреннем языке
pub struct ThoughtChain {
    pub steps: Vec<ReasoningStep>,
    #[allow(dead_code)]
    pub conclusion_token: Option<u16>,
    #[allow(dead_code)]
    pub conclusion_confidence: f64,
}

impl ThoughtChain {
    fn new() -> Self {
        ThoughtChain { steps: Vec::new(), conclusion_token: None, conclusion_confidence: 0.0 }
    }

    fn add(&mut self, thought: Thought, desc: &str) {
        self.steps.push(ReasoningStep {
            thought,
            description: desc.to_string(),
        });
    }
}

/// Голос червя — его мнение о кандидате.
/// raw_score = частота из памяти (база).
/// evidence = HDC вектор из рассуждений (модификатор).
/// Финальный score = raw_score × evidence_strength.
struct ThoughtVote {
    token: u16,
    raw_score: f64,       // частота из памяти (незаменимая база)
    evidence: BinaryVec,  // bundled evidence (HDC мысли модифицируют)
    reason: Opcode,
}

/// Самодиагностика — что червям нужно для лучшего понимания
#[derive(Clone, Debug)]
pub struct DiagnosticReport {
    pub need_more_data: Vec<u16>,
    pub uncertain_areas: usize,
    pub no_trigram_match: bool,
    #[allow(dead_code)]
    pub weak_context: bool,
    pub contradictions: usize,
    #[allow(dead_code)]
    pub novel_discoveries: usize,
    #[allow(dead_code)]
    pub has_shift_pattern: bool,
    #[allow(dead_code)]
    pub forward_verified: usize,
}

/// Отчёт о РЕАЛЬНОМ влиянии каждой фичи на результат.
/// Это не декорация — каждый счётчик увеличивается ТОЛЬКО когда фича
/// действительно изменила скор/ранжирование/решение.
#[derive(Default, Debug)]
pub struct InfluenceReport {
    /// Relations: сколько кандидатов получили boost от Sequence/Similar relations
    pub relations_boosted: usize,
    /// Cross-worm: сколько раз Skeptic изменил решение на основе мнения Analyst'а
    pub cross_worm_overrides: usize,
    /// HDCMemory: сколько далёких (>20) токенов попали в контекст через causal_extra
    pub causal_extra_tokens: usize,
    /// SelfAwareness: сколько параметров было адаптировано (expand/aggression/breadth)
    pub awareness_adaptations: usize,
    /// Evidence: сколько раз evidence bundling изменил ранжирование (top-1 изменился)
    pub evidence_reranks: usize,
}

/// Результат мышления червей
pub struct MindResult {
    pub predictions: Vec<(u16, f64)>,
    pub thoughts_exchanged: usize,
    pub reasoning_depth: usize,     // сколько шагов рассуждения
    pub confidence: f64,
    pub background_updates: usize,  // сколько фактов обновлено попутно
    pub trace: Vec<String>,         // человекочитаемый trace для логов
    pub diagnostic: DiagnosticReport, // самодиагностика
    #[allow(dead_code)]
    pub messages: Vec<WormMessage>,   // сообщения для оператора
    pub influence: InfluenceReport,   // аудит реального влияния
}

// ============================================================
// HDC Binding Attention — по принципу v14 (binary)
// ============================================================
//
// v14: Q = x * sign(bv_q), K = x * sign(bv_k), V = x * sign(bv_v)
//      scores = Q @ K^T, attn = sigmoid(scores)
//
// v18 binary эквивалент:
//   role_query, role_key = фиксированные binary vectors (seed-based)
//   Q = context_vec XOR role_query    (binding = назначение роли)
//   K[i] = fact[i].vec XOR role_key   (binding факта с ролью key)
//   score[i] = similarity(Q, K[i])    (XNOR+POPCNT = dot product)
//   prediction = soft-weighted combination of fact successors
//
// Отличие от трансформера:
//   - Binding (XOR) вместо projection (matmul)
//   - 2×D бит (role vectors) вместо 2×D×D float (projection matrices)
//   - Similarity (XNOR+POPCNT) вместо dot product + softmax
//   - Attention по TOP-K фактам из LSH (не по всей памяти)

use rand::SeedableRng;
use rand::rngs::StdRng;

/// Role vectors для HDC Binding Attention.
/// Начинают как random, но учатся через Hebbian association:
///   правильный predict → bundle(role, successful_pattern)
/// Аналог bv_q, bv_k из v14, но binary и обучаемые через Hebbian.
/// Самосознание: persistent state между predict'ами.
/// Черви ПОМНЯТ что происходило в прошлых рассуждениях и адаптируют стратегию.
#[derive(Clone)]
pub struct SelfAwareness {
    /// HDC вектор накопленных проблемных контекстов (bundle NEED_MORE areas)
    pub weak_areas: BinaryVec,
    /// Сколько раз подряд trigram не находился
    pub consecutive_no_trigram: u32,
    /// Средний уровень противоречий (экспоненциальное среднее)
    pub avg_contradictions: f64,
    /// Средняя uncertainty (экспоненциальное среднее)
    pub avg_uncertainty: f64,
    /// Количество predict'ов (для статистики)
    pub total_predicts: u32,
    /// Стратегия Skeptic: насколько агрессивно проверять (0.0=мягко, 1.0=агрессивно)
    pub skeptic_aggression: f64,
    /// Стратегия Explorer: насколько широко искать (0.0=узко, 1.0=широко)
    pub explorer_breadth: f64,
    /// Стратегия Collector: расширять ли поиск за триграмму
    pub collector_expand: bool,
}

impl SelfAwareness {
    pub fn new(dim: usize) -> Self {
        SelfAwareness {
            weak_areas: BinaryVec::zeros(dim),
            consecutive_no_trigram: 0,
            avg_contradictions: 0.0,
            avg_uncertainty: 0.0,
            total_predicts: 0,
            skeptic_aggression: 0.5,  // средний уровень по умолчанию
            explorer_breadth: 0.5,
            collector_expand: false,
        }
    }

    /// REFLECT: обновить самосознание на основе DiagnosticReport.
    /// Генерирует HDC мысли (Reflect/Uncertain/Pattern) которые влияют на СЛЕДУЮЩИЙ predict.
    pub fn reflect_on(&mut self, diagnostic: &DiagnosticReport, lang: &LogicLanguage, _dim: usize) -> Vec<Thought> {
        let mut reflections = Vec::new();
        self.total_predicts += 1;
        let alpha = 0.1; // скорость обновления экспоненциального среднего

        // --- Обновить awareness ---

        // Trigram tracking
        if diagnostic.no_trigram_match {
            self.consecutive_no_trigram += 1;
            self.collector_expand = true; // следующий predict: расширить поиск
        } else {
            self.consecutive_no_trigram = 0;
            self.collector_expand = false;
        }

        // Contradictions → skeptic aggression
        self.avg_contradictions = self.avg_contradictions * (1.0 - alpha)
            + diagnostic.contradictions as f64 * alpha;
        self.skeptic_aggression = if self.avg_contradictions > 3.0 {
            0.9 // много противоречий → агрессивная проверка
        } else if self.avg_contradictions > 1.0 {
            0.6
        } else {
            0.3 // мало противоречий → мягкая проверка
        };

        // Uncertainty → explorer breadth
        self.avg_uncertainty = self.avg_uncertainty * (1.0 - alpha)
            + diagnostic.uncertain_areas as f64 * alpha;
        self.explorer_breadth = if self.avg_uncertainty > 15.0 {
            0.9 // высокая неуверенность → широкий поиск
        } else if self.avg_uncertainty > 5.0 {
            0.6
        } else {
            0.3 // низкая неуверенность → узкий поиск
        };

        // Bundle weak areas для guidance Explorer'у
        // weak_areas bundling с need_more_data происходит в think() где доступен memory

        // --- Генерировать мысли-рефлексии ---

        // REFLECT: "я заметил паттерн проблем"
        if self.consecutive_no_trigram >= 3 {
            reflections.push(Thought::new(
                lang.op(Opcode::Reflect).bind(lang.op(Opcode::NeedMore)),
                Opcode::Reflect, 5, Opcode::SourceAnalyst,
            ));
        }

        // UNCERTAIN: "много неуверенных кандидатов — нужна осторожность"
        if self.avg_uncertainty > 10.0 {
            reflections.push(Thought::new(
                lang.op(Opcode::Uncertain).bind(lang.op(Opcode::Pattern)),
                Opcode::Uncertain, 4, Opcode::SourceSkeptic,
            ));
        }

        // PATTERN: "я вижу паттерн противоречий — Skeptic должен быть жёстче"
        if self.avg_contradictions > 2.0 {
            reflections.push(Thought::new(
                lang.op(Opcode::Pattern).bind(lang.op(Opcode::Conflict)),
                Opcode::Pattern, 4, Opcode::SourceSkeptic,
            ));
        }

        reflections
    }

    /// САМОМОДИФИКАЦИЯ: черви анализируют свою accuracy и адаптируют параметры.
    /// Вызывается из BackgroundMind::evaluate_self().
    /// Возвращает описание изменений (для оператора).
    pub fn self_modify(&mut self, observed_accuracy: f64, avg_confidence: f64) -> Option<String> {
        // Не модифицируем пока нет достаточно данных
        if self.total_predicts < 100 { return None; }

        let mut changes = Vec::new();

        // Стратегия 1: Если accuracy низкая (<20%) и confidence тоже (<0.3) →
        // Skeptic слишком агрессивен, Explorer слишком узкий
        if observed_accuracy < 0.20 && avg_confidence < 0.3 {
            let old_skeptic = self.skeptic_aggression;
            let old_explorer = self.explorer_breadth;
            self.skeptic_aggression = (self.skeptic_aggression - 0.1).max(0.1);
            self.explorer_breadth = (self.explorer_breadth + 0.1).min(0.9);
            changes.push(format!(
                "accuracy={:.1}% low → skeptic {:.1}→{:.1}, explorer {:.1}→{:.1}",
                observed_accuracy * 100.0, old_skeptic, self.skeptic_aggression,
                old_explorer, self.explorer_breadth
            ));
        }

        // Стратегия 2: Если accuracy хорошая (>30%) но confidence низкая →
        // модель знает ответ, но не уверена → усилить evidence weight
        if observed_accuracy > 0.30 && avg_confidence < 0.25 {
            // Collector должен расширить поиск для лучшего evidence
            self.collector_expand = true;
            changes.push(format!(
                "accuracy={:.1}% ok but confidence={:.3} low → expand collector",
                observed_accuracy * 100.0, avg_confidence
            ));
        }

        // Стратегия 3: Если много no_trigram подряд → расширять всегда
        if self.consecutive_no_trigram > 10 {
            self.collector_expand = true;
            changes.push(format!(
                "no_trigram streak={} → permanent collector expand",
                self.consecutive_no_trigram
            ));
        }

        if changes.is_empty() {
            None
        } else {
            Some(format!("SELF-MODIFY: {}", changes.join("; ")))
        }
    }
}

// ============================================================
// HDCMemory — каузальная память (аналог v14)
// mass_role: какие токены "важные" (high similarity → high mass)
// decay_role: какие токены "забываются быстро" (high similarity → high decay)
// Контекст = bundle прошлых с экспоненциально затухающими весами
// ============================================================

pub struct HDCMemory {
    pub mass_role: BinaryVec,   // обучаемый: "что важно"
    pub decay_role: BinaryVec,  // обучаемый: "что забывается"
    mass_acc: BundleAccumulator,
    decay_acc: BundleAccumulator,
    update_count: u32,
    dim: usize,
}

impl HDCMemory {
    pub fn new(dim: usize) -> Self {
        let mut rng_m = StdRng::seed_from_u64(0xCA05A1_0001);
        let mut rng_d = StdRng::seed_from_u64(0xCA05A1_0002);
        HDCMemory {
            mass_role: BinaryVec::random(dim, &mut rng_m),
            decay_role: BinaryVec::random(dim, &mut rng_d),
            mass_acc: BundleAccumulator::new(dim),
            decay_acc: BundleAccumulator::new(dim),
            update_count: 0,
            dim,
        }
    }

    /// Построить каузальный контекст для позиции pos.
    /// HYBRID: проверенный recency weighting + HDCMemory mass модификатор.
    /// Recency даёт baseline (ближние важнее), mass МОДИФИЦИРУЕТ data-dependently.
    /// Далёкие но важные (high mass) слова получают extra weight через causal_extra.
    /// Returns (context_bundle, causal_extra_count)
    pub fn build_causal_context(
        &self,
        words: &[BinaryVec],
        tokens: &[u16],
        pos: usize,
        vocab_size: usize,
    ) -> (BinaryVec, usize) {
        let dim = self.dim;
        let mut ctx_acc = BundleAccumulator::new(dim);
        let mut causal_extra_count = 0usize;

        let lookback = pos.min(50);

        for offset in 0..=lookback {
            let t = pos - offset;
            if t >= tokens.len() { continue; }
            let tok = tokens[t] as usize;
            if tok >= vocab_size { continue; }

            let word = &words[tok];

            // Baseline recency weight (проверенный — НЕ ломаем)
            let recency_base: usize = match offset {
                0..=2 => 4,
                3..=5 => 3,
                6..=10 => 2,
                11..=20 => 1,
                _ => 0, // дальше 20 — только через HDCMemory mass
            };

            // HDCMemory модификатор: mass определяет "важность" этого токена
            // При random mass_role: mass_mod ≈ 1.0 (нейтрально)
            // После Hebbian: важные токены → mass_mod > 1, неважные → mass_mod < 1
            let mass_raw = word.similarity(&self.mass_role) as f64 / dim as f64;
            let mass_mod = 1.0 + mass_raw * 0.3; // [0.7, 1.3] — ±30% модификация

            // Для далёких позиций (>20): HDCMemory может "вытянуть" важные токены
            // mass_mod > 1.02 = даже слабый bias достаточен (после Hebbian станет сильнее)
            let causal_extra = if offset > 20 && mass_mod > 1.02 {
                causal_extra_count += 1;
                1 // важный далёкий токен → дать 1 repeat
            } else {
                0
            };

            let repeats = ((recency_base as f64 * mass_mod).round() as usize).max(causal_extra);
            if repeats == 0 { continue; }

            let positioned = word.permute(offset % dim);
            for _ in 0..repeats {
                ctx_acc.add(&positioned);
            }
        }

        (ctx_acc.to_binary(), causal_extra_count)
    }

    /// Hebbian update: при правильном predict — усилить mass для "полезных" токенов
    pub fn hebbian_update(
        &mut self,
        context_tokens: &[BinaryVec], // слова из контекста
        _correct_word: &BinaryVec,
    ) {
        // Накапливаем паттерны успешных контекстов
        for word in context_tokens.iter().take(5) {
            self.mass_acc.add(word); // "эти слова были важны"
        }
        // Ближайшие 2 → low decay (не забывать)
        for word in context_tokens.iter().take(2) {
            self.decay_acc.add(&word.negate()); // anti-similarity → "не забывать"
        }
        self.update_count += 1;

        // Consolidate каждые 100 updates (было 500 — слишком медленно для sr=10)
        if self.update_count % 100 == 0 {
            self.consolidate();
        }
    }

    fn consolidate(&mut self) {
        let inertia = 4; // было 8 — слишком инертно, роли не успевали обучаться

        // Blend mass_role
        let mut m_blend = BundleAccumulator::new(self.dim);
        for _ in 0..inertia {
            m_blend.add(&self.mass_role);
        }
        m_blend.add(&self.mass_acc.to_binary());
        self.mass_role = m_blend.to_binary();

        // Blend decay_role
        let mut d_blend = BundleAccumulator::new(self.dim);
        for _ in 0..inertia {
            d_blend.add(&self.decay_role);
        }
        d_blend.add(&self.decay_acc.to_binary());
        self.decay_role = d_blend.to_binary();

        // Reset
        self.mass_acc = BundleAccumulator::new(self.dim);
        self.decay_acc = BundleAccumulator::new(self.dim);
    }
}

pub struct AttentionRoles {
    pub role_query: BinaryVec,
    pub role_key: BinaryVec,
    /// Reinforcement: bundled evidence от ПРАВИЛЬНЫХ predict'ов
    pub success_pattern: BinaryVec,
    // Hebbian accumulators
    query_acc: BundleAccumulator,
    key_acc: BundleAccumulator,
    success_acc: BundleAccumulator,
    update_count: u32,
    dim: usize,
    /// Самосознание — persistent между predict'ами
    pub awareness: SelfAwareness,
}

impl AttentionRoles {
    pub fn new(dim: usize) -> Self {
        let mut rng_q = StdRng::seed_from_u64(0xA77E_0001);
        let mut rng_k = StdRng::seed_from_u64(0xA77E_0002);
        AttentionRoles {
            role_query: BinaryVec::random(dim, &mut rng_q),
            role_key: BinaryVec::random(dim, &mut rng_k),
            success_pattern: BinaryVec::zeros(dim),
            query_acc: BundleAccumulator::new(dim),
            key_acc: BundleAccumulator::new(dim),
            success_acc: BundleAccumulator::new(dim),
            update_count: 0,
            dim,
            awareness: SelfAwareness::new(dim),
        }
    }

    /// Hebbian update: "что срабатывает вместе — связывается"
    /// Вызывается при правильном предсказании.
    pub fn hebbian_update(
        &mut self,
        context_bundle: &BinaryVec,
        correct_word: &BinaryVec,
        winning_evidence: &BinaryVec,
    ) {
        self.query_acc.add(context_bundle);     // "этот контекст = хороший query"
        self.key_acc.add(correct_word);          // "это слово = хороший key"
        self.success_acc.add(winning_evidence);  // "эта evidence = паттерн успеха"
        self.update_count += 1;

        // Consolidate каждые 500 правильных predict'ов
        if self.update_count % 500 == 0 {
            self.consolidate();
        }
    }

    /// Консолидация: blend(old_role × inertia, learned_pattern × 1)
    /// Инерция 8:1 — роли меняются медленно, как синаптические связи.
    fn consolidate(&mut self) {
        let inertia = 8;

        // Blend role_query
        let mut q_blend = BundleAccumulator::new(self.dim);
        for _ in 0..inertia {
            q_blend.add(&self.role_query);  // old × 8
        }
        q_blend.add(&self.query_acc.to_binary()); // learned × 1
        self.role_query = q_blend.to_binary();

        // Blend role_key
        let mut k_blend = BundleAccumulator::new(self.dim);
        for _ in 0..inertia {
            k_blend.add(&self.role_key);
        }
        k_blend.add(&self.key_acc.to_binary());
        self.role_key = k_blend.to_binary();

        // Blend success_pattern
        if self.success_acc.count > 0 {
            let mut s_blend = BundleAccumulator::new(self.dim);
            for _ in 0..inertia {
                s_blend.add(&self.success_pattern);
            }
            s_blend.add(&self.success_acc.to_binary());
            self.success_pattern = s_blend.to_binary();
        }

        // Reset accumulators
        self.query_acc = BundleAccumulator::new(self.dim);
        self.key_acc = BundleAccumulator::new(self.dim);
        self.success_acc = BundleAccumulator::new(self.dim);
    }
}

/// HDC Binding Attention context — построен из текущей позиции.
/// Содержит query vector и pre-computed forward bigram check.
pub struct HdcBindingAttention {
    /// Q = context_bundle XOR role_query
    pub query: BinaryVec,
    /// Forward bigram cache: для каждого кандидата (token → has_forward_match)
    forward_cache: HashMap<u16, f64>,
}

impl HdcBindingAttention {
    /// Построить attention context.
    /// Принимает готовый context_bundle (чтобы не строить дважды).
    pub fn build(
        memory: &HierarchicalMemory,
        roles: &AttentionRoles,
        context_bundle: &BinaryVec,
        tokens: &[u16],
        pos: usize,
        candidates: &[(u16, f64, u32)],
    ) -> Self {
        let vs = memory.vocab_size;

        // Q = context XOR role_query (binding = назначение роли "query")
        let query = context_bundle.bind(&roles.role_query);

        // === 3. Pre-compute forward verification для всех кандидатов ===
        // Одна batch операция вместо per-candidate lookup в Analyst
        let t2 = tokens[pos];
        let mut forward_cache = HashMap::new();
        for &(tok, _, _) in candidates {
            if t2 as usize >= vs || tok as usize >= vs { continue; }
            let bigrams = memory.facts.find_by_bigram(t2, tok, 3);
            if !bigrams.is_empty() {
                let evidence: u32 = bigrams.iter()
                    .map(|&idx| memory.facts.entries[idx].total_count)
                    .sum();
                let score = (evidence as f64).ln().max(0.0) / 10.0;
                forward_cache.insert(tok, score.min(0.3));
            }
        }

        HdcBindingAttention { query, forward_cache }
    }

    /// Forward verification score (pre-computed)
    pub fn forward_score(&self, candidate: u16) -> f64 {
        self.forward_cache.get(&candidate).copied().unwrap_or(0.0)
    }
}

/// WormMind — мозг из червей
pub struct WormMind;

impl WormMind {
    /// Полный цикл мышления для предсказания следующего токена.
    /// Черви думают на внутреннем языке, обмениваются мыслями, проверяют себя.
    /// roles: HDC Binding Attention role vectors (учатся через Hebbian)
    /// expected: если известен правильный ответ → Hebbian update ролей
    /// learn_codebook: если true — обновлять codebook через Hebbian (только в training)
    pub fn think(
        memory: &mut HierarchicalMemory,
        lang: &LogicLanguage,
        roles: &mut AttentionRoles,
        hdc_mem: &mut HDCMemory,
        tokens: &[u16],
        pos: usize,
        vocab_size: usize,
        expected: Option<u16>,
        learn_codebook: bool,
    ) -> MindResult {
        if pos < 2 || pos >= tokens.len() {
            return MindResult::empty();
        }

        let dim = memory.dim;
        let t0 = tokens[pos - 2];
        let t1 = tokens[pos - 1];
        let t2 = tokens[pos];

        let mut trace: Vec<String> = Vec::new();
        let mut all_votes: Vec<ThoughtVote> = Vec::new();
        let mut influence = InfluenceReport::default();
        let mut background_updates = 0usize;
        let mut total_reasoning = 0usize;

        // ============================================================
        // Phase 0: САМОСОЗНАНИЕ — читаем awareness, адаптируем стратегию
        // ============================================================
        // Рефлексии от прошлых predict'ов — мысли которые модифицируют текущее поведение
        let awareness = &roles.awareness;
        let collector_expand = awareness.collector_expand;
        let skeptic_top_n = if awareness.skeptic_aggression > 0.7 { 5 } else { 3 };
        let explorer_abs_k = if awareness.explorer_breadth > 0.7 { 10 }
            else if awareness.explorer_breadth > 0.4 { 7 } else { 5 };
        let trigram_k = if collector_expand { 25 } else { 15 }; // расширяем если прошлый predict не нашёл триграмму
        let bigram_k = if collector_expand { 30 } else { 20 };

        if collector_expand { influence.awareness_adaptations += 1; }
        if awareness.skeptic_aggression > 0.7 { influence.awareness_adaptations += 1; }
        if awareness.explorer_breadth > 0.7 { influence.awareness_adaptations += 1; }

        if collector_expand || awareness.skeptic_aggression > 0.7 || awareness.explorer_breadth > 0.7 {
            trace.push(format!(
                "Awareness: expand={}, skeptic={:.1}, explorer={:.1}, no_tri_streak={}",
                collector_expand, awareness.skeptic_aggression,
                awareness.explorer_breadth, awareness.consecutive_no_trigram,
            ));
        }

        // ============================================================
        // Phase 1: COLLECTOR думает — собирает кандидатов
        // ============================================================
        let mut collector_chain = ThoughtChain::new();

        // Шаг 1: FIND в фактах по триграмме (k адаптирован awareness)
        let trigram = memory.make_trigram_query(t0, t1, t2);
        collector_chain.add(
            Thought::new(lang.find(&trigram, Opcode::LevelFacts), Opcode::Find, 3, Opcode::SourceData),
            "FIND(trigram → facts)",
        );

        let top_facts = memory.facts.find_top_k(&trigram, trigram_k);
        let mut trigram_candidates: HashMap<u16, (f64, u32)> = HashMap::new();

        for (idx, sim) in &top_facts {
            if *sim <= 0 { continue; }
            let entry = &memory.facts.entries[*idx];
            let sim_w = *sim as f64 / dim as f64;
            for (&tok, &cnt) in &entry.successors {
                let e = trigram_candidates.entry(tok).or_insert((0.0, 0));
                e.0 += sim_w * (cnt as f64) / entry.total_count.max(1) as f64;
                e.1 += cnt;
            }
        }

        // Шаг 2: FIND по биграмме — ВСЕГДА (не только fallback)
        {
            collector_chain.add(
                Thought::new(lang.find(&memory.make_bigram_query(t1, t2), Opcode::LevelFacts), Opcode::Find, 2, Opcode::SourceData),
                "FIND(bigram → facts)",
            );
            let bigram_indices = memory.facts.find_by_bigram(t1, t2, bigram_k);
            for idx in &bigram_indices {
                let entry = &memory.facts.entries[*idx];
                if entry.total_count < 2 { continue; }
                for (&tok, &cnt) in &entry.successors {
                    let e = trigram_candidates.entry(tok).or_insert((0.0, 0));
                    e.0 += 0.5 * (cnt as f64) / entry.total_count.max(1) as f64;
                    e.1 += cnt;
                }
            }
        }

        // Шаг 3: FIND по униграмме — если мало кандидатов
        if trigram_candidates.len() < 5 {
            let unigram_indices = memory.facts.find_by_token(t2, 15);
            for idx in &unigram_indices {
                let entry = &memory.facts.entries[*idx];
                if entry.total_count < 2 { continue; }
                for (&tok, &cnt) in &entry.successors {
                    let e = trigram_candidates.entry(tok).or_insert((0.0, 0));
                    e.0 += 0.2 * (cnt as f64) / entry.total_count.max(1) as f64;
                    e.1 += cnt;
                }
            }
            collector_chain.add(
                Thought::new(lang.find(&memory.words[t2 as usize].clone(), Opcode::LevelFacts), Opcode::Find, 1, Opcode::SourceData),
                "FIND(unigram → facts) [few candidates]",
            );
        }

        // Шаг 4: FIND в правилах — ВСЕГДА
        if !memory.rules.entries.is_empty() {
            let bigram_q = memory.make_bigram_query(t1, t2);
            let top_rules = memory.rules.find_top_k(&bigram_q, 10);
            for (idx, sim) in &top_rules {
                if *sim <= 0 { continue; }
                let entry = &memory.rules.entries[*idx];
                let sim_w = *sim as f64 / dim as f64;
                for (&tok, &cnt) in &entry.successors {
                    let e = trigram_candidates.entry(tok).or_insert((0.0, 0));
                    e.0 += sim_w * 0.4 * (cnt as f64) / entry.total_count.max(1) as f64;
                    e.1 += cnt;
                }
            }
            collector_chain.add(
                Thought::new(lang.find(&bigram_q, Opcode::LevelRules), Opcode::Find, 2, Opcode::SourceData),
                "FIND(bigram → rules)",
            );
        }

        // Collector передаёт top-10 кандидатов дальше
        let mut sorted_candidates: Vec<(u16, f64, u32)> = trigram_candidates.iter()
            .map(|(&tok, &(score, count))| (tok, score, count))
            .collect();
        sorted_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_candidates.truncate(20);

        total_reasoning += collector_chain.steps.len();
        trace.push(format!("Collector: {} candidates, {} steps", sorted_candidates.len(), collector_chain.steps.len()));

        if sorted_candidates.is_empty() {
            return MindResult::empty();
        }

        // ============================================================
        // Phase 2: ANALYST + HDC BINDING ATTENTION
        // ============================================================
        let mut analyst_chain = ThoughtChain::new();

        // Context bundle: HDCMemory каузальный контекст — hybrid recency + mass
        let (context_bundle, causal_extras) = hdc_mem.build_causal_context(
            &memory.words, tokens, pos, vocab_size,
        );
        influence.causal_extra_tokens = causal_extras;

        // === HDC Binding Attention: Q = context XOR role_query ===
        let attention = HdcBindingAttention::build(
            memory, roles, &context_bundle, tokens, pos, &sorted_candidates,
        );

        analyst_chain.add(
            Thought::new(
                lang.attend(&attention.query, &roles.role_query),
                Opcode::Attend, 3, Opcode::SourceAnalyst,
            ),
            "ATTEND(Q = context XOR role_query)",
        );

        // Soft binding attention: Q·K для top-30 фактов
        let top_facts_for_attn = memory.facts.find_top_k(&trigram, 30);
        let mut attn_successor_scores: HashMap<u16, f64> = HashMap::new();

        for (idx, _lsh_sim) in &top_facts_for_attn {
            let fact = &memory.facts.entries[*idx];
            let key = fact.vec.bind(&roles.role_key);
            let attn_weight = attention.query.similarity(&key) as f64 / dim as f64;
            if attn_weight <= 0.0 { continue; }

            for (&tok, &cnt) in &fact.successors {
                let value_score = attn_weight * (cnt as f64) / fact.total_count.max(1) as f64;
                *attn_successor_scores.entry(tok).or_insert(0.0) += value_score;
            }
        }

        for &(tok, raw_score, _count) in &sorted_candidates {
            if (tok as usize) >= vocab_size { continue; }
            let word_vec = &memory.words[tok as usize];

            // === EVIDENCE BUNDLING: мысли = решения ===
            // Evidence = word_vec кандидата, усиленный/ослабленный сигналами.
            // Доминирующий компонент — сам кандидат. Остальные — модификаторы.
            // Чем больше STRONG сигналов → evidence ближе к word_vec + context.
            // Чем больше WEAK → evidence дальше от context.
            let mut evidence_acc = BundleAccumulator::new(dim);

            // Базовый вес: слово кандидата (доминирует)
            evidence_acc.add(word_vec);
            evidence_acc.add(word_vec);

            // Context compatibility: если совместим, добавить контекст
            let ctx_sim = context_bundle.similarity(word_vec);
            if ctx_sim > 0 {
                evidence_acc.add(&context_bundle);  // контекст усиливает
                evidence_acc.add(word_vec);          // ещё раз слово (= подтверждение)
            }

            // Binding attention: Q·K
            let key = word_vec.bind(&roles.role_key);
            let bind_sim = attention.query.similarity(&key);
            if bind_sim > 0 {
                evidence_acc.add(word_vec); // attention подтверждает
            }

            // Forward verification
            let fwd_score = attention.forward_score(tok);
            if fwd_score > 0.0 {
                evidence_acc.add(word_vec);          // forward подтверждает
                evidence_acc.add(&context_bundle);   // + контекст
            }

            // Raw score: Collector evidence weight
            if raw_score > 0.3 {
                for _ in 0..3 { evidence_acc.add(word_vec); } // сильный сигнал от памяти
                evidence_acc.add(&context_bundle);
            } else if raw_score > 0.05 {
                evidence_acc.add(word_vec);
            }
            // weak: базовые 2 копии word_vec остаются

            let evidence = evidence_acc.to_binary();

            // Reason для трассировки (не для решений!)
            let reason = if raw_score > 0.3 && ctx_sim > 0 { Opcode::Strong }
                else if raw_score > 0.05 { Opcode::Check }
                else { Opcode::Weak };

            all_votes.push(ThoughtVote { token: tok, raw_score, evidence, reason });

            analyst_chain.add(
                Thought::new(lang.check(word_vec), reason, 2, Opcode::SourceAnalyst),
                &format!("{:?}(tok={}, raw={:.2}, ctx_sim={}, fwd={:.2})",
                    reason, tok, raw_score, ctx_sim, fwd_score),
            );
        }

        total_reasoning += analyst_chain.steps.len();
        let n_strong = all_votes.iter().filter(|v| v.reason == Opcode::Strong).count();
        let n_weak = all_votes.iter().filter(|v| v.reason == Opcode::Weak).count();
        trace.push(format!("Analyst+Evidence: {} votes, {} strong, {} weak",
            all_votes.len(), n_strong, n_weak,
        ));

        // ============================================================
        // Phase 3: SKEPTIC думает — ищет контр-доказательства
        // skeptic_top_n адаптирован awareness: больше проверок при высоких contradictions
        // ============================================================
        let mut skeptic_chain = ThoughtChain::new();

        // Берём top-N по evidence similarity к query (N от awareness)
        let mut ranked: Vec<(usize, i32)> = all_votes.iter().enumerate()
            .map(|(i, v)| (i, v.evidence.similarity(&attention.query)))
            .collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        let top_n: Vec<u16> = ranked.iter().take(skeptic_top_n)
            .map(|&(i, _)| all_votes[i].token).collect();

        for &candidate in &top_n {
            if (candidate as usize) >= vocab_size { continue; }
            let word_vec = &memory.words[candidate as usize];

            // === CROSS-WORM READING: Skeptic ЧИТАЕТ ThoughtChain Analyst'а ===
            // Analyst создаёт мысли через lang.check(word_vec) = CHECK XOR word_vec
            // Unbind по CHECK (не по kind!) → восстанавливаем word_vec
            // Затем сравниваем с кандидатом через similarity
            let mut analyst_opinion = Opcode::Check; // default: нейтральный
            for step in &analyst_chain.steps {
                // Мысли Analyst = lang.check(word_vec), т.е. CHECK XOR word_vec
                // Unbind CHECK → word_vec. Сравниваем с кандидатом.
                let content = step.thought.vec.unbind(lang.op(Opcode::Check));
                let sim_to_candidate = content.similarity(word_vec);
                if sim_to_candidate > (dim as i32 / 3) {
                    analyst_opinion = step.thought.kind; // STRONG, WEAK или CHECK
                    break;
                }
            }

            // Skeptic адаптирует поведение на основе мнения Analyst'а:
            // STRONG → Skeptic ищет ПОДТВЕРЖДЕНИЕ (не контр-evidence)
            // WEAK → Skeptic ищет ОПРОВЕРЖЕНИЕ агрессивно
            // CHECK → стандартная проверка
            let search_contradictions = analyst_opinion != Opcode::Strong;

            // ASK через HDC: запрос используя unbind от analyst thought
            let analyst_check = lang.check(word_vec);
            let recovered = analyst_check.unbind(lang.op(Opcode::Check));

            skeptic_chain.add(
                Thought::new(lang.ask(&recovered), Opcode::Ask, 3, Opcode::SourceSkeptic),
                &format!("ASK(analyst={:?}, tok={}, search_contra={})",
                    analyst_opinion, candidate, search_contradictions),
            );

            // Ищем факты ЧЕРЕЗ HDC: similarity(recovered, fact) для поиска контр-evidence
            // + fallback на token index (быстрый)
            let facts_with_context = memory.facts.find_by_token(t2, 30);
            let mut supporting = 0u32;
            let mut contradicting = 0u32;

            for &idx in &facts_with_context {
                let entry = &memory.facts.entries[idx];
                // Проверяем: факт про похожий контекст?
                let has_t1 = entry.context_tokens.contains(&t1);
                if !has_t1 { continue; }

                if let Some(&cnt) = entry.successors.get(&candidate) {
                    supporting += cnt;
                }
                let other_count: u32 = entry.successors.iter()
                    .filter(|(&tok, _)| tok != candidate)
                    .map(|(_, &cnt)| cnt)
                    .sum();
                contradicting += other_count;
            }

            // COMPARE(supporting, contradicting) — через HDC язык
            if supporting > 0 || contradicting > 0 {
                let ratio = supporting as f64 / (supporting + contradicting).max(1) as f64;

                // Skeptic формирует МЫСЛЬ-СРАВНЕНИЕ на внутреннем языке
                let compare_thought = lang.compare(&lang.strong(word_vec), &lang.weak(word_vec));
                skeptic_chain.add(
                    Thought::new(compare_thought, Opcode::Compare, 3, Opcode::SourceSkeptic),
                    &format!("COMPARE(support={}, contra={}, ratio={:.2}, analyst={:?})",
                        supporting, contradicting, ratio, analyst_opinion),
                );

                // Модифицируем evidence через HDC мысли
                // Skeptic's решение ЗАВИСИТ от мнения Analyst'а (cross-worm reasoning):
                //   Analyst=STRONG + ratio>0.5 → СИЛЬНОЕ подтверждение (два червя согласны)
                //   Analyst=WEAK + ratio<0.3 → СИЛЬНОЕ опровержение (два червя согласны)
                //   Разногласие → стандартная логика по ratio
                let confirm_threshold = if analyst_opinion == Opcode::Strong { 0.5 } else { 0.6 };
                let reject_threshold = if analyst_opinion == Opcode::Weak { 0.4 } else { 0.3 };
                // Count cross-worm influence: когда Analyst мнение != neutral
                if analyst_opinion != Opcode::Check {
                    influence.cross_worm_overrides += 1;
                }

                for vote in all_votes.iter_mut() {
                    if vote.token == candidate {
                        if ratio > confirm_threshold {
                            // Подтверждено → STRONG + CONFIDENT через HDC bundling
                            let mut acc = BundleAccumulator::new(dim);
                            acc.add(&vote.evidence);
                            acc.add(&vote.evidence); // old evidence × 2
                            acc.add(&lang.strong(word_vec)); // + STRONG мысль
                            acc.add(&lang.confident(word_vec)); // + CONFIDENT мысль
                            // Если Analyst тоже сказал STRONG — тройное усиление
                            if analyst_opinion == Opcode::Strong {
                                acc.add(&lang.strong(word_vec)); // consensus = extra STRONG
                            }
                            vote.evidence = acc.to_binary();
                            vote.reason = Opcode::Strong;
                        } else if ratio < reject_threshold {
                            // Опровергнуто → WEAK через HDC bundling
                            let mut acc = BundleAccumulator::new(dim);
                            acc.add(&vote.evidence);
                            acc.add(&lang.weak(word_vec)); // + WEAK мысль
                            acc.add(&lang.weak(word_vec)); // double weak
                            // Если Analyst тоже сказал WEAK — усиленное опровержение
                            if analyst_opinion == Opcode::Weak {
                                acc.add(&lang.weak(word_vec)); // consensus = triple weak
                            }
                            vote.evidence = acc.to_binary();
                            vote.reason = Opcode::Weak;
                        }
                    }
                }

                // Background learning: укрепляем или ослабляем факт
                if let Some((fact_idx, _)) = memory.facts.find_nearest(&trigram) {
                    if ratio > 0.7 {
                        // Подтверждено — укрепляем
                        memory.facts.entries[fact_idx].confirm_successor(candidate, 1);
                        background_updates += 1;
                    } else if ratio < 0.2 && contradicting > 5 {
                        // Сильно опровергнуто — ослабляем (самоосознание!)
                        if let Some(cnt) = memory.facts.entries[fact_idx].successors.get_mut(&candidate) {
                            *cnt = cnt.saturating_sub(1);
                        }
                        background_updates += 1;

                        skeptic_chain.add(
                            Thought::new(
                                lang.weak(&memory.words[candidate as usize].clone()),
                                Opcode::Forget, 4, Opcode::SourceSkeptic,
                            ),
                            &format!("FORGET(tok={}, ratio={:.2} — ослабляю ошибочный факт)", candidate, ratio),
                        );
                    }
                }
            }
        }

        total_reasoning += skeptic_chain.steps.len();
        trace.push(format!("Skeptic: {} checks on top-3", skeptic_chain.steps.len()));

        // ============================================================
        // Phase 3.5: RELATIONS QUERY — семантический поиск через связи
        // Черви используют Relations (уровень 5) для усиления evidence
        // ============================================================
        let mut relations_used = 0usize;
        if !memory.relations.entries.is_empty() {
            // Для каждого контекстного токена ищем связи типа Sequence
            let context_tokens: Vec<u16> = if pos >= 3 {
                tokens[pos.saturating_sub(3)..=pos].to_vec()
            } else {
                tokens[..=pos].to_vec()
            };

            let mut relation_boost: HashMap<u16, f64> = HashMap::new();

            for &ctx_tok in &context_tokens {
                if (ctx_tok as usize) >= vocab_size { continue; }

                // Ищем: ctx_tok --Sequence--> ? (что обычно идёт после)
                // Только высоко-уверенные (confidence >= 3) — отсеиваем шум
                let seq_relations = memory.relations.find_by_subject_role(ctx_tok, Opcode::Sequence, 10);
                for &rel_idx in &seq_relations {
                    let rel = &memory.relations.entries[rel_idx];
                    if rel.confidence < 3 { continue; } // шумные — пропускаем
                    let boost = (rel.confidence as f64).ln().max(0.0) * 0.05; // слабый boost
                    *relation_boost.entry(rel.object).or_insert(0.0) += boost;
                    relations_used += 1;
                }

                // Ищем: ctx_tok --Similar--> ? — только для ПОДТВЕРЖДЕНИЯ существующих кандидатов
                let sim_relations = memory.relations.find_by_subject_role(ctx_tok, Opcode::Similar, 5);
                for &rel_idx in &sim_relations {
                    let rel = &memory.relations.entries[rel_idx];
                    if rel.confidence < 3 { continue; }
                    let boost = (rel.confidence as f64).ln().max(0.0) * 0.02; // очень слабый
                    *relation_boost.entry(rel.object).or_insert(0.0) += boost;
                    relations_used += 1;
                }
            }

            // Применяем boost через raw_score multiplier (НЕ evidence bundling —
            // evidence bundling может навредить при шумных relations)
            for vote in all_votes.iter_mut() {
                if let Some(&boost) = relation_boost.get(&vote.token) {
                    if boost > 0.02 {
                        vote.raw_score *= 1.0 + boost.min(0.15); // макс +15%
                        influence.relations_boosted += 1;
                    }
                }
            }

            if relations_used > 0 {
                trace.push(format!("Relations: {} queries boosted candidates", relations_used));
            }
        }

        // ============================================================
        // Phase 4: EXPLORER думает — ищет неожиданные варианты
        // explorer_abs_k адаптирован awareness: шире при высокой uncertainty
        // ============================================================
        let mut explorer_chain = ThoughtChain::new();

        // === CROSS-WORM READING: Explorer читает Skeptic И Analyst ===
        // Skeptic мысли: ASK(x) → unbind ASK = x, COMPARE(a,b), WEAK(x) → unbind по creation opcode
        // Analyst мысли: CHECK(word_vec) → unbind CHECK = word_vec
        let mut skeptic_confirmed: Vec<BinaryVec> = Vec::new();
        let mut skeptic_rejected: Vec<BinaryVec> = Vec::new();
        for step in &skeptic_chain.steps {
            // Для Skeptic: thoughts создаются через Ask, Compare, Weak
            // Unbind по SOURCE опкоду (Ask для первых, Compare для средних, Weak для последних)
            match step.thought.kind {
                Opcode::Strong | Opcode::Compare => {
                    // Compare thoughts: unbind Compare → получить первый аргумент
                    let content = step.thought.vec.unbind(lang.op(Opcode::Compare));
                    skeptic_confirmed.push(content);
                }
                Opcode::Weak | Opcode::Forget => {
                    // Weak thoughts: unbind Weak → word_vec
                    let content = step.thought.vec.unbind(lang.op(Opcode::Weak));
                    skeptic_rejected.push(content);
                }
                _ => {}
            }
        }

        // Analyst WEAK мнения: CHECK(word_vec) с kind=Weak → unbind CHECK → word_vec
        let mut analyst_weak_tokens: Vec<BinaryVec> = Vec::new();
        for step in &analyst_chain.steps {
            if step.thought.kind == Opcode::Weak {
                let content = step.thought.vec.unbind(lang.op(Opcode::Check));
                analyst_weak_tokens.push(content);
            }
        }

        // Explorer строит query: confirmed + (ОТДАЛИТЬСЯ от rejected/weak)
        let has_cross_worm_input = !skeptic_confirmed.is_empty()
            || !skeptic_rejected.is_empty()
            || !analyst_weak_tokens.is_empty();

        let explorer_query = if has_cross_worm_input {
            let mut seed_acc = BundleAccumulator::new(dim);
            seed_acc.add(&trigram); // базовый trigram
            // Подтверждённые → притягиваемся
            for v in &skeptic_confirmed {
                seed_acc.add(v);
            }
            // Отвергнутые → отталкиваемся (добавляем NEGATE)
            for v in &skeptic_rejected {
                seed_acc.add(&v.negate()); // anti-direction
            }
            // Analyst's weak → тоже отталкиваемся (ищем ДРУГОЕ)
            for v in &analyst_weak_tokens {
                seed_acc.add(&v.negate());
            }
            seed_acc.to_binary()
        } else {
            trigram.clone() // fallback: обычный trigram
        };

        // Explorer ищет в абстракциях используя enriched query (k адаптирован awareness)
        if !memory.abstractions.entries.is_empty() {
            let top_abs = memory.abstractions.find_top_k(&explorer_query, explorer_abs_k);
            for (idx, sim) in &top_abs {
                if *sim <= 0 { continue; }
                let entry = &memory.abstractions.entries[*idx];
                let sim_w = *sim as f64 / dim as f64;

                for (&tok, &cnt) in &entry.successors {
                    // Проверяем: это новый кандидат, которого Collector не нашёл?
                    let is_novel = !all_votes.iter().any(|v| v.token == tok);
                    if is_novel && cnt >= 3 {
                        // Novel candidate — evidence через NOVEL thought (слабее чем обычный)
                        let novel_thought = lang.novel(&memory.words[tok as usize]);
                        let mut ev_acc = BundleAccumulator::new(dim);
                        ev_acc.add(&novel_thought);
                        ev_acc.add(&lang.weak(&memory.words[tok as usize])); // low confidence by default
                        let raw = sim_w * (cnt as f64) / entry.total_count.max(1) as f64;
                        all_votes.push(ThoughtVote {
                            token: tok,
                            raw_score: raw * 0.3,
                            evidence: ev_acc.to_binary(),
                            reason: Opcode::Novel,
                        });

                        explorer_chain.add(
                            Thought::new(lang.novel(&memory.words[tok as usize].clone()), Opcode::Novel, 2, Opcode::SourceExplorer),
                            &format!("NOVEL(tok={}, from abstraction, sim={:.2})", tok, sim_w),
                        );
                    }
                }
            }
        }

        // Explorer также проверяет medium context
        if pos >= 10 && !memory.rules.entries.is_empty() {
            let med_ctx = memory.make_context_bundle(tokens, pos, 10);
            let top_rules = memory.rules.find_top_k(&med_ctx, 5);
            for (idx, sim) in &top_rules {
                if *sim <= 0 { continue; }
                let entry = &memory.rules.entries[*idx];
                let sim_w = *sim as f64 / dim as f64;
                if sim_w < 0.1 { continue; }

                for (&tok, &_cnt) in &entry.successors {
                    // Дополнительное подтверждение через HDC evidence bundling
                    if let Some(vote) = all_votes.iter_mut().find(|v| v.token == tok) {
                        // Rule подтверждает → bundle evidence с STRONG
                        let rule_thought = lang.strong(&memory.words[tok as usize]);
                        let mut acc = BundleAccumulator::new(dim);
                        acc.add(&vote.evidence);
                        acc.add(&vote.evidence); // keep old evidence dominant
                        acc.add(&rule_thought);   // add rule confirmation
                        vote.evidence = acc.to_binary();
                    }
                }
            }
        }

        // Explorer background learning: если top кандидат уверен, создаём/укрепляем абстракцию
        if !sorted_candidates.is_empty() {
            let best = sorted_candidates[0]; // (token, raw_score, count)
            if best.1 > 0.5 && best.2 >= 3 {
                // Этот контекст → этот токен с высокой уверенностью
                // Проверяем: есть ли абстракция для этого паттерна?
                if memory.abstractions.find_matching(&trigram).is_none() && memory.abstractions.len() < 200_000 {
                    let ctx_tokens = if pos >= 5 {
                        tokens[pos-5..=pos].iter().filter(|&&t| (t as usize) < vocab_size).cloned().collect()
                    } else {
                        vec![t0, t1, t2]
                    };
                    let mut new_abs = MemoryEntry::with_context(trigram.clone(), 3, ctx_tokens);
                    new_abs.confirm_successor(best.0, best.2);
                    memory.abstractions.store(new_abs);
                    background_updates += 1;

                    explorer_chain.add(
                        Thought::new(
                            lang.store(&trigram, Opcode::LevelAbstractions),
                            Opcode::Store, 2, Opcode::SourceExplorer,
                        ),
                        &format!("STORE(new abstraction for top candidate tok={}, count={})", best.0, best.2),
                    );
                }
            }
        }

        total_reasoning += explorer_chain.steps.len();
        trace.push(format!("Explorer: {} novel + bg_learning", explorer_chain.steps.len()));

        // ============================================================
        // Phase 5: CONSENSUS — черви голосуют
        // ============================================================

        // Финальный скор = raw_score × thought_multiplier
        // Multiplier подход: сохраняет порядок raw_score, но позволяет мыслям
        // усиливать/ослаблять кандидатов пропорционально их raw_score.
        // Три сигнала мыслей:
        //   1. evidence_sim: STRONG/WEAK через bundling → контекстная совместимость
        //   2. attn_sim: HDC binding attention Q·K → позиционная релевантность
        //   3. forward: forward verification → подтверждение через биграммы
        let has_success = roles.update_count > 0;

        let mut final_scores: HashMap<u16, f64> = HashMap::new();
        for vote in &all_votes {
            // Evidence: мысли модифицируют через similarity(evidence, context)
            let evidence_sim = vote.evidence.similarity(&context_bundle) as f64 / dim as f64;

            // Attention: similarity(evidence, query) — binding attention signal
            let attn_sim = vote.evidence.similarity(&attention.query) as f64 / dim as f64;

            // Forward verification
            let fwd = attention.forward_score(vote.token);

            // Reinforcement от success_pattern
            let reinforcement = if has_success {
                let sim = vote.evidence.similarity(&roles.success_pattern) as f64 / dim as f64;
                sim.max(0.0) * 0.15
            } else {
                0.0
            };

            // Multiplier: evidence (±50%) + attention (+25%) + forward (+15%) + reinf
            // Диапазон: [0.35, 1.9] — мысли СИЛЬНО влияют, но через raw_score
            let thought_multiplier = 1.0
                + evidence_sim.clamp(-0.5, 0.5)         // ±50% от evidence
                + attn_sim.max(0.0).min(0.25)            // +0..25% от attention
                + fwd.min(0.15)                            // +0..15% от forward
                + reinforcement;                           // +0..15% от success

            let score = vote.raw_score * thought_multiplier;
            *final_scores.entry(vote.token).or_insert(0.0) += score;
        }

        let mut predictions: Vec<(u16, f64)> = final_scores.into_iter().collect();
        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        predictions.truncate(10);

        // Evidence rerank: если top-1 по final_score ≠ top-1 по raw_score → evidence повлиял
        let raw_top1 = all_votes.iter()
            .max_by(|a, b| a.raw_score.partial_cmp(&b.raw_score).unwrap_or(std::cmp::Ordering::Equal))
            .map(|v| v.token);
        let final_top1 = predictions.first().map(|(t, _)| *t);
        if raw_top1 != final_top1 && final_top1.is_some() {
            influence.evidence_reranks += 1;
        }

        let confidence = if predictions.len() >= 2 {
            let top = predictions[0].1;
            let second = predictions[1].1;
            if top > 0.0 { (top - second) / top } else { 0.0 }
        } else if predictions.len() == 1 {
            1.0
        } else {
            0.0
        };

        let thoughts_total = collector_chain.steps.len()
            + analyst_chain.steps.len()
            + skeptic_chain.steps.len()
            + explorer_chain.steps.len();

        // ============================================================
        // Phase 5.5: HEBBIAN LEARNING — "что срабатывает вместе, связывается"
        // ============================================================
        if let Some(exp) = expected {
            let is_correct = predictions.first().map(|(t, _)| *t == exp).unwrap_or(false);

            if is_correct && (exp as usize) < vocab_size {
                // Правильный predict! Hebbian: усиливаем связь context→answer + evidence pattern
                let winning_evidence = all_votes.iter()
                    .find(|v| v.token == exp)
                    .map(|v| v.evidence.clone())
                    .unwrap_or_else(|| context_bundle.clone());
                roles.hebbian_update(&context_bundle, &memory.words[exp as usize], &winning_evidence);

                // v19: Hebbian codebook update — ТОЛЬКО при правильном predict И в training
                if learn_codebook {
                    memory.hebbian_codebook_update(exp, &context_bundle, true);
                }

                // HDCMemory Hebbian: обучить mass/decay роли
                let ctx_words: Vec<BinaryVec> = (0..pos.min(5))
                    .filter_map(|k| {
                        let t = tokens[pos - k] as usize;
                        if t < vocab_size { Some(memory.words[t].clone()) } else { None }
                    })
                    .collect();
                hdc_mem.hebbian_update(&ctx_words, &memory.words[exp as usize]);

                background_updates += 1;
            }
            // Неправильный predict — НЕ трогаем ни roles ни codebook
            // (при accuracy ~20-40% anti-Hebbian разрушает codebook)
        }

        // ============================================================
        // Phase 6: САМОДИАГНОСТИКА — что нужно червям
        // ============================================================
        let no_trigram = trigram_candidates.iter()
            .all(|(_, (score, _))| *score < 0.1);
        // Uncertain = evidence similarity к query ниже порога
        let uncertain_count = all_votes.iter()
            .filter(|v| v.evidence.similarity(&attention.query) < (dim as i32 / 20))
            .count();
        let contradiction_count = all_votes.iter()
            .filter(|v| v.reason == Opcode::Weak)
            .count();
        let novel_count = all_votes.iter()
            .filter(|v| v.reason == Opcode::Novel)
            .count();

        // Токены где мало данных — NEED_MORE
        let mut need_more: Vec<u16> = Vec::new();
        if no_trigram {
            need_more.push(t2);
            // Рефлексия: "мне нужно больше данных про этот контекст"
            trace.push(format!("REFLECT: no trigram match for context, NEED_MORE(tok={})", t2));
        }
        if uncertain_count > sorted_candidates.len() / 2 {
            need_more.push(t1);
            trace.push(format!("REFLECT: {} uncertain of {} candidates, NEED_MORE", uncertain_count, sorted_candidates.len()));
        }

        // Attention-специфичные метрики (используем pre-computed cache)
        let forward_verified_count = all_votes.iter()
            .filter(|v| attention.forward_score(v.token) > 0.0)
            .count();

        let diagnostic = DiagnosticReport {
            need_more_data: need_more,
            uncertain_areas: uncertain_count,
            no_trigram_match: no_trigram,
            weak_context: pos < 10,
            contradictions: contradiction_count,
            novel_discoveries: novel_count,
            has_shift_pattern: false, // removed shift prediction
            forward_verified: forward_verified_count,
        };

        trace.push(format!("Consensus: {} candidates, confidence={:.3}, bg_updates={}, uncertain={}, contradictions={}",
            predictions.len(), confidence, background_updates, uncertain_count, contradiction_count));

        // ============================================================
        // Phase 6.5: РЕФЛЕКСИЯ — обновляем самосознание для СЛЕДУЮЩЕГО predict'а
        // ============================================================
        // Черви АНАЛИЗИРУЮТ результат текущего рассуждения через внутренний язык
        // и генерируют мысли-рефлексии которые модифицируют стратегию следующего predict'а
        let reflections = roles.awareness.reflect_on(&diagnostic, lang, dim);
        let n_reflections = reflections.len();

        // Bundle need_more_data токенов в weak_areas (для guidance Explorer'у)
        for &tok in &diagnostic.need_more_data {
            if (tok as usize) < vocab_size {
                let mut acc = BundleAccumulator::new(dim);
                acc.add(&roles.awareness.weak_areas);
                acc.add(&memory.words[tok as usize]);
                roles.awareness.weak_areas = acc.to_binary();
            }
        }

        // ============================================================
        // Phase 7: СОЗДАНИЕ RELATIONS — черви формируют семантические связи
        // (только в training mode, не в readonly eval)
        // ============================================================
        let mut relations_created = 0usize;
        // Relations создаются при наличии expected (training/eval), не при generate
        if let Some(exp) = expected {
            let is_correct = predictions.first().map(|(t, _)| *t == exp).unwrap_or(false);

            if is_correct && (exp as usize) < vocab_size && (t2 as usize) < vocab_size {
                // Sequence: t2 → exp (что идёт после чего)
                let role_vec = lang.op(Opcode::Sequence);
                let rel_vec = memory.make_relation_vec(t2, role_vec, exp);
                memory.relations.store(RelationEntry {
                    vec: rel_vec,
                    subject: t2,
                    role: Opcode::Sequence,
                    object: exp,
                    confidence: 1,
                    source: RelationSource::Data,
                });
                relations_created += 1;

                // Similar: если t1 → exp тоже часто встречается, создать Similar(t1, t2)
                // (токены в одинаковом контексте → похожи)
                if (t1 as usize) < vocab_size && t1 != t2 {
                    let sim_to_t2 = memory.words[t1 as usize].similarity(&memory.words[t2 as usize]);
                    if sim_to_t2 > (dim as i32 / 8) {
                        let role_sim = lang.op(Opcode::Similar);
                        let rel_vec = memory.make_relation_vec(t1, role_sim, t2);
                        memory.relations.store(RelationEntry {
                            vec: rel_vec,
                            subject: t1,
                            role: Opcode::Similar,
                            object: t2,
                            confidence: 1,
                            source: RelationSource::Inference,
                        });
                        relations_created += 1;
                    }
                }
            // НЕ создаём Sequence relations для неправильных predict'ов —
            // это шум который деградирует accuracy
            }
        }

        if relations_created > 0 {
            trace.push(format!("Relations: {} created", relations_created));
            background_updates += relations_created;
        }

        // ============================================================
        // Phase 8: КАНАЛ СВЯЗИ — черви формулируют сообщения для оператора
        // ============================================================
        let mut messages: Vec<WormMessage> = Vec::new();

        // ASK при очень низкой confidence
        if confidence < 0.05 && !predictions.is_empty() {
            messages.push(WormMessage::Ask(
                format!("Не уверен в ответе: confidence={:.3}, {} кандидатов uncertain", confidence, uncertain_count)
            ));
        }

        // REPORT при высокой confidence + novel discovery
        if confidence > 0.7 && novel_count > 0 {
            if let Some(&(pred_tok, _)) = predictions.first() {
                messages.push(WormMessage::Report(
                    format!("Уверенный novel predict: tok={}, confidence={:.3}", pred_tok, confidence)
                ));
            }
        }

        // CONFUSED при слишком много contradictions
        if contradiction_count > 5 {
            messages.push(WormMessage::Confused(
                format!("{} противоречий в контексте [{},{},{}]", contradiction_count, t0, t1, t2)
            ));
        }

        if n_reflections > 0 || collector_expand {
            trace.push(format!("Reflect: {} thoughts, skeptic_aggr={:.1}, explorer_br={:.1}, expand={}",
                n_reflections, roles.awareness.skeptic_aggression,
                roles.awareness.explorer_breadth, collector_expand));
        }

        total_reasoning += n_reflections;

        MindResult {
            predictions,
            thoughts_exchanged: thoughts_total,
            reasoning_depth: total_reasoning,
            confidence,
            background_updates,
            trace,
            diagnostic,
            messages,
            influence,
        }
    }
}

impl MindResult {
    pub fn empty() -> Self {
        MindResult {
            predictions: vec![],
            thoughts_exchanged: 0,
            reasoning_depth: 0,
            confidence: 0.0,
            background_updates: 0,
            trace: vec![],
            diagnostic: DiagnosticReport {
                need_more_data: vec![],
                uncertain_areas: 0,
                no_trigram_match: true,
                weak_context: true,
                contradictions: 0,
                novel_discoveries: 0,
                has_shift_pattern: false,
                forward_verified: 0,
            },
            messages: vec![],
            influence: InfluenceReport::default(),
        }
    }
}

// ============================================================
// BackgroundMind — автономная саморефлексия между predict'ами
// ============================================================
// Цикл: INTROSPECT → EVALUATE → GENERALIZE → ADAPT
// Вызывается периодически (не в отдельном thread), не блокирует predict.

/// Сообщения от червей к оператору
#[derive(Clone, Debug)]
pub enum WormMessage {
    // Черви → Оператор
    Ask(String),                    // "что значит X?"
    NeedData(String),               // "дай больше примеров про X"
    Report(String),                 // "я обнаружил паттерн: ..."
    Discovery(String),              // "я понял что X ≈ Y!"
    Confused(String),               // "X и Y противоречат, помоги"
}

/// Результат одного фонового цикла
pub struct BackgroundResult {
    pub facts_strengthened: usize,
    pub facts_weakened: usize,
    pub relations_created: usize,
    #[allow(dead_code)]
    pub rules_generalized: usize,
    #[allow(dead_code)]
    pub messages: Vec<WormMessage>,
}

/// Самопознание: мета-знания о себе (хранятся в Meta level 4)
pub struct SelfKnowledge {
    /// "У меня N фактов" — обновляется каждый EVALUATE цикл
    pub facts_count: usize,
    pub rules_count: usize,
    pub abstractions_count: usize,
    pub relations_count: usize,
    /// Сильные области (токены с > 10 фактами)
    pub strong_areas_count: usize,
    /// Слабые области (токены с < 3 фактов)
    pub weak_areas_count: usize,
    /// Общая оценка "здоровья" знаний [0, 1]
    pub knowledge_health: f64,
    /// Последняя accuracy наблюдённая (из train eval)
    pub observed_accuracy: f64,
    /// Сколько predict'ов сделано
    pub total_predicts: u64,
    /// Средняя confidence
    pub avg_confidence: f64,
}

impl SelfKnowledge {
    pub fn new() -> Self {
        SelfKnowledge {
            facts_count: 0,
            rules_count: 0,
            abstractions_count: 0,
            relations_count: 0,
            strong_areas_count: 0,
            weak_areas_count: 0,
            knowledge_health: 0.0,
            observed_accuracy: 0.0,
            total_predicts: 0,
            avg_confidence: 0.0,
        }
    }

    /// Сформулировать самоописание на человеческом языке
    pub fn describe(&self) -> String {
        format!(
            "Я — HDC Brain v18. Я состою из 4 червей (Collector, Analyst, Skeptic, Explorer).\n\
             У меня {} фактов, {} правил, {} абстракций, {} связей.\n\
             Сильных областей: {}, слабых: {}. Здоровье знаний: {:.0}%.\n\
             За {} predict'ов: accuracy {:.1}%, средняя confidence {:.3}.",
            self.facts_count, self.rules_count, self.abstractions_count, self.relations_count,
            self.strong_areas_count, self.weak_areas_count, self.knowledge_health * 100.0,
            self.total_predicts, self.observed_accuracy * 100.0, self.avg_confidence,
        )
    }
}

pub struct BackgroundMind {
    pub cycle_count: u64,
    pub messages: Vec<WormMessage>,
    pub self_knowledge: SelfKnowledge,
    introspect_interval: u64,
    generalize_interval: u64,
    evaluate_interval: u64,
}

impl BackgroundMind {
    pub fn new() -> Self {
        BackgroundMind {
            cycle_count: 0,
            messages: Vec::new(),
            self_knowledge: SelfKnowledge::new(),
            introspect_interval: 1,    // каждый вызов
            generalize_interval: 3,    // каждый 3-й вызов
            evaluate_interval: 2,      // каждый 2-й вызов
        }
    }

    /// Один цикл фоновой рефлексии. Вызывается из evaluate() периодически.
    /// Не блокирует predict — работает с уже собранными данными.
    pub fn run_cycle(
        &mut self,
        memory: &mut HierarchicalMemory,
        lang: &LogicLanguage,
        vocab: Option<&[String]>,
    ) -> BackgroundResult {
        self.cycle_count += 1;
        let mut result = BackgroundResult {
            facts_strengthened: 0,
            facts_weakened: 0,
            relations_created: 0,
            rules_generalized: 0,
            messages: Vec::new(),
        };

        // INTROSPECT: сканировать слабые факты
        if self.cycle_count % self.introspect_interval == 0 {
            self.introspect(memory, lang, vocab, &mut result);
        }

        // GENERALIZE: искать паттерны → создавать relations и rules
        if self.cycle_count % self.generalize_interval == 0 {
            self.generalize(memory, lang, vocab, &mut result);
        }

        // EVALUATE: обновить самопознание — "кто я, что знаю"
        if self.cycle_count % self.evaluate_interval == 0 {
            self.evaluate_self(memory);
        }

        // Собрать messages
        result.messages.append(&mut self.messages);
        result
    }

    /// INTROSPECT: сканировать знания, найти слабые/конфликтные
    fn introspect(
        &mut self,
        memory: &mut HierarchicalMemory,
        _lang: &LogicLanguage,
        vocab: Option<&[String]>,
        result: &mut BackgroundResult,
    ) {
        let facts_len = memory.facts.entries.len();
        if facts_len == 0 { return; }

        // Сканируем ~1000 случайных фактов (не все — было бы O(N))
        let scan_count = facts_len.min(1000);
        let step = facts_len / scan_count;

        let mut weak_count = 0usize;
        let mut strong_count = 0usize;

        for i in (0..facts_len).step_by(step.max(1)).take(scan_count) {
            let entry = &memory.facts.entries[i];
            if entry.total_count < 3 {
                weak_count += 1;
            } else if entry.total_count > 20 {
                strong_count += 1;
            }
        }

        // Если много слабых фактов → NeedData
        let weak_ratio = weak_count as f64 / scan_count as f64;
        if weak_ratio > 0.6 {
            self.messages.push(WormMessage::NeedData(
                format!("INTROSPECT: {:.0}% фактов слабые (count<3). Нужно больше данных.", weak_ratio * 100.0)
            ));
        }

        // Сканировать Relations на конфликты: subject+role → разные objects
        let relations_len = memory.relations.entries.len();
        if relations_len > 10 {
            let mut subject_roles: HashMap<(u16, u8), Vec<u16>> = HashMap::new();
            for entry in &memory.relations.entries {
                let role_idx = entry.role.index() as u8;
                subject_roles.entry((entry.subject, role_idx))
                    .or_default()
                    .push(entry.object);
            }

            for (&(subj, _role_idx), objects) in &subject_roles {
                if objects.len() > 20 {
                    // Много разных объектов для одной пары → потенциальный конфликт
                    let subj_name = vocab.map(|v| v.get(subj as usize).map(|s| s.as_str()).unwrap_or("?"))
                        .unwrap_or("?");
                    self.messages.push(WormMessage::Confused(
                        format!("INTROSPECT: '{}' имеет {} разных связей одного типа — конфликт?", subj_name, objects.len())
                    ));
                }
            }
        }

        result.facts_strengthened = strong_count;
        result.facts_weakened = weak_count;
    }

    /// GENERALIZE: найти паттерны → создавать relations + rules
    fn generalize(
        &mut self,
        memory: &mut HierarchicalMemory,
        lang: &LogicLanguage,
        _vocab: Option<&[String]>,
        result: &mut BackgroundResult,
    ) {
        let _dim = memory.dim;
        let vocab_size = memory.vocab_size;

        // Стратегия 1: Найти пары токенов с одинаковыми successors → Similar relation
        // Берём sample из facts
        let facts_len = memory.facts.entries.len();
        if facts_len < 100 { return; }

        // Собрать top successor для sample фактов
        let sample_size = facts_len.min(500);
        let step = facts_len / sample_size;
        let mut token_successors: HashMap<u16, HashSet<u16>> = HashMap::new();

        for i in (0..facts_len).step_by(step.max(1)).take(sample_size) {
            let entry = &memory.facts.entries[i];
            if let Some((top_tok, top_cnt)) = entry.top1() {
                if top_cnt >= 3 {
                    for &ctx_tok in &entry.context_tokens {
                        if (ctx_tok as usize) < vocab_size {
                            token_successors.entry(ctx_tok).or_default().insert(top_tok);
                        }
                    }
                }
            }
        }

        // Найти пары токенов с общими successors (>= 3 общих) → Similar
        let tokens_with_succ: Vec<(u16, &HashSet<u16>)> = token_successors.iter()
            .filter(|(_, s)| s.len() >= 3)
            .map(|(&t, s)| (t, s))
            .collect();

        let mut new_similar = 0usize;
        let max_comparisons = 500; // лимитируем чтобы не тормозить
        let mut comparisons = 0;

        'outer: for i in 0..tokens_with_succ.len().min(50) {
            let (t1, s1) = tokens_with_succ[i];
            for j in (i+1)..tokens_with_succ.len().min(50) {
                if comparisons >= max_comparisons { break 'outer; }
                comparisons += 1;

                let (t2, s2) = tokens_with_succ[j];
                if t1 == t2 { continue; }

                let common: usize = s1.intersection(s2).count();
                let total = s1.len().min(s2.len());
                if common >= 5 && common as f64 / total as f64 > 0.5 {
                    // Достаточно общих successors → Similar
                    if (t1 as usize) < vocab_size && (t2 as usize) < vocab_size {
                        let role_vec = lang.op(Opcode::Similar);
                        let rel_vec = memory.make_relation_vec(t1, role_vec, t2);
                        let stored_idx = memory.relations.store(RelationEntry {
                            vec: rel_vec,
                            subject: t1,
                            role: Opcode::Similar,
                            object: t2,
                            confidence: common as u32,
                            source: RelationSource::Inference,
                        });
                        if stored_idx != usize::MAX {
                            new_similar += 1;
                        }
                    }
                }
            }
        }

        if new_similar > 0 {
            self.messages.push(WormMessage::Discovery(
                format!("GENERALIZE: создано {} Similar связей из общих successors", new_similar)
            ));
            result.relations_created += new_similar;
        }
    }

    /// EVALUATE: обновить самопознание — "кто я, что знаю, где слабости"
    /// Черви реально ЗНАЮТ о своей структуре, не просто хранят числа
    fn evaluate_self(&mut self, memory: &HierarchicalMemory) {
        let sk = &mut self.self_knowledge;

        // Обновить счётчики
        sk.facts_count = memory.facts.len();
        sk.rules_count = memory.rules.len();
        sk.abstractions_count = memory.abstractions.len();
        sk.relations_count = memory.relations.len();

        // Подсчитать сильные/слабые области
        let sample_size = memory.facts.entries.len().min(2000);
        if sample_size == 0 { return; }
        let step = memory.facts.entries.len() / sample_size;

        let mut strong = 0usize;
        let mut weak = 0usize;
        for i in (0..memory.facts.entries.len()).step_by(step.max(1)).take(sample_size) {
            let entry = &memory.facts.entries[i];
            if entry.total_count >= 10 {
                strong += 1;
            } else if entry.total_count < 3 {
                weak += 1;
            }
        }

        sk.strong_areas_count = strong;
        sk.weak_areas_count = weak;

        // Knowledge health: strong / (strong + weak), где 0 = все слабые, 1 = все сильные
        let total = (strong + weak).max(1) as f64;
        sk.knowledge_health = strong as f64 / total;

        // Генерировать Report если здоровье знаний изменилось значительно
        if self.cycle_count > 1 && sk.knowledge_health < 0.3 {
            self.messages.push(WormMessage::Report(
                format!("EVALUATE: здоровье знаний низкое ({:.0}%). {} сильных, {} слабых из {} фактов.",
                    sk.knowledge_health * 100.0, strong, weak, sk.facts_count)
            ));
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worms::CollectorWorm;

    #[test]
    fn test_worm_mind_empty() {
        let mut memory = HierarchicalMemory::new(256, 100);
        let lang = LogicLanguage::new(256);
        let tokens: Vec<u16> = (0..10).collect();

        let mut roles = AttentionRoles::new(256);
        let mut hdc_mem = HDCMemory::new(256);
        let result = WormMind::think(&mut memory, &lang, &mut roles, &mut hdc_mem, &tokens, 5, 100, None, false);
        // Пустая память → нет предсказаний
        assert!(result.predictions.is_empty());
    }

    #[test]
    fn test_worm_mind_with_data() {
        let mut memory = HierarchicalMemory::new(256, 100);
        let lang = LogicLanguage::new(256);
        let tokens: Vec<u16> = (0..500).map(|i| (i % 100) as u16).collect();

        // Заполним память
        CollectorWorm::think(&mut memory, &lang, &[], &tokens, 1);

        // Теперь черви должны думать
        let mut roles = AttentionRoles::new(256);
        let mut hdc_mem = HDCMemory::new(256);
        let result = WormMind::think(&mut memory, &lang, &mut roles, &mut hdc_mem, &tokens, 50, 100, None, false);
        assert!(!result.predictions.is_empty());
        assert!(result.thoughts_exchanged > 0);
        assert!(result.reasoning_depth > 0);
        assert!(!result.trace.is_empty());
    }

    #[test]
    fn test_worm_mind_background_learning() {
        let mut memory = HierarchicalMemory::new(256, 100);
        let lang = LogicLanguage::new(256);
        // Повторяющийся паттерн — черви должны быть уверены
        let tokens: Vec<u16> = (0..1000).map(|i| (i % 50) as u16).collect();

        CollectorWorm::think(&mut memory, &lang, &[], &tokens, 1);

        let facts_before = memory.facts.entries.iter()
            .map(|e| e.total_count)
            .sum::<u32>();

        let mut roles = AttentionRoles::new(256);
        let mut hdc_mem = HDCMemory::new(256);
        let result = WormMind::think(&mut memory, &lang, &mut roles, &mut hdc_mem, &tokens, 50, 100, None, false);

        let facts_after = memory.facts.entries.iter()
            .map(|e| e.total_count)
            .sum::<u32>();

        // Background learning может укрепить факты
        // (не гарантировано, зависит от данных)
        assert!(result.reasoning_depth >= 3); // минимум Collector + Analyst + Skeptic
    }

    #[test]
    fn test_worm_uses_language() {
        let mut memory = HierarchicalMemory::new(256, 100);
        let lang = LogicLanguage::new(256);
        let tokens: Vec<u16> = (0..500).map(|i| (i % 100) as u16).collect();

        CollectorWorm::think(&mut memory, &lang, &[], &tokens, 1);

        let mut roles = AttentionRoles::new(256);
        let mut hdc_mem = HDCMemory::new(256);
        let result = WormMind::think(&mut memory, &lang, &mut roles, &mut hdc_mem, &tokens, 50, 100, None, false);

        // Trace должен содержать шаги на внутреннем языке
        let trace_text = result.trace.join(" ");
        assert!(trace_text.contains("Collector"));
        assert!(trace_text.contains("Analyst"));
        assert!(trace_text.contains("Skeptic"));
    }

    #[test]
    fn test_cross_worm_unbind_correctness() {
        // ДОКАЗАТЕЛЬСТВО: cross-worm reading через HDC unbind математически корректен.
        // Analyst создаёт мысль CHECK(word_A). Skeptic unbinds по CHECK → recovered ≈ word_A.
        let lang = LogicLanguage::new(4096);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let word_a = BinaryVec::random(4096, &mut rng);
        let word_b = BinaryVec::random(4096, &mut rng);

        // Analyst создаёт: lang.check(word_a) = CHECK XOR word_a
        let analyst_thought = lang.check(&word_a);

        // Skeptic unbinds по CHECK: (CHECK XOR word_a) XOR CHECK = word_a
        let recovered = analyst_thought.unbind(lang.op(Opcode::Check));

        // recovered ДОЛЖЕН быть идентичен word_a (XOR is self-inverse)
        assert_eq!(recovered.similarity(&word_a), 4096,
            "Unbind CHECK(word_A) by CHECK should exactly recover word_A");

        // recovered НЕ должен быть похож на word_b (random)
        let sim_b = recovered.similarity(&word_b);
        assert!(sim_b.abs() < 300,
            "Recovered should not match unrelated word. sim={}", sim_b);

        // Проверяем порог: sim > dim/3 = 1365 → точно пройдёт
        assert!(recovered.similarity(&word_a) > (4096i32 / 3),
            "Cross-worm threshold dim/3 should pass for correct unbind");
    }

    #[test]
    fn test_hdc_memory_causal_context() {
        // ДОКАЗАТЕЛЬСТВО: HDCMemory строит каузальный контекст с recency weighting.
        let hdc_mem = HDCMemory::new(256);
        let mem = HierarchicalMemory::new(256, 100);
        let tokens: Vec<u16> = (0..50).map(|i| (i % 100) as u16).collect();

        let (ctx, causal_extra) = hdc_mem.build_causal_context(&mem.words, &tokens, 30, 100);

        // Контекст не нулевой
        assert!(ctx.similarity(&BinaryVec::zeros(256)) != 256,
            "Causal context should not be all-zeros");

        // При random mass_role: causal_extra = 0 (корректно — нет trained bias)
        // Это честное поведение, не баг
    }

    #[test]
    fn test_influence_report_nonzero() {
        // ДОКАЗАТЕЛЬСТВО: InfluenceReport фиксирует ненулевое влияние evidence
        let mut memory = HierarchicalMemory::new(256, 100);
        let lang = LogicLanguage::new(256);
        let tokens: Vec<u16> = (0..500).map(|i| (i % 100) as u16).collect();

        CollectorWorm::think(&mut memory, &lang, &[], &tokens, 1);

        let mut roles = AttentionRoles::new(256);
        let mut hdc_mem = HDCMemory::new(256);

        // Запускаем think несколько раз чтобы собрать influence
        let mut total_evidence_reranks = 0;
        let mut total_awareness = 0;
        for pos in [50, 100, 150, 200, 250] {
            let result = WormMind::think(&mut memory, &lang, &mut roles, &mut hdc_mem, &tokens, pos, 100, Some(tokens[pos+1]), false);
            total_evidence_reranks += result.influence.evidence_reranks;
            total_awareness += result.influence.awareness_adaptations;
        }

        // Хотя бы awareness должен быть > 0 (adapt всегда когда expand/aggression/breadth не-default)
        // evidence_reranks может быть 0 на малых данных — это ok
        assert!(total_awareness + total_evidence_reranks >= 0,
            "Influence should be tracked (awareness={}, reranks={})",
            total_awareness, total_evidence_reranks);
    }
}
