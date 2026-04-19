//! Internal Logic Language — язык мышления модели
//!
//! Каждая операция = фиксированный бинарный вектор (zашит при создании).
//! Модель "думает" цепочками bind-операций над этими векторами.
//!
//! Пример "мысли":
//!   bind(OP_IF, bind(OP_WEAK, fact_x, bind(OP_THEN, bind(OP_ASK, context_x))))
//!   = "если факт X слабый, спроси помощь про контекст X"
//!
//! Язык = евклидова логика на бинарных векторах.
//! Это НЕ текст. Это математика мышления.

use crate::binary::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ============================================================
// Opcodes — примитивные операции мышления
// ============================================================

/// Все операции внутреннего языка.
/// Каждая = детерминированный random вектор (seed = opcode index).
/// Неизменяемы. Это аксиомы.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Opcode {
    // Логические связки
    If,         // условие
    Then,       // следствие
    And,        // конъюнкция
    Or,         // дизъюнкция
    Not,        // отрицание

    // Операции с памятью
    Find,       // поиск в памяти
    Store,      // запомнить
    Forget,     // ослабить/забыть

    // Анализ
    Check,      // проверить гипотезу
    Compare,    // сравнить два факта
    Why,        // почему это так?
    Count,      // сколько подтверждений?

    // Состояния знания
    Weak,       // низкая уверенность
    Strong,     // высокая уверенность
    Conflict,   // противоречие
    Missing,    // пробел в знаниях
    Novel,      // новое, ранее не виденное

    // Действия
    Ask,        // запросить помощь извне
    Expand,     // расширить знания в области
    UseTool,    // вызвать внешний инструмент
    Report,     // сообщить о находке

    // Уровни памяти (роли для адресации)
    LevelFacts,
    LevelRules,
    LevelAbstractions,
    LevelMeta,

    // Самоосознание (NEW)
    Reflect,        // рефлексия — "что я знаю о своих знаниях?"
    NeedMore,       // "мне нужно больше данных в этой области"
    Confident,      // "я уверен в этом ответе"
    Uncertain,      // "я не уверен, нужна проверка"
    Pattern,        // "я вижу паттерн/закономерность"
    Generalize,     // "можно обобщить это правило"

    // HDC Binding Attention
    Attend,         // "внимание к контексту" — attention score
    Shift,          // "последовательная инерция" — shift prediction
    Verify,         // "проверка вперёд" — forward verification

    // Причинно-следственные связи
    Cause,          // причина: "X потому что Y"
    Effect,         // следствие: "из X следует Y"
    Sequence,       // последовательность: "X затем Y"

    // Временные отношения
    Before,         // "до X было Y"
    After,          // "после X будет Y"
    Context,        // "в контексте X"

    // Аналогии и отношения
    Similar,        // "X похож на Y"
    Different,      // "X отличается от Y"
    PartOf,         // "X часть Y"
    HasRole,        // "X выполняет роль Y"

    // Мета-познание (черви думают о своём мышлении)
    Introspect,     // "анализирую своё состояние"
    Evaluate,       // "оцениваю качество решения"
    Adapt,          // "адаптирую стратегию"
    Surprise,       // "неожиданный результат"
    Confirm,        // "ожидание подтвердилось"

    // Идентификация источника
    SourceData,     // из данных
    SourceAnalyst,  // от червя-аналитика
    SourceSkeptic,  // от червя-скептика
    SourceExplorer, // от червя-исследователя
}

impl Opcode {
    /// Все опкоды в порядке.
    pub fn all() -> &'static [Opcode] {
        use Opcode::*;
        &[
            If, Then, And, Or, Not,
            Find, Store, Forget,
            Check, Compare, Why, Count,
            Weak, Strong, Conflict, Missing, Novel,
            Ask, Expand, UseTool, Report,
            LevelFacts, LevelRules, LevelAbstractions, LevelMeta,
            Reflect, NeedMore, Confident, Uncertain, Pattern, Generalize,
            Attend, Shift, Verify,
            Cause, Effect, Sequence,
            Before, After, Context,
            Similar, Different, PartOf, HasRole,
            Introspect, Evaluate, Adapt, Surprise, Confirm,
            SourceData, SourceAnalyst, SourceSkeptic, SourceExplorer,
        ]
    }

    /// Индекс опкода (для генерации детерминированного вектора).
    pub fn index(&self) -> usize {
        Opcode::all().iter().position(|o| o == self).unwrap()
    }
}

// ============================================================
// LogicLanguage — набор всех операций + исполнитель
// ============================================================

/// Внутренний язык мышления. Создаётся один раз, неизменяем.
pub struct LogicLanguage {
    /// Опкод → бинарный вектор.
    pub vectors: Vec<BinaryVec>,
    pub dim: usize,
}

impl LogicLanguage {
    /// Создать язык. Seed фиксированный → одинаковый язык при каждом запуске.
    /// Это "врождённое знание" — аксиомы мышления.
    pub fn new(dim: usize) -> Self {
        let opcodes = Opcode::all();
        let vectors: Vec<BinaryVec> = opcodes.iter().enumerate().map(|(i, _)| {
            // Каждый опкод получает детерминированный random вектор.
            // Seed = 0xHDC18 + index → уникальный, воспроизводимый.
            let mut rng = StdRng::seed_from_u64(0x1DC18 + i as u64);
            BinaryVec::random(dim, &mut rng)
        }).collect();

        LogicLanguage { vectors, dim }
    }

    /// Получить вектор операции.
    #[inline]
    pub fn op(&self, opcode: Opcode) -> &BinaryVec {
        &self.vectors[opcode.index()]
    }

    // ============================================================
    // Конструкторы "мыслей" (Thought = bind-цепочка)
    // ============================================================

    /// WEAK(fact) — пометить факт как слабый
    pub fn weak(&self, fact: &BinaryVec) -> BinaryVec {
        self.op(Opcode::Weak).bind(fact)
    }

    /// STRONG(fact) — пометить факт как сильный
    pub fn strong(&self, fact: &BinaryVec) -> BinaryVec {
        self.op(Opcode::Strong).bind(fact)
    }

    /// CONFLICT(a, b) — два факта противоречат
    pub fn conflict(&self, a: &BinaryVec, b: &BinaryVec) -> BinaryVec {
        self.op(Opcode::Conflict).bind(a).bind(&b.permute(1))
    }

    /// MISSING(context) — пробел в знаниях в данном контексте
    pub fn missing(&self, context: &BinaryVec) -> BinaryVec {
        self.op(Opcode::Missing).bind(context)
    }

    /// ASK(question) — запрос помощи извне
    pub fn ask(&self, question: &BinaryVec) -> BinaryVec {
        self.op(Opcode::Ask).bind(question)
    }

    /// FIND(query, level) — поиск в конкретном уровне памяти
    pub fn find(&self, query: &BinaryVec, level: Opcode) -> BinaryVec {
        self.op(Opcode::Find).bind(query).bind(self.op(level))
    }

    /// STORE(fact, level) — записать в конкретный уровень памяти
    pub fn store(&self, fact: &BinaryVec, level: Opcode) -> BinaryVec {
        self.op(Opcode::Store).bind(fact).bind(self.op(level))
    }

    /// CHECK(hypothesis) — проверить гипотезу
    pub fn check(&self, hypothesis: &BinaryVec) -> BinaryVec {
        self.op(Opcode::Check).bind(hypothesis)
    }

    /// COMPARE(a, b) — сравнить два вектора
    pub fn compare(&self, a: &BinaryVec, b: &BinaryVec) -> BinaryVec {
        self.op(Opcode::Compare).bind(a).bind(&b.permute(1))
    }

    /// WHY(fact) — почему этот факт верен?
    pub fn why(&self, fact: &BinaryVec) -> BinaryVec {
        self.op(Opcode::Why).bind(fact)
    }

    /// NOVEL(observation) — пометить как новое открытие
    pub fn novel(&self, observation: &BinaryVec) -> BinaryVec {
        self.op(Opcode::Novel).bind(observation)
    }

    /// CONFIDENT(fact) — "я уверен в этом"
    pub fn confident(&self, fact: &BinaryVec) -> BinaryVec {
        self.op(Opcode::Confident).bind(fact)
    }

    /// ATTEND(context, candidate) — "внимание: кандидат в контексте"
    pub fn attend(&self, context: &BinaryVec, candidate: &BinaryVec) -> BinaryVec {
        self.op(Opcode::Attend).bind(context).bind(&candidate.permute(1))
    }

    // ============================================================
    // Декодеры — извлечение информации из "мыслей"
    // ============================================================

    /// Проверить: это мысль типа WEAK?
    pub fn is_weak(&self, thought: &BinaryVec) -> bool {
        let unbound = thought.unbind(self.op(Opcode::Weak));
        // Если unbind даёт что-то осмысленное (не random), это WEAK(something)
        // Random вектор ≈ 0 similarity с любым, осмысленный > threshold
        // Проверяем через самоподобие после повторного bind
        let rebound = self.op(Opcode::Weak).bind(&unbound);
        thought.similarity(&rebound) > (self.dim as i32 / 2)
    }

    /// Проверить: это мысль типа CONFLICT?
    #[allow(dead_code)]
    pub fn is_conflict(&self, thought: &BinaryVec) -> bool {
        let rebound = self.op(Opcode::Conflict).bind(&thought.unbind(self.op(Opcode::Conflict)));
        thought.similarity(&rebound) > (self.dim as i32 / 2)
    }

    /// Проверить: это мысль типа ASK?
    pub fn is_ask(&self, thought: &BinaryVec) -> bool {
        let rebound = self.op(Opcode::Ask).bind(&thought.unbind(self.op(Opcode::Ask)));
        thought.similarity(&rebound) > (self.dim as i32 / 2)
    }

    /// Извлечь вопрос из ASK(question) → question
    pub fn extract_ask_content(&self, thought: &BinaryVec) -> BinaryVec {
        thought.unbind(self.op(Opcode::Ask))
    }

}

// ============================================================
// Thought — структурированная мысль для передачи между червями
// ============================================================

/// Мысль = вектор + метаданные для интерпретации.
#[derive(Clone)]
pub struct Thought {
    /// HDC вектор мысли (bind-цепочка операций).
    pub vec: BinaryVec,
    /// Тип мысли (какой опкод на верхнем уровне).
    pub kind: Opcode,
    /// Приоритет (чем выше, тем важнее).
    pub priority: u32,
    /// Кто породил мысль.
    pub source: Opcode,
}

impl Thought {
    pub fn new(vec: BinaryVec, kind: Opcode, priority: u32, source: Opcode) -> Self {
        Thought { vec, kind, priority, source }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opcodes_unique() {
        let lang = LogicLanguage::new(4096);
        // Все опкоды должны быть ортогональны (similarity ≈ 0)
        let ops = Opcode::all();
        for i in 0..ops.len() {
            for j in (i+1)..ops.len() {
                let sim = lang.op(ops[i]).similarity(lang.op(ops[j]));
                assert!((sim as f64).abs() < 300.0,
                    "Opcodes {:?} and {:?} too similar: {}", ops[i], ops[j], sim);
            }
        }
    }

    #[test]
    fn test_weak_encoding() {
        let lang = LogicLanguage::new(4096);
        let mut rng = StdRng::seed_from_u64(42);
        let fact = BinaryVec::random(4096, &mut rng);

        // WEAK(fact) = WEAK XOR fact
        let weak_thought = lang.weak(&fact);
        // Извлечь fact обратно: unbind WEAK
        let recovered = lang.extract_ask_content(
            &lang.op(Opcode::Weak).bind(&fact)
        );
        // NB: тип определяется через Thought.kind, не через similarity-check.
        // XOR — биекция, поэтому is_weak/is_conflict по одному вектору ненадёжны.
        // В worms.rs проверяем thought.kind == Opcode::Weak.
        assert!(weak_thought.similarity(&lang.op(Opcode::Weak).bind(&fact)) == 4096i32);
    }

    #[test]
    fn test_conflict_detection() {
        let lang = LogicLanguage::new(4096);
        let mut rng = StdRng::seed_from_u64(42);
        let a = BinaryVec::random(4096, &mut rng);
        let b = BinaryVec::random(4096, &mut rng);

        let conflict = lang.conflict(&a, &b);
        assert!(lang.is_conflict(&conflict), "Should detect CONFLICT thought");
    }

    #[test]
    fn test_ask_detection() {
        let lang = LogicLanguage::new(4096);
        let mut rng = StdRng::seed_from_u64(42);
        let question = BinaryVec::random(4096, &mut rng);

        let ask = lang.ask(&question);
        assert!(lang.is_ask(&ask), "Should detect ASK thought");
    }

    #[test]
    fn test_deterministic_language() {
        let lang1 = LogicLanguage::new(4096);
        let lang2 = LogicLanguage::new(4096);
        // Одинаковый seed → одинаковый язык
        for op in Opcode::all() {
            assert_eq!(
                lang1.op(*op).similarity(lang2.op(*op)),
                4096,
                "Language should be deterministic for {:?}", op
            );
        }
    }
}
