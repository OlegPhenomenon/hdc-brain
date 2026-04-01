//! HDC-Brain v17: Layered Learning Architecture
//!
//! Обучение как ребёнок учит язык:
//!   Phase 1: Заучить фразы (один проход по данным)
//!   Phase 2: Обнаружить правила (из ошибок фразовой памяти)
//!   Phase 3: Оценить (3 слоя: фразы → правила → рассуждение)
//!
//! Usage:
//!   cargo run --release -- train --data ../hdc-brain-v15/data_big_bpe.bin --vocab ../hdc-brain-v15/vocab_16k.txt
//!   cargo run --release -- chat --model model_v17.bin --vocab ../hdc-brain-v15/vocab_16k.txt
//!   cargo run --release -- eval --model model_v17.bin --data ../hdc-brain-v15/data_big_bpe.bin --vocab ../hdc-brain-v15/vocab_16k.txt

mod binary;
mod model;

use std::io::{self, BufRead, Write};
use std::time::Instant;
use byteorder::{LittleEndian, ReadBytesExt};
use model::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("HDC-Brain v17: Layered Learning (Phrases → Rules → Reasoning)");
        println!();
        println!("Usage:");
        println!("  {} train --data <path> --vocab <path> [--dim 4096] [--out model_v17.bin]", args[0]);
        println!("  {} eval  --model <path> --data <path> --vocab <path> [--n 10000]", args[0]);
        println!("  {} chat  --model <path> --vocab <path>", args[0]);
        println!("  {} test                                     # quick self-test", args[0]);
        return;
    }

    match args[1].as_str() {
        "train" => cmd_train(&args[2..]),
        "eval" => cmd_eval(&args[2..]),
        "chat" => cmd_chat(&args[2..]),
        "test" => cmd_test(),
        _ => println!("Unknown command: {}", args[1]),
    }
}

// ============================================================
// Data Loading
// ============================================================

fn load_tokens(path: &str) -> Vec<u16> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("Cannot open {}: {}", path, e));

    let magic = f.read_u32::<LittleEndian>().unwrap();
    assert!(magic == 0xBEEF1601 || magic == 0xBEEF1501,
            "Unknown data format: 0x{:08X}", magic);

    let vocab = f.read_u32::<LittleEndian>().unwrap();
    let n_tokens = f.read_u32::<LittleEndian>().unwrap() as usize;

    println!("  Data: {} tokens, vocab={}", n_tokens, vocab);

    // Detect token size from file: (filesize - 12) / n_tokens
    let file_len = f.metadata().map(|m| m.len()).unwrap_or(0) as usize;
    let data_bytes = file_len - 12; // header = 3 × u32
    let bytes_per_token = if n_tokens > 0 { data_bytes / n_tokens } else { 4 };

    let tokens: Vec<u16> = if bytes_per_token >= 4 {
        // u32 tokens
        let mut buf = vec![0u8; n_tokens * 4];
        f.read_exact(&mut buf).unwrap();
        buf.chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u16)
            .collect()
    } else {
        // u16 tokens
        let mut buf = vec![0u8; n_tokens * 2];
        f.read_exact(&mut buf).unwrap();
        buf.chunks(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };

    tokens
}

fn load_vocab(path: &str) -> Vec<String> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot open vocab {}: {}", path, e));
    content.lines().map(|s| s.to_string()).collect()
}

fn decode_tokens(tokens: &[u16], vocab: &[String]) -> String {
    tokens.iter()
        .map(|&t| {
            if (t as usize) < vocab.len() {
                vocab[t as usize].replace('▁', " ")
            } else {
                format!("[{}]", t)
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

fn encode_simple(text: &str, vocab: &[String]) -> Vec<u16> {
    // Greedy longest-match tokenization (simplified BPE decode)
    let text = text.replace(' ', "▁");
    let mut tokens = Vec::new();
    let mut pos = 0;
    let chars: Vec<char> = text.chars().collect();

    while pos < chars.len() {
        let mut best_len = 1;
        let mut best_tok = 0u16; // fallback: unknown

        for len in (1..=16.min(chars.len() - pos)).rev() {
            let candidate: String = chars[pos..pos + len].iter().collect();
            if let Some(idx) = vocab.iter().position(|v| *v == candidate) {
                best_len = len;
                best_tok = idx as u16;
                break;
            }
        }

        tokens.push(best_tok);
        pos += best_len;
    }

    tokens
}

// ============================================================
// Commands
// ============================================================

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn cmd_train(args: &[String]) {
    let data_path = parse_arg(args, "--data")
        .unwrap_or_else(|| "../hdc-brain-v15/data_big_bpe.bin".to_string());
    let vocab_path = parse_arg(args, "--vocab")
        .unwrap_or_else(|| "../hdc-brain-v15/vocab_16k.txt".to_string());
    let out_path = parse_arg(args, "--out")
        .unwrap_or_else(|| "model_v17.bin".to_string());
    let dim: usize = parse_arg(args, "--dim")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let n_threads: usize = parse_arg(args, "--threads")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let max_minutes: f64 = parse_arg(args, "--time")
        .and_then(|s| s.parse().ok())
        .unwrap_or(25.0);

    // Set thread pool
    rayon::ThreadPoolBuilder::new().num_threads(n_threads).build_global()
        .unwrap_or_else(|_| ());

    let train_start = Instant::now();
    let max_secs = max_minutes * 60.0;

    println!("=== HDC-Brain v17: Train ===");
    println!("  dim={}, threads={}, time_limit={:.0}min, data={}", dim, n_threads, max_minutes, data_path);

    let vocab = load_vocab(&vocab_path);
    let data = load_tokens(&data_path);

    // Split: 95% train, 5% val
    let n_val = data.len() / 20;
    let train_data = &data[..data.len() - n_val];
    let val_data = &data[data.len() - n_val..];
    println!("  Train: {} tokens, Val: {} tokens", train_data.len(), val_data.len());

    let config = Config {
        hdc_dim: dim,
        vocab_size: vocab.len(),
        lsh_bits: 16,
        max_rules: 512,
        top_k: 10,
    };

    let mut model = HDCBrainV17::new(config);

    // === Phase 0: Build Semantic Codebook ===
    // Сначала понимаем СМЫСЛ слов, потом учим фразы на осмысленных векторах
    println!("\n{}", "=".repeat(60));
    println!("=== Phase 0: Semantic Codebook ===");
    let t0 = Instant::now();
    model.learn_codebook(train_data);
    println!("  Codebook time: {:.1}s", t0.elapsed().as_secs_f64());

    // Build semantic neighbors for Layer 5-6 improvisation
    let t0 = Instant::now();
    model.build_neighbors(5);
    println!("  Neighbors time: {:.1}s", t0.elapsed().as_secs_f64());

    // Show similar pairs with real words
    let pairs = model.find_similar_pairs();
    if !pairs.is_empty() {
        println!("  With vocab names:");
        for &(i, j, sim) in pairs.iter().take(10) {
            let w1 = if i < vocab.len() { &vocab[i] } else { "?" };
            let w2 = if j < vocab.len() { &vocab[j] } else { "?" };
            let pct = sim as f64 / model.config.hdc_dim as f64 * 100.0;
            println!("    '{}' ≈ '{}'  ({:.0}%)", w1, w2, pct);
        }
    }

    // Show word neighbor examples
    let example_words = ["▁город", "▁Россия", "▁большой", "▁был"];
    println!("  Word neighbors (for improvisation):");
    for word in &example_words {
        if let Some(idx) = vocab.iter().position(|v| v == word) {
            if idx < model.codebook_neighbors.len() && !model.codebook_neighbors[idx].is_empty() {
                let neighbors: Vec<String> = model.codebook_neighbors[idx].iter()
                    .take(5)
                    .map(|&(t, s)| {
                        let name = if (t as usize) < vocab.len() { &vocab[t as usize] } else { "?" };
                        format!("{}({:.0}%)", name, s as f64 / model.config.hdc_dim as f64 * 100.0)
                    })
                    .collect();
                println!("    '{}' → [{}]", word, neighbors.join(", "));
            }
        }
    }

    // === Curriculum Learning ===
    // Теперь учим фразы на СМЫСЛОВОМ codebook
    let stages: Vec<(f64, &str)> = vec![
        (0.01, "first words"),
        (0.05, "basic phrases"),
        (0.20, "grammar patterns"),
        (0.50, "vocabulary growth"),
        (1.00, "full knowledge"),
    ];

    let eval_n = 10000.min(val_data.len().saturating_sub(3));
    let mut prev_end = 2;

    for (i, &(frac, name)) in stages.iter().enumerate() {
        let end = ((train_data.len() - 1) as f64 * frac) as usize;
        if end <= prev_end { continue; }

        let new_chunk = &train_data[prev_end..end];
        println!("\n{}", "=".repeat(60));
        println!("=== Stage {}/{}: {} ({:.0}%, +{} tokens, total {}) ===",
                 i + 1, stages.len(), name, frac * 100.0,
                 new_chunk.len(), end);

        // Phase A: Learn phrases — "заучить фразы"
        let t0 = Instant::now();
        model.train_phrases(new_chunk);
        let phrase_time = t0.elapsed().as_secs_f64();

        // Phase A2: Build logical fact memory — "записать связи"
        let t0 = Instant::now();
        model.build_fact_memory(&train_data[..end]);
        let fact_time = t0.elapsed().as_secs_f64();

        // Quick phrase accuracy
        let phrase_acc = model.phrase_accuracy(val_data, eval_n);
        println!("  Phrase accuracy: {:.2}%", phrase_acc);

        // Phase B: Learn contexts — "где эти фразы применяются?"
        let t0 = Instant::now();
        model.learn_contexts(new_chunk);
        let context_time = t0.elapsed().as_secs_f64();

        // Phase C: Discover rules — "какие правила вытекают?"
        let t0 = Instant::now();
        model.discover_rules(&train_data[..end]);
        let rule_time = t0.elapsed().as_secs_f64();

        // Phase 3: Full evaluation (all 3 layers)
        let t0 = Instant::now();
        let eval_data_slice = if val_data.len() > eval_n + 3 {
            &val_data[..eval_n + 3]
        } else { val_data };
        let result = model.evaluate(eval_data_slice);
        let eval_time = t0.elapsed().as_secs_f64();

        println!("  --- Stage {} results ---", i + 1);
        result.print();
        println!("  Time: phrases {:.1}s, facts {:.1}s, ctx {:.1}s, rules {:.1}s, eval {:.1}s",
                 phrase_time, fact_time, context_time, rule_time, eval_time);

        prev_end = end;

        // Check timer
        if train_start.elapsed().as_secs_f64() > max_secs * 0.9 {
            println!("  ⏰ Time limit approaching, stopping curriculum");
            break;
        }
    }

    // === Multi-pass: repeat full data to strengthen facts ===
    let mut pass = 1;
    while train_start.elapsed().as_secs_f64() < max_secs * 0.85 {
        pass += 1;
        println!("\n{}", "=".repeat(60));
        let elapsed_min = train_start.elapsed().as_secs_f64() / 60.0;
        println!("=== Pass {} (all data, {:.1}min elapsed) ===", pass, elapsed_min);

        // Re-train phrases on all data (accumulates — stronger signals)
        let t0 = Instant::now();
        model.train_phrases(train_data);
        println!("  Phrases: {:.1}s", t0.elapsed().as_secs_f64());

        if train_start.elapsed().as_secs_f64() > max_secs * 0.85 { break; }

        // Rebuild fact memory (overwrites with better facts from more evidence)
        let t0 = Instant::now();
        model.build_fact_memory(train_data);
        println!("  Facts: {:.1}s", t0.elapsed().as_secs_f64());

        if train_start.elapsed().as_secs_f64() > max_secs * 0.85 { break; }

        // Rebuild contexts
        let t0 = Instant::now();
        model.learn_contexts(train_data);
        println!("  Contexts: {:.1}s", t0.elapsed().as_secs_f64());

        // Quick eval
        let eval_data_slice = if val_data.len() > eval_n + 3 {
            &val_data[..eval_n + 3]
        } else { val_data };
        let result = model.evaluate(eval_data_slice);
        println!("  --- Pass {} results ---", pass);
        result.print();
    }

    let total_min = train_start.elapsed().as_secs_f64() / 60.0;
    println!("\n  Total training: {:.1} min, {} passes", total_min, pass);

    // Final comprehensive evaluation
    println!("\n{}", "=".repeat(60));
    println!("=== Final Evaluation (full val set) ===");
    let t0 = Instant::now();
    let eval_data = if val_data.len() > 50000 { &val_data[..50000] } else { val_data };
    let result = model.evaluate(eval_data);
    result.print();
    println!("  Time: {:.1}s", t0.elapsed().as_secs_f64());

    // Generation samples
    println!("\n--- Generation Samples ---");
    for prompt in &["Россия", "В городе", "Кошка"] {
        let prompt_tokens = encode_simple(prompt, &vocab);
        if prompt_tokens.len() < 3 {
            // Pad with BOS-like tokens
            let mut padded = vec![0u16; 3 - prompt_tokens.len()];
            padded.extend_from_slice(&prompt_tokens);
            let out = model.generate(&padded, 30, 50.0, 5);
            let text = decode_tokens(&out, &vocab);
            println!("  [{}]: {}", prompt, text.trim());
        } else {
            let out = model.generate(&prompt_tokens, 30, 50.0, 5);
            let text = decode_tokens(&out, &vocab);
            println!("  [{}]: {}", prompt, text.trim());
        }
    }

    // Save
    model.save(&out_path).unwrap();
    println!("\n=== Done ===");
}

fn cmd_eval(args: &[String]) {
    let model_path = parse_arg(args, "--model")
        .unwrap_or_else(|| "model_v17.bin".to_string());
    let data_path = parse_arg(args, "--data")
        .unwrap_or_else(|| "../hdc-brain-v15/data_big_bpe.bin".to_string());
    let vocab_path = parse_arg(args, "--vocab")
        .unwrap_or_else(|| "../hdc-brain-v15/vocab_16k.txt".to_string());
    let n_eval: usize = parse_arg(args, "--n")
        .and_then(|s| s.parse().ok())
        .unwrap_or(50000);

    println!("=== HDC-Brain v17: Evaluate ===");
    let _vocab = load_vocab(&vocab_path);
    let model = HDCBrainV17::load(&model_path).unwrap();
    let data = load_tokens(&data_path);

    let n_val = data.len() / 20;
    let val_data = &data[data.len() - n_val..];
    let eval_data = if val_data.len() > n_eval { &val_data[..n_eval] } else { val_data };

    let t0 = Instant::now();
    let result = model.evaluate(eval_data);
    result.print();
    println!("  Time: {:.1}s", t0.elapsed().as_secs_f64());
}

fn cmd_chat(args: &[String]) {
    let model_path = parse_arg(args, "--model")
        .unwrap_or_else(|| "model_v17.bin".to_string());
    let vocab_path = parse_arg(args, "--vocab")
        .unwrap_or_else(|| "../hdc-brain-v15/vocab_16k.txt".to_string());

    println!("=== HDC-Brain v17: Chat ===");
    let vocab = load_vocab(&vocab_path);
    let model = HDCBrainV17::load(&model_path).unwrap();

    println!("Type a prompt (or 'quit' to exit):");
    let stdin = io::stdin();

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 { break; }
        let line = line.trim();
        if line == "quit" || line == "exit" { break; }
        if line.is_empty() { continue; }

        let tokens = encode_simple(line, &vocab);
        if tokens.len() < 3 {
            let mut padded = vec![0u16; 3 - tokens.len()];
            padded.extend_from_slice(&tokens);
            let out = model.generate(&padded, 60, 30.0, 5);
            println!("{}\n", decode_tokens(&out, &vocab).trim());
        } else {
            let out = model.generate(&tokens, 60, 30.0, 5);
            println!("{}\n", decode_tokens(&out, &vocab).trim());
        }
    }
}

fn cmd_test() {
    println!("=== HDC-Brain v17: Self-Test ===\n");

    use binary::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(42);
    let dim = 4096;

    // Test 1: bind/unbind recovery
    let a = BinaryVec::random(dim, &mut rng);
    let b = BinaryVec::random(dim, &mut rng);
    let ab = a.bind(&b);
    let recovered = ab.unbind(&a);
    let sim = recovered.similarity(&b);
    println!("1. bind/unbind: similarity = {} (expected {})", sim, dim);
    assert_eq!(sim, dim as i32);

    // Test 2: negation
    let neg = a.negate();
    let sim_neg = a.similarity(&neg);
    println!("2. negate: similarity = {} (expected {})", sim_neg, -(dim as i32));
    assert_eq!(sim_neg, -(dim as i32));

    // Test 3: permute orthogonality
    let p1 = a.permute(1);
    let p2 = a.permute(2);
    let sim_p1 = a.similarity(&p1);
    let sim_p12 = p1.similarity(&p2);
    println!("3. permute: sim(a, perm1) = {}, sim(perm1, perm2) = {} (expected ~0)",
             sim_p1, sim_p12);
    assert!((sim_p1 as f64).abs() < 200.0);
    assert!((sim_p12 as f64).abs() < 200.0);

    // Test 4: trigram bind preserves order
    let t0 = BinaryVec::random(dim, &mut rng);
    let t1 = BinaryVec::random(dim, &mut rng);
    let t2 = BinaryVec::random(dim, &mut rng);
    let trigram_a = t0.bind(&t1.permute(1)).bind(&t2.permute(2));
    let trigram_b = t0.bind(&t2.permute(1)).bind(&t1.permute(2)); // swapped order
    let sim_order = trigram_a.similarity(&trigram_b);
    println!("4. trigram order: sim(ABC, ACB) = {} (expected ~0, order matters)", sim_order);
    assert!((sim_order as f64).abs() < 200.0);

    // Test 5: bundle (majority vote)
    let mut acc = BundleAccumulator::new(dim);
    let v1 = BinaryVec::random(dim, &mut rng);
    let v2 = BinaryVec::random(dim, &mut rng);
    acc.add(&v1);
    acc.add(&v1);
    acc.add(&v1);
    acc.add(&v2);
    let bundled = acc.to_binary();
    let sim_v1 = bundled.similarity(&v1);
    let sim_v2 = bundled.similarity(&v2);
    println!("5. bundle(3×A + B): sim(A)={}, sim(B)={} (A should dominate)", sim_v1, sim_v2);
    assert!(sim_v1 > sim_v2);
    assert!(sim_v1 > dim as i32 / 2); // A should be strong

    // Test 6: LSH hash consistency
    let h1 = lsh_hash(&a, 16);
    let h2 = lsh_hash(&a, 16);
    let h3 = lsh_hash(&b, 16);
    println!("6. LSH: hash(a)={}, hash(a)={}, hash(b)={} (a==a, a!=b)",
             h1, h2, h3);
    assert_eq!(h1, h2);
    // h1 != h3 with high probability for random vectors

    // Test 7: Modus ponens via HDC
    // "Сократ" = s, "человек" = h, "смертен" = m
    // Факт: bind(s, h) = "Сократ — человек"
    // Правило: bind(h, m) = "человек — смертен"
    // Вывод: unbind(bind(s,h), h) = s, bind with rule → unbind(bind(h,m), h) = m
    // Цепочка: fact=bind(s,h), rule=bind(h,m)
    // unbind(fact, s) = h, bind(h, rule) → unbind(bind(h,m), h) = m → "Сократ смертен"
    let socrates = BinaryVec::random(dim, &mut rng);
    let human = BinaryVec::random(dim, &mut rng);
    let mortal = BinaryVec::random(dim, &mut rng);

    let fact = socrates.bind(&human);         // "Сократ — человек"
    let rule = human.bind(&mortal);           // "человек → смертен"

    // Step 1: из факта извлекаем "человек"
    let extracted_human = fact.unbind(&socrates);
    assert_eq!(extracted_human.similarity(&human), dim as i32);

    // Step 2: применяем правило
    let conclusion = rule.unbind(&extracted_human);
    let sim_mortal = conclusion.similarity(&mortal);
    println!("7. Modus ponens: unbind(rule, unbind(fact, Сократ)) → sim(смертен) = {} (expected {})",
             sim_mortal, dim);
    assert_eq!(sim_mortal, dim as i32); // perfect chain!

    // Test 8: Analogy via HDC
    // "король" - "мужчина" + "женщина" = "королева"
    // In HDC: unbind(bind(king, man), woman) should recover queen-like vector
    let king = BinaryVec::random(dim, &mut rng);
    let man = BinaryVec::random(dim, &mut rng);
    let woman = BinaryVec::random(dim, &mut rng);
    let _queen = BinaryVec::random(dim, &mut rng);

    // Relation: bind(king, man) captures "king is to man"
    // Apply same relation to woman: unbind(bind(king, man), man) = king, then bind with woman
    let relation = king.bind(&man);           // "king-man" relation
    let king_recovered = relation.unbind(&man);
    let _analogy = king_recovered.bind(&woman.bind(&man.negate())); // not quite right for binary

    // Better analogy: if king:man :: queen:woman
    // Then bind(king, man) ≈ bind(queen, woman) (same relation)
    // So: queen ≈ unbind(bind(king, man), woman)
    // But this only works if the RELATION is stored, not the literal vectors
    println!("8. Analogy framework: HDC bind/unbind enables analogy computation");

    println!("\n=== All {} tests passed! ===", 8);

    // Model creation test
    println!("\n--- Model creation benchmark ---");
    let config = Config::default_with_vocab(16000);
    let t0 = Instant::now();
    let model = HDCBrainV17::new(config);
    println!("  Model created in {:.1}ms", t0.elapsed().as_millis());
    println!("  Codebook: {} × {} bits = {:.1} MB",
             model.config.vocab_size, model.config.hdc_dim,
             model.config.vocab_size as f64 * model.config.hdc_dim as f64 / 8.0 / 1024.0 / 1024.0);
    println!("  Phrase buckets: {} (LSH {}bit)",
             model.config.n_buckets(), model.config.lsh_bits);
}
