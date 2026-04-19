//! HDC-Brain v18: Conscious HDC Model
//!
//! Пайплайн:
//!   Phase 0: Semantic codebook (слова → вектора с семантикой)
//!   Phase 1: Worm Training (Collector→Analyst→Skeptic→Explorer)
//!   Phase 2: Evaluate
//!
//! Черви = алгоритм обучения. Они думают на внутреннем языке,
//! обмениваются мыслями, используют инструменты для работы с памятью.
//!
//! Usage:
//!   cargo run --release -- train --data ../hdc-brain-v15/data_big_bpe.bin --vocab ../hdc-brain-v15/vocab_16k.txt
//!   cargo run --release -- test

mod binary;
mod memory;
mod language;
mod worms;
mod model;
mod working_memory;
mod logger;
mod worm_mind;
mod pipeline;

use std::time::Instant;
use byteorder::{LittleEndian, ReadBytesExt};
use model::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("HDC-Brain v18: Conscious HDC Model");
        println!("  Worm-Driven Learning + Internal Language + Self-Awareness");
        println!();
        println!("Usage:");
        println!("  {} train --data <path> --vocab <path> [--dim 4096] [--threads 4]", args[0]);
        println!("         [--chunk 100000] [--sample-rate 1] [--worm-checks 10000] [--passes 1]");
        println!("  {} eval  --data <path> --vocab <path> [--n 10000]", args[0]);
        println!("  {} test                                     # quick self-test", args[0]);
        return;
    }

    match args[1].as_str() {
        "train" => cmd_train(&args[2..]),
        "train-pipeline" => cmd_train_pipeline(&args[2..]),
        "eval" => cmd_eval(&args[2..]),
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

    let n_tokens = f.read_u32::<LittleEndian>().unwrap() as usize;
    let vocab_size = f.read_u32::<LittleEndian>().unwrap() as usize;
    let _dim = f.read_u32::<LittleEndian>().unwrap();

    println!("Data: {} tokens, vocab_size={}", n_tokens, vocab_size);

    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();

    let bytes_per_token = buf.len() / n_tokens;
    println!("  Format: {} bytes/token ({})",
        bytes_per_token,
        if bytes_per_token == 2 { "u16" } else { "u32" });

    let tokens: Vec<u16> = if bytes_per_token <= 2 {
        buf.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    } else {
        buf.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u16)
            .collect()
    };

    println!("  Loaded {} tokens", tokens.len());
    tokens
}

fn load_vocab(path: &str) -> Vec<String> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read vocab {}: {}", path, e));
    content.lines().map(|l| l.to_string()).collect()
}

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn parse_arg_or(args: &[String], flag: &str, default: &str) -> String {
    parse_arg(args, flag).unwrap_or_else(|| default.to_string())
}

// ============================================================
// Train Command
// ============================================================

fn cmd_train(args: &[String]) {
    let data_path = parse_arg(args, "--data")
        .expect("--data <path> required");
    let vocab_path = parse_arg(args, "--vocab")
        .expect("--vocab <path> required");
    let dim: usize = parse_arg_or(args, "--dim", "4096").parse().unwrap();
    let threads: usize = parse_arg_or(args, "--threads", "4").parse().unwrap();
    let chunk_size: usize = parse_arg_or(args, "--chunk", "100000").parse().unwrap();
    let sample_rate: usize = parse_arg_or(args, "--sample-rate", "1").parse().unwrap();
    let worm_checks: usize = parse_arg_or(args, "--worm-checks", "10000").parse().unwrap();
    let n_passes: usize = parse_arg_or(args, "--passes", "1").parse().unwrap();
    let n_eval: usize = parse_arg_or(args, "--n-eval", "10000").parse().unwrap();
    let log_path = parse_arg_or(args, "--log", "training.log");

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok();

    // Создаём лог-файл
    let log = logger::Logger::create(&log_path);

    log.section("HDC-Brain v18: Worm-Driven + Working Memory");
    log.log(&format!("dim={}, threads={}, chunk={}, sample={}, checks={}, passes={}",
        dim, threads, chunk_size, sample_rate, worm_checks, n_passes));

    let total_start = Instant::now();

    let tokens = load_tokens(&data_path);
    let vocab = load_vocab(&vocab_path);
    let vocab_size = vocab.len();
    log.log(&format!("Data: {} tokens, vocab_size={}", tokens.len(), vocab_size));

    let mut config = Config::default_with_vocab(vocab_size);
    config.hdc_dim = dim;
    config.chunk_size = chunk_size;
    config.sample_rate = sample_rate;
    config.worm_checks = worm_checks;
    config.n_passes = n_passes;
    let mut model = HDCBrainV18::new(config);

    // === Phase 0: Semantic Codebook (с кешем) ===
    let codebook_cache = format!("codebook_{}_{}.bin", vocab_size, dim);
    log.section("Phase 0: Semantic Codebook");
    if model.memory.load_codebook(&codebook_cache) {
        log.log(&format!("Codebook loaded from cache: {}", codebook_cache));
    } else {
        model.learn_codebook(&tokens);
        if let Err(e) = model.memory.save_codebook(&codebook_cache) {
            log.log(&format!("Warning: failed to save codebook cache: {}", e));
        } else {
            log.log(&format!("Codebook saved to cache: {}", codebook_cache));
        }
    }

    let sample_tokens: Vec<u16> = find_sample_tokens(&vocab,
        &["Россия", "город", "большой", "один", "время"]);
    model.show_neighbors(&vocab, &sample_tokens);

    // === Phase 1: Worm Training ===
    model.train(&tokens, Some(&vocab), Some(&log));

    // === Phase 2: Final Evaluation with WormMind ===
    log.section(&format!("Final Evaluation ({} positions) with WormMind", n_eval));
    let eval_final = model.evaluate_mind_readonly(&tokens, n_eval);
    let t = eval_final.tested.max(1) as f64;
    log.metric_pct("Final Top-1", eval_final.correct_top1 as f64 / t * 100.0);
    log.metric_pct("Final Top-5", eval_final.correct_top5 as f64 / t * 100.0);
    log.metric("Final avg_reasoning", eval_final.avg_reasoning);
    log.metric("Final avg_confidence", eval_final.avg_confidence);
    log.metric("Final bg_updates", eval_final.total_bg_updates as f64);
    log.section("Self-Diagnostic Report");
    log.metric_pct("no_trigram_match", eval_final.pct_no_trigram);
    log.metric("avg_uncertain", eval_final.avg_uncertain);
    log.metric("avg_contradictions", eval_final.avg_contradictions);
    log.metric("need_more_data", eval_final.total_need_more as f64);

    let total_time = total_start.elapsed();
    log.section(&format!("Complete in {:.1}s ({:.1} min)",
        total_time.as_secs_f64(), total_time.as_secs_f64() / 60.0));
    log.log(&model.memory.stats());

    // Генерация на stdout (для быстрого взгляда)
    show_generation_samples(&mut model, &tokens, &vocab);

    log.log(&format!("Log saved to: {}", log_path));
}

fn find_sample_tokens(vocab: &[String], words: &[&str]) -> Vec<u16> {
    words.iter()
        .filter_map(|&w| {
            vocab.iter().position(|v| v == w).map(|i| i as u16)
        })
        .collect()
}

fn show_generation_samples(model: &mut HDCBrainV18, tokens: &[u16], vocab: &[String]) {
    use worm_mind::WormMind;

    println!("\n--- Generation Samples (WormMind) ---");
    let starts = [100, 5000, 20000, 50000, 100000];
    for &start in &starts {
        if start + 3 >= tokens.len() { continue; }

        let context: Vec<String> = (0..3).map(|j| {
            let t = tokens[start + j] as usize;
            if t < vocab.len() { vocab[t].clone() } else { "?".into() }
        }).collect();

        let mind_result = WormMind::think(
            &mut model.memory, &model.lang, &mut model.attn_roles, &mut model.hdc_memory,
            tokens, start + 2, model.config.vocab_size, None, false,
        );

        print!("  {} {} {} →", context[0], context[1], context[2]);
        for (tok, score) in mind_result.predictions.iter().take(5) {
            if (*tok as usize) < vocab.len() {
                print!(" {}({:.2})", vocab[*tok as usize], score);
            }
        }

        let expected = tokens[start + 3] as usize;
        if expected < vocab.len() {
            let hit = mind_result.predictions.iter().take(5).any(|(t, _)| *t as usize == expected);
            print!("  [exp: {}] {} (think={}, conf={:.2})",
                vocab[expected],
                if hit { "✓" } else { "✗" },
                mind_result.reasoning_depth,
                mind_result.confidence,
            );
        }
        println!();
    }

    // Авто-генерация через WormMind — черви думают на каждом шаге
    println!("\n--- Auto-generation via WormMind (20 tokens) ---");
    let gen_starts = [100, 5000, 20000];
    for &gen_start in &gen_starts {
        if gen_start + 50 >= tokens.len() { continue; }

        let ctx_begin = if gen_start >= 50 { gen_start - 50 } else { 0 };
        let mut generated: Vec<u16> = tokens[ctx_begin..gen_start + 3].to_vec();
        let original_len = generated.len();

        for _step in 0..20 {
            let ctx_start = if generated.len() > 60 { generated.len() - 60 } else { 0 };
            let ctx = &generated[ctx_start..];
            if ctx.len() < 3 { break; }

            let mind_result = WormMind::think(
                &mut model.memory, &model.lang, &mut model.attn_roles, &mut model.hdc_memory,
                ctx, ctx.len() - 1, model.config.vocab_size, None, false,
            );

            // Repetition penalty: skip recent tokens + detect loops
            let gen_len = generated.len();
            let recent: std::collections::HashSet<u16> = generated.iter()
                .rev().take(15).cloned().collect();

            // Detect loop: if last N tokens repeat a pattern from earlier
            let in_loop = if gen_len >= 12 {
                let last6: Vec<u16> = generated[gen_len-6..].to_vec();
                let prev6: Vec<u16> = generated[gen_len-12..gen_len-6].to_vec();
                last6 == prev6
            } else { false };

            let next = if in_loop {
                // В цикле — выбрать НАИМЕНЕЕ частый кандидат (разрыв цикла)
                mind_result.predictions.iter()
                    .find(|(tok, _)| !recent.contains(tok))
                    .or_else(|| mind_result.predictions.last()) // наименее вероятный
            } else {
                mind_result.predictions.iter()
                    .find(|(tok, _)| !recent.contains(tok))
                    .or_else(|| mind_result.predictions.first())
            };

            if let Some((next_tok, _score)) = next {
                generated.push(*next_tok);
            } else {
                break;
            }
        }

        // Показываем только сгенерированную часть (после контекста)
        let seed: Vec<String> = generated[original_len - 3..original_len].iter().map(|&t| {
            if (t as usize) < vocab.len() { vocab[t as usize].clone() } else { "?".into() }
        }).collect();
        let gen_part: Vec<String> = generated[original_len..].iter().map(|&t| {
            if (t as usize) < vocab.len() { vocab[t as usize].clone() } else { "?".into() }
        }).collect();
        println!("  [{}] → {}", seed.join(" "), gen_part.join(" "));
    }
}

// ============================================================
// ============================================================
// Pipeline Training — v19 Binary Pipeline (sequence processing + Hebbian)
// ============================================================

fn cmd_train_pipeline(args: &[String]) {
    use pipeline::BinaryPipeline;

    let data_path = parse_arg(args, "--data")
        .expect("--data <path> required");
    let vocab_path = parse_arg(args, "--vocab")
        .expect("--vocab <path> required");
    let sample_rate: usize = parse_arg_or(args, "--sample-rate", "10").parse().unwrap();
    let n_blocks: usize = parse_arg_or(args, "--blocks", "2").parse().unwrap();
    let inner_dim: usize = parse_arg_or(args, "--inner-dim", "128").parse().unwrap();
    let n_passes: usize = parse_arg_or(args, "--passes", "3").parse().unwrap();
    let chunk_size: usize = parse_arg_or(args, "--chunk", "512").parse().unwrap();
    let dim: usize = parse_arg_or(args, "--dim", "4096").parse().unwrap();

    println!("============================================================");
    println!("Binary Pipeline Training (v19)");
    println!("============================================================");
    println!("dim={}, blocks={}, inner_dim={}, chunk={}, sample_rate={}, passes={}",
        dim, n_blocks, inner_dim, chunk_size, sample_rate, n_passes);

    let tokens = load_tokens(&data_path);
    let vocab = load_vocab(&vocab_path);
    let vocab_size = vocab.len();

    let start = Instant::now();

    // Создать pipeline
    let mut pipeline = BinaryPipeline::new(dim, vocab_size, n_blocks, inner_dim);

    // Загрузить semantic codebook если есть
    let codebook_path = format!("codebook_{}_{}.bin", vocab_size, dim);
    {
        use crate::memory::HierarchicalMemory;
        let mut mem = HierarchicalMemory::new(dim, vocab_size);
        if mem.load_codebook(&codebook_path) {
            println!("Codebook loaded from {}", codebook_path);
            pipeline.load_codebook(&mem.words);
        } else {
            println!("Building semantic codebook...");
            mem.build_semantic_codebook(&tokens, 3);
            mem.save_codebook(&codebook_path).ok();
            pipeline.load_codebook(&mem.words);
            println!("Codebook built and saved.");
        }
    }

    // Evaluate before training
    let (c1, c5, tested) = pipeline.evaluate(&tokens, 5000.min(tokens.len() / sample_rate));
    let t = tested.max(1) as f64;
    println!("Before training: Top-1={:.2}%, Top-5={:.2}% ({} positions)",
        c1 as f64 / t * 100.0, c5 as f64 / t * 100.0, tested);

    // Training: process data in chunks
    for pass in 0..n_passes {
        let mut total_correct = 0usize;
        let mut total_total = 0usize;

        let n_chunks = (tokens.len() / chunk_size).max(1);
        // sample_rate применяется к chunks: обрабатываем каждый sample_rate-й chunk
        let chunk_step = sample_rate.max(1);
        for chunk_i in (0..n_chunks).step_by(chunk_step) {
            let chunk_start = chunk_i * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(tokens.len());
            if chunk_end - chunk_start < 4 { continue; }

            let chunk = &tokens[chunk_start..chunk_end];
            let (correct, total) = pipeline.train_chunk(chunk, 1);
            total_correct += correct;
            total_total += total;

            // Progress report
            let _effective_chunks = n_chunks / chunk_step;
            if chunk_i % ((n_chunks / 5).max(1) * chunk_step) == 0 {
                let pct = chunk_i as f64 / n_chunks as f64 * 100.0;
                let acc = if total_total > 0 { total_correct as f64 / total_total as f64 * 100.0 } else { 0.0 };
                println!("  Pass {}/{} | {:.0}% | acc={:.2}% ({}/{})",
                    pass + 1, n_passes, pct, acc, total_correct, total_total);
            }
        }

        // Eval after each pass
        let (c1, c5, tested) = pipeline.evaluate(&tokens, 5000.min(tokens.len() / sample_rate));
        let t = tested.max(1) as f64;
        println!("Pass {}/{} done: Train acc={:.2}% | Eval Top-1={:.2}%, Top-5={:.2}% | time={:.1}s",
            pass + 1, n_passes,
            total_correct as f64 / total_total.max(1) as f64 * 100.0,
            c1 as f64 / t * 100.0, c5 as f64 / t * 100.0,
            start.elapsed().as_secs_f64());
    }

    // Generation sample
    println!("\n--- Pipeline Generation (20 tokens) ---");
    let seed_start = tokens.len() / 3;
    let seed: Vec<u16> = tokens[seed_start..seed_start + 3].to_vec();
    let mut generated = seed.clone();

    for _ in 0..20 {
        let hidden = pipeline.forward(&generated);
        let pos = hidden.len() - 1;
        let (preds, _conf) = pipeline.predict_at(&hidden[pos], 10);

        // Repetition penalty
        let recent: std::collections::HashSet<u16> = generated.iter().rev().take(10).cloned().collect();
        let next = preds.iter()
            .find(|(tok, _)| !recent.contains(tok))
            .or_else(|| preds.first());

        if let Some(&(tok, _)) = next {
            generated.push(tok);
        } else {
            break;
        }
    }

    let gen_text: Vec<String> = generated.iter()
        .map(|&t| if (t as usize) < vocab.len() { vocab[t as usize].clone() } else { "?".into() })
        .collect();
    println!("  {}", gen_text.join(""));

    println!("\n============================================================");
    println!("Pipeline Training Complete ({:.1}s)", start.elapsed().as_secs_f64());
    println!("============================================================");
}

// Eval Command
// ============================================================

fn cmd_eval(args: &[String]) {
    let data_path = parse_arg(args, "--data")
        .expect("--data <path> required");
    let vocab_path = parse_arg(args, "--vocab")
        .expect("--vocab <path> required");
    let n_eval: usize = parse_arg_or(args, "--n", "10000").parse().unwrap();

    println!("Eval mode...");
    let tokens = load_tokens(&data_path);
    let vocab = load_vocab(&vocab_path);

    let config = Config::default_with_vocab(vocab.len());
    let mut model = HDCBrainV18::new(config);

    model.learn_codebook(&tokens);
    model.train(&tokens, Some(&vocab), None);

    let result = model.evaluate_wm(&tokens, n_eval);
    result.print();
}

// ============================================================
// Self-Test
// ============================================================

fn cmd_test() {
    println!("HDC-Brain v18: Self-Test");
    println!("{}", "=".repeat(60));

    let dim = 4096;

    // Test 1: Binary operations
    println!("\n[1] Binary operations...");
    {
        use binary::*;
        let mut rng = {
            use rand::SeedableRng;
            rand::rngs::StdRng::seed_from_u64(42)
        };
        let a = BinaryVec::random(dim, &mut rng);
        let b = BinaryVec::random(dim, &mut rng);
        let ab = a.bind(&b);
        let recovered = ab.unbind(&a);
        let sim = recovered.similarity(&b);
        assert_eq!(sim, dim as i32);
        println!("  bind/unbind: PASS (perfect recovery, sim={})", sim);

        let neg = a.negate();
        assert_eq!(a.similarity(&neg), -(dim as i32));
        println!("  negate: PASS");

        let p = a.permute(1);
        assert!((a.similarity(&p) as f64).abs() < 200.0);
        println!("  permute: PASS (orthogonal)");
    }

    // Test 2: Logic language
    println!("\n[2] Logic language...");
    {
        use language::*;
        let lang = LogicLanguage::new(dim);

        let ops = Opcode::all();
        let mut max_sim = 0i32;
        for i in 0..ops.len() {
            for j in (i+1)..ops.len() {
                let sim = lang.op(ops[i]).similarity(lang.op(ops[j])).abs();
                if sim > max_sim { max_sim = sim; }
            }
        }
        println!("  {} opcodes, max cross-similarity: {} (should be <300)", ops.len(), max_sim);
        assert!(max_sim < 300);
        println!("  opcodes orthogonal: PASS");

        let mut rng = {
            use rand::SeedableRng;
            rand::rngs::StdRng::seed_from_u64(42)
        };
        let fact = binary::BinaryVec::random(dim, &mut rng);
        let weak = lang.weak(&fact);
        assert!(lang.is_weak(&weak));
        println!("  WEAK detection: PASS");

        let ask = lang.ask(&fact);
        assert!(lang.is_ask(&ask));
        println!("  ASK detection: PASS");

        let lang2 = LogicLanguage::new(dim);
        assert_eq!(lang.op(Opcode::If).similarity(lang2.op(Opcode::If)), dim as i32);
        println!("  deterministic language: PASS");
    }

    // Test 3: Hierarchical memory
    println!("\n[3] Hierarchical memory...");
    {
        use memory::*;
        let mem = HierarchicalMemory::new(dim, 100);

        let mut facts = MemoryLevel::new("test", dim, 12);
        let trigram = mem.make_trigram_query(10, 20, 30);
        let mut entry = MemoryEntry::with_context(trigram.clone(), 0, vec![10, 20, 30]);
        entry.add_successor(42);
        entry.add_successor(42);
        entry.add_successor(7);
        facts.store(entry);
        let result = facts.find_exact(&trigram);
        assert!(result.is_some());
        let (idx, _) = result.unwrap();
        let top = facts.entries[idx].top1().unwrap();
        assert_eq!(top.0, 42);
        println!("  store + find + successor: PASS (top={}, count={})", top.0, top.1);

        assert!(facts.find_matching(&trigram).is_some());
        println!("  find_matching: PASS");
    }

    // Test 4: Worm-driven training (tiny)
    println!("\n[4] Worm training (tiny)...");
    {
        use worms::*;

        let mut memory = memory::HierarchicalMemory::new(256, 100);
        let lang = language::LogicLanguage::new(256);

        let data: Vec<u16> = (0..500).map(|i| (i % 100) as u16).collect();

        // Collector
        let col_out = CollectorWorm::think(&mut memory, &lang, &[], &data, 1);
        println!("  Collector: {} stored, {} strengthened, {} thoughts",
            col_out.stored, col_out.strengthened, col_out.thoughts.len());
        assert!(col_out.stored > 0);

        // Analyst reads Collector's thoughts
        let analyst_out = AnalystWorm::think(
            &mut memory, &lang, &col_out.thoughts, 100, None,
        );
        println!("  Analyst: {} scanned, {} strengthened, {} thoughts",
            analyst_out.scanned, analyst_out.strengthened, analyst_out.thoughts.len());

        // Skeptic reads Analyst's thoughts
        let mut all_thoughts = col_out.thoughts;
        all_thoughts.extend(analyst_out.thoughts);
        let skeptic_out = SkepticWorm::think(
            &mut memory, &lang, &all_thoughts, 100, None,
        );
        println!("  Skeptic: {} scanned, {} strengthened, {} asked",
            skeptic_out.scanned, skeptic_out.strengthened, skeptic_out.asked);

        // Explorer
        all_thoughts.extend(skeptic_out.thoughts);
        let explorer_out = ExplorerWorm::think(
            &mut memory, &lang, &all_thoughts, 100, None,
        );
        println!("  Explorer: {} stored (abstractions), {} asked",
            explorer_out.stored, explorer_out.asked);

        println!("  {}", memory.stats());
        println!("  Worm training: PASS");
    }

    // Test 5: Full model (tiny)
    println!("\n[5] Full model (tiny)...");
    {
        let mut model = HDCBrainV18::new(Config {
            hdc_dim: 256,
            vocab_size: 100,
            codebook_window: 2,
            chunk_size: 500,
            sample_rate: 1,
            worm_checks: 50,
            n_passes: 1,
        });

        let data: Vec<u16> = (0..1000).map(|i| (i % 100) as u16).collect();

        model.train(&data, None, None);

        let predictions = model.predict(data[10], data[11], data[12]);
        println!("  predictions: {} candidates", predictions.len());
        assert!(predictions.len() > 0);
        println!("  Full model: PASS");
    }

    // Test 6: Working Memory
    println!("\n[6] Working Memory...");
    {
        use working_memory::*;

        let mut mem = memory::HierarchicalMemory::new(256, 100);
        let lang = language::LogicLanguage::new(256);
        let data: Vec<u16> = (0..500).map(|i| (i % 100) as u16).collect();

        // Fill memory
        worms::CollectorWorm::think(&mut mem, &lang, &[], &data, 1);

        // Working Memory pipeline
        let result = WorkingMemory::full_pipeline(&mem, &data, 50, 100, 256);
        println!("  evidence: {}, sources: {:?}, predictions: {}",
            result.total_evidence, result.sources_used, result.predictions.len());
        println!("  verified: {}, confidence: {:.3}",
            result.verified_evidence, result.confidence);
        assert!(result.total_evidence > 0);
        println!("  Working Memory: PASS");
    }

    // Test 7: WormMind — черви думают при predict
    println!("\n[7] WormMind (thinking worms)...");
    {
        use worm_mind::*;

        let mut mem = memory::HierarchicalMemory::new(256, 100);
        let lang = language::LogicLanguage::new(256);
        let data: Vec<u16> = (0..500).map(|i| (i % 100) as u16).collect();

        // Fill memory
        worms::CollectorWorm::think(&mut mem, &lang, &[], &data, 1);

        // WormMind thinks
        let mut roles = AttentionRoles::new(256);
        let mut hdc_mem = HDCMemory::new(256);
        let result = WormMind::think(&mut mem, &lang, &mut roles, &mut hdc_mem, &data, 50, 100, None, false);
        println!("  predictions: {}, reasoning: {} steps, thoughts: {}",
            result.predictions.len(), result.reasoning_depth, result.thoughts_exchanged);
        println!("  confidence: {:.3}, bg_updates: {}", result.confidence, result.background_updates);
        for line in &result.trace {
            println!("    {}", line);
        }
        assert!(!result.predictions.is_empty());
        assert!(result.reasoning_depth >= 3);
        println!("  WormMind: PASS");
    }

    println!("\n{}", "=".repeat(60));
    println!("ALL TESTS PASSED");
    println!("{}", "=".repeat(60));
}
