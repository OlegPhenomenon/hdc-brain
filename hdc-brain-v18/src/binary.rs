//! HDC Binary Vector Operations — XNOR/POPCNT packed u64
//!
//! Логические интерпретации:
//!   bind(A, B)   = XOR  → ассоциация / AND
//!   unbind(A, B)  = XOR  → извлечение (self-inverse)
//!   bundle(vecs)  = MAJ  → множество / OR
//!   permute(A, k) = ROT  → позиция / роль
//!   negate(A)     = NOT  → логическое отрицание
//!   similarity    = XNOR+POPCNT → истина/ложь

use rand::Rng;

pub const BITS_PER_WORD: usize = 64;

#[derive(Clone)]
pub struct BinaryVec {
    pub data: Vec<u64>,
    pub dim: usize,
}

impl BinaryVec {
    pub fn zeros(dim: usize) -> Self {
        let words = (dim + BITS_PER_WORD - 1) / BITS_PER_WORD;
        BinaryVec { data: vec![0u64; words], dim }
    }

    pub fn random(dim: usize, rng: &mut impl Rng) -> Self {
        let words = (dim + BITS_PER_WORD - 1) / BITS_PER_WORD;
        let data: Vec<u64> = (0..words).map(|_| rng.next_u64()).collect();
        BinaryVec { data, dim }
    }

    #[inline]
    pub fn words(&self) -> usize { self.data.len() }

    /// Hamming similarity: 2 * matching_bits - D. Range [-D, D].
    #[inline]
    pub fn similarity(&self, other: &BinaryVec) -> i32 {
        debug_assert_eq!(self.dim, other.dim);
        let matching: u32 = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| (!(a ^ b)).count_ones())
            .sum();
        2 * matching as i32 - self.dim as i32
    }

    /// Bind (XOR): ассоциация. bind("кот", "мяукает") = факт.
    /// Self-inverse: unbind(bind(A,B), A) = B.
    #[inline]
    pub fn bind(&self, other: &BinaryVec) -> BinaryVec {
        debug_assert_eq!(self.dim, other.dim);
        BinaryVec {
            data: self.data.iter().zip(other.data.iter())
                .map(|(&a, &b)| a ^ b).collect(),
            dim: self.dim,
        }
    }

    /// Unbind = bind (self-inverse in binary/bipolar).
    #[inline]
    pub fn unbind(&self, key: &BinaryVec) -> BinaryVec {
        self.bind(key)
    }

    /// Negate: flip all bits. NOT A.
    #[inline]
    pub fn negate(&self) -> BinaryVec {
        BinaryVec {
            data: self.data.iter().map(|&w| !w).collect(),
            dim: self.dim,
        }
    }

    /// Cyclic permute by `shift` bits. Encodes position/role.
    /// permute(x, 1) ≈ orthogonal to x.
    pub fn permute(&self, shift: usize) -> BinaryVec {
        let shift = shift % self.dim;
        if shift == 0 { return self.clone(); }

        let n_words = self.data.len();
        let word_shift = shift / BITS_PER_WORD;
        let bit_shift = shift % BITS_PER_WORD;

        let mut result = vec![0u64; n_words];

        if bit_shift == 0 {
            for i in 0..n_words {
                result[i] = self.data[(i + n_words - word_shift) % n_words];
            }
        } else {
            for i in 0..n_words {
                let src1 = (i + n_words - word_shift) % n_words;
                let src2 = (i + n_words - word_shift + n_words - 1) % n_words;
                result[i] = (self.data[src1] << bit_shift)
                          | (self.data[src2] >> (BITS_PER_WORD - bit_shift));
            }
        }

        BinaryVec { data: result, dim: self.dim }
    }

    #[inline]
    pub fn get_bit(&self, pos: usize) -> bool {
        (self.data[pos / BITS_PER_WORD] >> (pos % BITS_PER_WORD)) & 1 == 1
    }

    #[inline]
    pub fn set_bit(&mut self, pos: usize, val: bool) {
        let word = pos / BITS_PER_WORD;
        let bit = pos % BITS_PER_WORD;
        if val {
            self.data[word] |= 1u64 << bit;
        } else {
            self.data[word] &= !(1u64 << bit);
        }
    }
}

/// LSH hash: first n_bits of the vector → bucket index.
#[inline]
pub fn lsh_hash(vec: &BinaryVec, n_bits: usize) -> u32 {
    let mut hash = 0u32;
    let words_needed = (n_bits + BITS_PER_WORD - 1) / BITS_PER_WORD;
    for w in 0..words_needed.min(vec.words()) {
        let bits_from_word = if (w + 1) * BITS_PER_WORD <= n_bits {
            BITS_PER_WORD
        } else {
            n_bits - w * BITS_PER_WORD
        };
        let mask = if bits_from_word >= 64 { u64::MAX } else { (1u64 << bits_from_word) - 1 };
        hash |= ((vec.data[w] & mask) as u32) << (w * BITS_PER_WORD);
    }
    hash
}

/// Bundle accumulator: incremental majority vote with i32 counters.
pub struct BundleAccumulator {
    pub counters: Vec<i32>,
    pub dim: usize,
    pub count: u32,
}

impl BundleAccumulator {
    pub fn new(dim: usize) -> Self {
        BundleAccumulator { counters: vec![0i32; dim], dim, count: 0 }
    }

    /// Add a vector: +1 for set bits, -1 for clear bits.
    /// Optimized: processes 64 bits per word using bit extraction.
    #[inline]
    pub fn add(&mut self, vec: &BinaryVec) {
        debug_assert_eq!(self.dim, vec.dim);
        let n_full_words = self.dim / BITS_PER_WORD;
        let remainder = self.dim % BITS_PER_WORD;

        for w in 0..n_full_words {
            let word = vec.data[w];
            let base = w * BITS_PER_WORD;
            // Branchless: bit is 0 or 1, map to -1 or +1 via (bit*2 - 1)
            let counters = &mut self.counters[base..base + BITS_PER_WORD];
            let mut bits = word;
            for c in counters.iter_mut() {
                *c += ((bits & 1) as i32) * 2 - 1;
                bits >>= 1;
            }
        }

        // Handle remaining bits
        if remainder > 0 {
            let word = vec.data[n_full_words];
            let base = n_full_words * BITS_PER_WORD;
            let mut bits = word;
            for i in 0..remainder {
                self.counters[base + i] += ((bits & 1) as i32) * 2 - 1;
                bits >>= 1;
            }
        }

        self.count += 1;
    }

    /// Collapse to binary: majority vote.
    /// Optimized: builds words directly instead of per-bit set_bit().
    pub fn to_binary(&self) -> BinaryVec {
        let n_words = (self.dim + BITS_PER_WORD - 1) / BITS_PER_WORD;
        let mut data = vec![0u64; n_words];

        for w in 0..n_words {
            let base = w * BITS_PER_WORD;
            let mut word = 0u64;
            let bits_in_word = BITS_PER_WORD.min(self.dim - base);
            for bit in 0..bits_in_word {
                if self.counters[base + bit] > 0 {
                    word |= 1u64 << bit;
                }
            }
            data[w] = word;
        }

        BinaryVec { data, dim: self.dim }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_bind_unbind() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = BinaryVec::random(4096, &mut rng);
        let b = BinaryVec::random(4096, &mut rng);

        let ab = a.bind(&b);
        let recovered = ab.unbind(&a);
        assert_eq!(recovered.similarity(&b), b.dim as i32); // perfect recovery
    }

    #[test]
    fn test_negate() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = BinaryVec::random(4096, &mut rng);
        let neg_a = a.negate();
        assert_eq!(a.similarity(&neg_a), -(a.dim as i32)); // perfect negation
    }

    #[test]
    fn test_permute_orthogonal() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = BinaryVec::random(4096, &mut rng);
        let p1 = a.permute(1);
        let sim = a.similarity(&p1);
        // Random vectors: expected similarity ≈ 0 (±√D)
        assert!((sim as f64).abs() < 200.0, "sim = {}", sim);
    }

    #[test]
    fn test_bundle_accumulator() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = BinaryVec::random(256, &mut rng);
        let b = BinaryVec::random(256, &mut rng);
        let c = BinaryVec::random(256, &mut rng);

        let mut acc = BundleAccumulator::new(256);
        acc.add(&a);
        acc.add(&a); // add a twice
        acc.add(&b);
        let result = acc.to_binary();

        // Result should be more similar to a (added twice) than to b or c
        assert!(result.similarity(&a) > result.similarity(&c));
    }
}
