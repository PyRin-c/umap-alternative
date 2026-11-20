//! Deterministic MT19937 draws used by seeded UMAP initialization.
//!
//! This is the original 32-bit Mersenne Twister recurrence (period
//! `2^19937 - 1`) with the reference integer seeding procedure.  It is kept
//! deliberately small: callers get the raw `u32` stream and the 53-bit
//! double-precision construction used by NumPy's legacy `RandomState`, but no
//! distribution or range helpers whose rounding would need a separate
//! compatibility contract.

const STATE_WORDS: usize = 624;
const PERIOD_OFFSET: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;
const TEMPER_B: u32 = 0x9d2c_5680;
const TEMPER_C: u32 = 0xefc6_0000;
const SEED_MULTIPLIER: u32 = 1_812_433_253;

/// The 32-bit MT19937 pseudorandom number generator.
///
/// `new` uses the reference `init_genrand` procedure for one `u32` seed.
/// Each call to [`Mt19937::next_u32`] advances the generator by one tempered
/// word.  [`Mt19937::next_f64`] consumes two words and joins their high bits
/// into an exactly 53-bit integer, yielding a value in `[0, 1)`.
#[derive(Clone, Debug)]
pub struct Mt19937 {
    state: [u32; STATE_WORDS],
    index: usize,
}

impl Mt19937 {
    /// Construct a generator with the reference MT19937 integer seed.
    #[must_use]
    pub fn new(seed: u32) -> Self {
        let mut state = [0; STATE_WORDS];
        state[0] = seed;
        for index in 1..STATE_WORDS {
            let previous = state[index - 1];
            state[index] = SEED_MULTIPLIER
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(index as u32);
        }

        Self {
            state,
            // A freshly initialized state is twisted before its first draw.
            index: STATE_WORDS,
        }
    }

    /// Return the next tempered 32-bit MT19937 word.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= STATE_WORDS {
            self.twist();
        }

        let mut value = self.state[self.index];
        self.index += 1;

        // MT19937 tempering, as specified by the reference recurrence.
        value ^= value >> 11;
        value ^= (value << 7) & TEMPER_B;
        value ^= (value << 15) & TEMPER_C;
        value ^= value >> 18;
        value
    }

    /// Return the next NumPy-legacy-compatible uniform `f64` in `[0, 1)`.
    ///
    /// Two successive words contribute 27 and 26 high bits respectively;
    /// their concatenation is divided by `2^53`.  This is the standard
    /// `RandomState.random_sample` construction and is intentionally not the
    /// one-word `u32 / 2^32` conversion.
    pub fn next_f64(&mut self) -> f64 {
        let upper = (self.next_u32() >> 5) as u64;
        let lower = (self.next_u32() >> 6) as u64;
        let mantissa = (upper << 26) | lower;
        mantissa as f64 / 9_007_199_254_740_992.0
    }

    fn twist(&mut self) {
        for index in 0..STATE_WORDS {
            let joined = (self.state[index] & UPPER_MASK)
                | (self.state[(index + 1) % STATE_WORDS] & LOWER_MASK);
            let mut value = self.state[(index + PERIOD_OFFSET) % STATE_WORDS] ^ (joined >> 1);
            if joined & 1 != 0 {
                value ^= MATRIX_A;
            }
            self.state[index] = value;
        }
        self.index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::Mt19937;

    #[test]
    fn matches_published_reference_sequence_for_seed_5489() {
        let mut generator = Mt19937::new(5489);
        let expected = [
            3_499_211_612,
            581_869_302,
            3_890_346_734,
            3_586_334_585,
            545_404_204,
            4_161_255_391,
            3_922_919_429,
            949_333_985,
            2_715_962_298,
            1_323_567_403,
        ];

        for expected in expected {
            assert_eq!(generator.next_u32(), expected);
        }
    }

    #[test]
    fn matches_53_bit_uniform_construction() {
        let mut generator = Mt19937::new(5489);
        let expected_bits = [
            0x3fea_1237_688a_ba7b,
            0x3fec_fc3f_5f57_0c7d,
            0x3fc0_411a_9f80_7b7c,
        ];

        for expected in expected_bits {
            let value = generator.next_f64();
            assert!((0.0..1.0).contains(&value));
            assert_eq!(value.to_bits(), expected);
        }
    }

    #[test]
    fn twist_continues_the_stream_after_one_state() {
        let mut generator = Mt19937::new(5489);
        for _ in 0..624 {
            let _ = generator.next_u32();
        }

        // The 625th word is the first word after the reference twist.
        assert_eq!(generator.next_u32(), 4_178_893_912);
    }
}
