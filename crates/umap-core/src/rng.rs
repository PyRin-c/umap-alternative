//! Deterministic RNGs for seeded initialization and serial negative sampling.
//!
//! This is the original 32-bit Mersenne Twister recurrence (period
//! `2^19937 - 1`) with the reference integer seeding procedure.  It is kept
//! deliberately small: callers get the raw `u32` stream and the 53-bit
//! double-precision construction used by NumPy's legacy `RandomState`.
//! The serial layout sampler seeds its per-head streams from the same caller
//! RNG after initialization.

use std::fmt;

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

/// An invalid MT19937 state snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mt19937StateError {
    /// The word array does not contain exactly 624 values.
    InvalidLength { actual: usize },
    /// The cursor is outside the inclusive range `0..=624`.
    InvalidCursor { cursor: usize },
}

impl fmt::Display for Mt19937StateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { actual } => {
                write!(
                    formatter,
                    "MT19937 state has {actual} words, expected {STATE_WORDS}"
                )
            }
            Self::InvalidCursor { cursor } => {
                write!(
                    formatter,
                    "MT19937 cursor {cursor} is outside 0..={STATE_WORDS}"
                )
            }
        }
    }
}

impl std::error::Error for Mt19937StateError {}

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

    /// Restore the word array and cursor from a stream snapshot.
    ///
    /// These are the `keys` and `pos` fields in NumPy's legacy MT19937
    /// `RandomState.get_state()` tuple. The array must contain 624 words, and
    /// the cursor must be in `0..=624`. Cursor zero reads the first word in the
    /// supplied block. Cursor 624 twists the block before the next draw.
    /// The tuple's Gaussian cache belongs to `RandomState` and is independent
    /// of the integer and uniform draws supported here.
    pub fn from_state(words: &[u32], cursor: usize) -> Result<Self, Mt19937StateError> {
        let state = words
            .try_into()
            .map_err(|_| Mt19937StateError::InvalidLength {
                actual: words.len(),
            })?;
        if cursor > STATE_WORDS {
            return Err(Mt19937StateError::InvalidCursor { cursor });
        }
        Ok(Self {
            state,
            index: cursor,
        })
    }

    /// Copy the current word array and cursor for a later [`Self::from_state`] call.
    #[must_use]
    pub fn state(&self) -> ([u32; STATE_WORDS], usize) {
        (self.state, self.index)
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

    /// Draw a standard normal using the polar Box–Muller transform.
    ///
    /// Start with an empty cache and carry it across consecutive draws to
    /// preserve the legacy `RandomState` Gaussian stream.
    pub(crate) fn next_standard_normal(&mut self, cache: &mut Option<f64>) -> f64 {
        if let Some(spare) = cache.take() {
            return spare;
        }
        loop {
            let x = 2.0 * self.next_f64() - 1.0;
            let y = 2.0 * self.next_f64() - 1.0;
            let squared_radius = x * x + y * y;
            if squared_radius > 0.0 && squared_radius < 1.0 {
                let factor = (-2.0 * squared_radius.ln() / squared_radius).sqrt();
                *cache = Some(factor * x);
                return factor * y;
            }
        }
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

/// Invalid initial layout for a serial negative sampler.
#[derive(Clone, Debug, PartialEq)]
pub enum NegativeSamplerError {
    /// The dimensions are empty or do not describe the supplied layout.
    InvalidShape {
        /// Number of vertices.
        n_rows: usize,
        /// Number of coordinates per vertex.
        n_components: usize,
        /// Number of row-major values supplied.
        values: usize,
    },
    /// A head's first coordinate cannot seed its stream.
    NonFiniteHeadCoordinate {
        /// Vertex whose first coordinate is invalid.
        head: usize,
        /// Invalid coordinate.
        value: f32,
    },
}

impl fmt::Display for NegativeSamplerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape {
                n_rows,
                n_components,
                values,
            } => write!(
                formatter,
                "negative sampler layout shape {n_rows}x{n_components} is invalid for {values} values"
            ),
            Self::NonFiniteHeadCoordinate { head, value } => {
                write!(
                    formatter,
                    "negative sampler head {head} has non-finite coordinate {value}"
                )
            }
        }
    }
}

impl std::error::Error for NegativeSamplerError {}

/// Per-head negative vertex streams for the serial compatibility layout.
///
/// Each stream is seeded once from its head's first normalized coordinate and
/// the caller RNG. A draw advances only the requested head. Keep this sampler
/// for the entire optimization, including across edges and epochs.
#[derive(Clone, Debug)]
pub struct SerialNegativeSampler {
    heads: Vec<[i64; 3]>,
}

impl SerialNegativeSampler {
    /// Seed the sampler from the caller RNG and the initial row-major layout.
    ///
    /// Random initialization should consume draws from `caller_rng` before
    /// this call. For explicit initialization, pass the normalized layout with
    /// the caller RNG at its current position. An invalid shape or non-finite
    /// first coordinate is reported before drawing from the caller RNG.
    pub fn new(
        caller_rng: &mut Mt19937,
        normalized_layout: &[f32],
        n_rows: usize,
        n_components: usize,
    ) -> Result<Self, NegativeSamplerError> {
        Self::from_raw_words(normalized_layout, n_rows, n_components, || {
            caller_rng.next_u32()
        })
    }

    /// Draw a vertex from the full input layout for `head`.
    ///
    /// Returns `None` for an out-of-range head. A valid draw can select any
    /// vertex, including `head` itself.
    pub fn sample(&mut self, head: usize) -> Option<usize> {
        self.sample_from(head, self.heads.len())
    }

    // For a held-out embedding the heads and the sampled vertices belong
    // to different matrices. The caller supplies the positive tail count.
    pub(crate) fn sample_from(&mut self, head: usize, n_vertices: usize) -> Option<usize> {
        // The validated input slice held at least one f32 per vertex, so the
        // vertex count fits in i64 on Rust's supported pointer widths.
        let word = self.next_word(head)?;
        Some(i64::from(word).rem_euclid(n_vertices as i64) as usize)
    }

    fn from_raw_words(
        normalized_layout: &[f32],
        n_rows: usize,
        n_components: usize,
        next_raw_word: impl FnMut() -> u32,
    ) -> Result<Self, NegativeSamplerError> {
        if n_rows == 0
            || n_components == 0
            || n_rows.checked_mul(n_components) != Some(normalized_layout.len())
        {
            return Err(NegativeSamplerError::InvalidShape {
                n_rows,
                n_components,
                values: normalized_layout.len(),
            });
        }

        let rows = normalized_layout.chunks_exact(n_components);
        for (head, row) in rows.clone().enumerate() {
            if !row[0].is_finite() {
                return Err(NegativeSamplerError::NonFiniteHeadCoordinate {
                    head,
                    value: row[0],
                });
            }
        }

        Ok(Self::from_head_coordinates(
            rows.map(|row| row[0]),
            next_raw_word,
        ))
    }

    pub(crate) fn from_head_coordinates(
        coordinates: impl Iterator<Item = f32>,
        next_raw_word: impl FnMut() -> u32,
    ) -> Self {
        let seed = draw_negative_seed(next_raw_word);
        let heads = coordinates
            .map(|coordinate| {
                // Promote the coordinate before reinterpreting its binary64 bits.
                let coordinate_bits = f64::from(coordinate).to_bits() as i64;
                seed.map(|lane| coordinate_bits.wrapping_add(lane))
            })
            .collect();
        Self { heads }
    }

    fn next_word(&mut self, head: usize) -> Option<i32> {
        self.heads.get_mut(head).map(next_negative_word)
    }
}

pub(crate) fn draw_negative_seed(mut next_raw_word: impl FnMut() -> u32) -> [i64; 3] {
    std::array::from_fn(|_| {
        loop {
            let raw = next_raw_word();
            // NumPy's observed seed interval rejects the three largest raw words.
            if raw < 0xffff_fffd {
                return i64::from(raw) - i64::from(i32::MAX);
            }
        }
    })
}

pub(crate) fn sample_vertex(lanes: &mut [i64; 3], n_vertices: usize) -> usize {
    i64::from(next_negative_word(lanes)).rem_euclid(n_vertices as i64) as usize
}

fn next_negative_word(lanes: &mut [i64; 3]) -> i32 {
    lanes[0] = advance_lane(lanes[0], 0xffff_fffe, 13, 12, 19);
    lanes[1] = advance_lane(lanes[1], 0xffff_fff8, 2, 4, 25);
    lanes[2] = advance_lane(lanes[2], 0xffff_fff0, 3, 17, 11);
    // Interpret the low 32 bits of the XOR as a signed word.
    (lanes[0] ^ lanes[1] ^ lanes[2]) as i32
}

fn advance_lane(lane: i64, mask: u32, q: u32, s: u32, r: u32) -> i64 {
    // Both left shifts keep only their low 32 bits. The unshifted feedback
    // operand and the saved result retain their signed 64-bit width.
    let low = lane as u32;
    let advance = i64::from((low & mask) << s);
    let feedback = (i64::from(low << q) ^ lane) >> r;
    advance ^ feedback
}

#[cfg(test)]
mod negative_sampler_tests;

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
