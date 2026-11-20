//! Per-edge positive and negative sample timing for the UMAP optimizer.
//!
//! This module owns schedule arithmetic only.  It does not select negative
//! vertices, draw random numbers, or apply embedding updates.  A cursor is
//! initialized with the first positive and negative due times equal to their
//! respective intervals.  That convention matches the published checkpoint
//! behavior: with a unit positive interval and a unit negative sample rate,
//! epoch one reports a positive edge but zero negative samples; the first
//! negative sample is reported at epoch two.  Doubling the negative sample
//! rate makes the negative interval one half, so one negative sample is
//! reported at epoch one.

use std::fmt;

/// Failures while constructing or advancing an [`EdgeSamplingCursor`].
#[derive(Clone, Debug, PartialEq)]
pub enum SamplingError {
    /// The positive interval must be finite and strictly positive.
    InvalidPositiveInterval { interval: f64 },
    /// The number of negative samples per positive sample must be finite and
    /// strictly positive.
    InvalidNegativeSampleRate { rate: f64 },
    /// A graph's maximum positive membership must be finite and strictly
    /// positive.
    InvalidMaximumPositiveWeight { weight: f64 },
    /// A graph edge used by this cursor must have a finite, strictly positive
    /// membership.
    InvalidEdgeWeight { weight: f64 },
    /// An edge membership cannot be greater than the graph maximum used to
    /// normalize it.
    EdgeWeightExceedsMaximum {
        edge_weight: f64,
        maximum_positive_weight: f64,
    },
    /// Dividing the positive interval by the requested rate did not produce
    /// a finite, strictly positive negative interval.
    InvalidNegativeInterval { interval: f64 },
    /// Epochs are non-negative finite values.
    InvalidEpoch { epoch: f64 },
    /// Advancing the positive cursor would overflow or make no progress.
    PositiveScheduleOverflow {
        next_positive: f64,
        positive_interval: f64,
    },
    /// The negative backlog cannot be represented as a `usize` count or its
    /// due-time advance would overflow or make no progress.
    NegativeScheduleOverflow {
        epoch: f64,
        next_negative: f64,
        negative_interval: f64,
    },
}

impl fmt::Display for SamplingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPositiveInterval { interval } => write!(
                formatter,
                "positive sample interval must be finite and positive, got {interval}"
            ),
            Self::InvalidNegativeSampleRate { rate } => write!(
                formatter,
                "negative sample rate must be finite and positive, got {rate}"
            ),
            Self::InvalidMaximumPositiveWeight { weight } => write!(
                formatter,
                "maximum positive graph weight must be finite and positive, got {weight}"
            ),
            Self::InvalidEdgeWeight { weight } => write!(
                formatter,
                "edge graph weight must be finite and positive, got {weight}"
            ),
            Self::EdgeWeightExceedsMaximum {
                edge_weight,
                maximum_positive_weight,
            } => write!(
                formatter,
                "edge graph weight {edge_weight} exceeds maximum positive weight {maximum_positive_weight}"
            ),
            Self::InvalidNegativeInterval { interval } => write!(
                formatter,
                "negative sample interval must be finite and positive, got {interval}"
            ),
            Self::InvalidEpoch { epoch } => {
                write!(
                    formatter,
                    "sampling epoch must be finite and non-negative, got {epoch}"
                )
            }
            Self::PositiveScheduleOverflow {
                next_positive,
                positive_interval,
            } => write!(
                formatter,
                "positive schedule cannot advance {next_positive} by {positive_interval}"
            ),
            Self::NegativeScheduleOverflow {
                epoch,
                next_negative,
                negative_interval,
            } => write!(
                formatter,
                "negative schedule cannot represent backlog at epoch {epoch} from {next_negative} with interval {negative_interval}"
            ),
        }
    }
}

impl std::error::Error for SamplingError {}

/// Independent positive/negative timing for one positive graph edge.
///
/// The positive interval is either supplied directly with
/// [`EdgeSamplingCursor::from_positive_interval`] or derived as
/// `maximum_positive_weight / edge_weight` with
/// [`EdgeSamplingCursor::from_graph_weights`].  The negative interval is the
/// positive interval divided by `negative_sample_rate`.
///
/// `positive_due` reports at most one positive sample per call and advances
/// its cursor by one positive interval.  `take_negative_samples` reports all
/// samples caught up at the supplied epoch using
/// `floor((epoch - next_negative) / negative_interval)` when the epoch is at
/// or after the current negative due time.  A zero result deliberately does
/// not advance the negative cursor.  This is the observed UMAP scheduling
/// convention, and is why unit-rate negatives first appear one epoch after a
/// unit-rate positive edge.
///
/// The cursor stores no random-number generator and performs no embedding
/// arithmetic, so callers can keep negative vertex selection and update
/// precision in separate components.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeSamplingCursor {
    positive_interval: f64,
    negative_interval: f64,
    next_positive: f64,
    next_negative: f64,
}

impl EdgeSamplingCursor {
    /// Construct a cursor from the maximum positive graph weight and one edge
    /// weight.
    ///
    /// The weights are accepted as `f32` because they are the representation
    /// used by [`crate::fuzzy_graph::CsrGraph`].  Timing is then carried out
    /// in `f64` so fractional intervals do not depend on embedding precision.
    pub fn from_graph_weights(
        maximum_positive_weight: f32,
        edge_weight: f32,
        negative_sample_rate: f64,
    ) -> Result<Self, SamplingError> {
        let maximum_positive_weight = f64::from(maximum_positive_weight);
        let edge_weight = f64::from(edge_weight);
        validate_graph_weight(maximum_positive_weight, true)?;
        validate_graph_weight(edge_weight, false)?;
        if edge_weight > maximum_positive_weight {
            return Err(SamplingError::EdgeWeightExceedsMaximum {
                edge_weight,
                maximum_positive_weight,
            });
        }

        let positive_interval = maximum_positive_weight / edge_weight;
        Self::from_positive_interval(positive_interval, negative_sample_rate)
    }

    /// Construct a cursor from an explicit positive interval and the number
    /// of negative samples per positive sample.
    pub fn from_positive_interval(
        positive_interval: f64,
        negative_sample_rate: f64,
    ) -> Result<Self, SamplingError> {
        if !positive_interval.is_finite() || positive_interval <= 0.0 {
            return Err(SamplingError::InvalidPositiveInterval {
                interval: positive_interval,
            });
        }
        if !negative_sample_rate.is_finite() || negative_sample_rate <= 0.0 {
            return Err(SamplingError::InvalidNegativeSampleRate {
                rate: negative_sample_rate,
            });
        }

        let negative_interval = positive_interval / negative_sample_rate;
        if !negative_interval.is_finite() || negative_interval <= 0.0 {
            return Err(SamplingError::InvalidNegativeInterval {
                interval: negative_interval,
            });
        }

        Ok(Self {
            positive_interval,
            negative_interval,
            next_positive: positive_interval,
            next_negative: negative_interval,
        })
    }

    /// Return the positive interval used by this cursor.
    #[must_use]
    pub const fn positive_interval(&self) -> f64 {
        self.positive_interval
    }

    /// Return the negative interval used by this cursor.
    #[must_use]
    pub const fn negative_interval(&self) -> f64 {
        self.negative_interval
    }

    /// Return the next positive due time.
    #[must_use]
    pub const fn next_positive_epoch(&self) -> f64 {
        self.next_positive
    }

    /// Return the next negative due time.
    #[must_use]
    pub const fn next_negative_epoch(&self) -> f64 {
        self.next_negative
    }

    /// Report whether one positive update is due at `epoch`.
    ///
    /// At most one positive update is consumed per call.  This preserves the
    /// one-edge-per-epoch behavior of the UMAP optimizer when an edge's
    /// interval is shorter than one epoch.
    pub fn positive_due(&mut self, epoch: f64) -> Result<bool, SamplingError> {
        validate_epoch(epoch)?;
        if epoch < self.next_positive {
            return Ok(false);
        }

        let next_positive = self.next_positive + self.positive_interval;
        if !next_positive.is_finite() || next_positive <= self.next_positive {
            return Err(SamplingError::PositiveScheduleOverflow {
                next_positive: self.next_positive,
                positive_interval: self.positive_interval,
            });
        }
        self.next_positive = next_positive;
        Ok(true)
    }

    /// Return and consume the negative samples due by `epoch`.
    ///
    /// If `epoch` precedes the current due time, or exactly equals it, this
    /// returns zero and leaves the cursor unchanged.  A positive count is
    /// advanced in one checked multiplication/addition so an overflowing or
    /// non-progressing update cannot leave an infinite/invalid backlog.
    pub fn take_negative_samples(&mut self, epoch: f64) -> Result<usize, SamplingError> {
        validate_epoch(epoch)?;
        if epoch < self.next_negative {
            return Ok(0);
        }

        let count_as_f64 = ((epoch - self.next_negative) / self.negative_interval).floor();
        if !count_as_f64.is_finite() || count_as_f64 < 0.0 || exceeds_usize(count_as_f64) {
            return Err(SamplingError::NegativeScheduleOverflow {
                epoch,
                next_negative: self.next_negative,
                negative_interval: self.negative_interval,
            });
        }

        let count = count_as_f64 as usize;
        if count == 0 {
            return Ok(0);
        }

        let advance = self.negative_interval * count_as_f64;
        let next_negative = self.next_negative + advance;
        if !advance.is_finite() || !next_negative.is_finite() || next_negative <= self.next_negative
        {
            return Err(SamplingError::NegativeScheduleOverflow {
                epoch,
                next_negative: self.next_negative,
                negative_interval: self.negative_interval,
            });
        }

        self.next_negative = next_negative;
        Ok(count)
    }
}

fn validate_graph_weight(weight: f64, maximum: bool) -> Result<(), SamplingError> {
    if !weight.is_finite() || weight <= 0.0 {
        if maximum {
            Err(SamplingError::InvalidMaximumPositiveWeight { weight })
        } else {
            Err(SamplingError::InvalidEdgeWeight { weight })
        }
    } else {
        Ok(())
    }
}

fn validate_epoch(epoch: f64) -> Result<(), SamplingError> {
    if !epoch.is_finite() || epoch < 0.0 {
        Err(SamplingError::InvalidEpoch { epoch })
    } else {
        Ok(())
    }
}

fn exceeds_usize(value: f64) -> bool {
    let maximum = usize::MAX as f64;
    // On 64-bit targets, converting usize::MAX to f64 rounds to 2^64, which
    // cannot be represented as a usize.  The strict comparison on that target
    // prevents Rust's saturating float-to-int cast from hiding the overflow.
    if usize::BITS == 64 {
        value >= maximum
    } else {
        value > maximum
    }
}

#[cfg(test)]
mod tests {
    use super::{EdgeSamplingCursor, SamplingError};

    #[test]
    fn published_unit_interval_checkpoints_match() {
        let mut rate_one = EdgeSamplingCursor::from_positive_interval(1.0, 1.0)
            .expect("unit schedule must be valid");
        assert!(!rate_one.positive_due(0.0).unwrap());
        assert_eq!(rate_one.take_negative_samples(0.0).unwrap(), 0);
        assert!(rate_one.positive_due(1.0).unwrap());
        assert_eq!(rate_one.take_negative_samples(1.0).unwrap(), 0);
        assert!(rate_one.positive_due(2.0).unwrap());
        assert_eq!(rate_one.take_negative_samples(2.0).unwrap(), 1);

        let mut rate_two = EdgeSamplingCursor::from_positive_interval(1.0, 2.0)
            .expect("double-rate schedule must be valid");
        assert!(rate_two.positive_due(1.0).unwrap());
        assert_eq!(rate_two.take_negative_samples(1.0).unwrap(), 1);
    }

    #[test]
    fn delayed_and_fractional_due_times_are_exactly_consumed() {
        let mut cursor = EdgeSamplingCursor::from_positive_interval(1.5, 2.0)
            .expect("fractional schedule must be valid");
        assert_eq!(cursor.positive_interval(), 1.5);
        assert_eq!(cursor.negative_interval(), 0.75);
        assert!(!cursor.positive_due(1.499_999).unwrap());
        assert!(!cursor.positive_due(0.5).unwrap());
        assert_eq!(cursor.take_negative_samples(0.75).unwrap(), 0);
        assert_eq!(cursor.take_negative_samples(1.5).unwrap(), 1);
        assert!(cursor.positive_due(1.5).unwrap());
        assert!(!cursor.positive_due(1.5).unwrap());
        assert_eq!(cursor.next_positive_epoch(), 3.0);
        assert_eq!(cursor.next_negative_epoch(), 1.5);
    }

    #[test]
    fn graph_weights_derive_the_positive_interval() {
        let cursor = EdgeSamplingCursor::from_graph_weights(0.75, 0.25, 2.0)
            .expect("valid graph weights must construct a cursor");
        assert_eq!(cursor.positive_interval(), 3.0);
        assert_eq!(cursor.negative_interval(), 1.5);
        assert_eq!(cursor.next_positive_epoch(), 3.0);
        assert_eq!(cursor.next_negative_epoch(), 1.5);
    }

    #[test]
    fn construction_rejects_invalid_intervals_rates_and_weights() {
        assert!(matches!(
            EdgeSamplingCursor::from_positive_interval(0.0, 1.0),
            Err(SamplingError::InvalidPositiveInterval { .. })
        ));
        assert!(matches!(
            EdgeSamplingCursor::from_positive_interval(f64::NAN, 1.0),
            Err(SamplingError::InvalidPositiveInterval { .. })
        ));
        assert!(matches!(
            EdgeSamplingCursor::from_positive_interval(1.0, 0.0),
            Err(SamplingError::InvalidNegativeSampleRate { .. })
        ));
        assert!(matches!(
            EdgeSamplingCursor::from_positive_interval(f64::MIN_POSITIVE, f64::MAX),
            Err(SamplingError::InvalidNegativeInterval { .. })
        ));
        assert!(matches!(
            EdgeSamplingCursor::from_graph_weights(0.0, 0.25, 1.0),
            Err(SamplingError::InvalidMaximumPositiveWeight { .. })
        ));
        assert!(matches!(
            EdgeSamplingCursor::from_graph_weights(1.0, 0.0, 1.0),
            Err(SamplingError::InvalidEdgeWeight { .. })
        ));
        assert!(matches!(
            EdgeSamplingCursor::from_graph_weights(0.25, 0.5, 1.0),
            Err(SamplingError::EdgeWeightExceedsMaximum { .. })
        ));
    }

    #[test]
    fn epoch_and_backlog_validation_preserve_cursor_state() {
        let mut cursor = EdgeSamplingCursor::from_positive_interval(1.0, 1.0)
            .expect("unit schedule must be valid");
        assert!(matches!(
            cursor.positive_due(-1.0),
            Err(SamplingError::InvalidEpoch { .. })
        ));
        assert!(matches!(
            cursor.take_negative_samples(f64::INFINITY),
            Err(SamplingError::InvalidEpoch { .. })
        ));

        let before = cursor.next_negative_epoch();
        assert!(matches!(
            cursor.take_negative_samples(f64::MAX),
            Err(SamplingError::NegativeScheduleOverflow { .. })
        ));
        assert_eq!(cursor.next_negative_epoch(), before);

        let mut huge_interval = EdgeSamplingCursor::from_positive_interval(f64::MAX, 1.0)
            .expect("finite maximum interval is valid initially");
        let before = huge_interval.next_positive_epoch();
        assert!(matches!(
            huge_interval.positive_due(f64::MAX),
            Err(SamplingError::PositiveScheduleOverflow { .. })
        ));
        assert_eq!(huge_interval.next_positive_epoch(), before);
    }
}
