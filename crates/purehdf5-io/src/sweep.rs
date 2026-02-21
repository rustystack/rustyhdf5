//! Sweep detection and prediction for N-dimensional chunked dataset access.
//!
//! Based on the insight from "Larger than memory image processing" (arXiv 2601.18407)
//! that 3D chunked layouts force redundant chunk access for non-aligned sweeps.
//! This module detects sweep patterns from chunk access history and predicts
//! the next chunks that will be accessed.

/// Coordinate key for a chunk — the N-dimensional offset vector.
pub type ChunkCoord = Vec<u64>;

/// Detected sweep direction across an N-dimensional chunked dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepDirection {
    /// Sweeping along the first (outermost) dimension — row-major order.
    RowMajor,
    /// Sweeping along the last (innermost) dimension — column-major order.
    ColumnMajor,
    /// Sweeping along a specific slice dimension (middle axis in 3D+).
    SliceMajor(usize),
    /// No discernible pattern — random or too few samples.
    Random,
}

/// Detect the sweep direction from a history of chunk coordinate accesses.
///
/// Examines consecutive differences in the coordinate history to determine
/// which dimension is being swept. Requires at least 3 entries to detect
/// a pattern.
///
/// - `history`: recent chunk coordinates, oldest first.
/// - `ndims`: number of dimensions in the dataset.
pub fn detect_sweep(history: &[ChunkCoord], ndims: usize) -> SweepDirection {
    if history.len() < 3 || ndims == 0 {
        return SweepDirection::Random;
    }

    // Compute deltas between consecutive accesses
    let num_deltas = history.len() - 1;
    let mut changing_dim_counts = vec![0usize; ndims];
    let mut constant_dim_counts = vec![0usize; ndims];

    for i in 0..num_deltas {
        let prev = &history[i];
        let curr = &history[i + 1];
        if prev.len() < ndims || curr.len() < ndims {
            return SweepDirection::Random;
        }

        for d in 0..ndims {
            if curr[d] != prev[d] {
                changing_dim_counts[d] += 1;
            } else {
                constant_dim_counts[d] += 1;
            }
        }
    }

    // A sweep along dimension D means:
    // - Dimension D changes frequently (the "fast" axis)
    // - Other dimensions change rarely (the "slow" axes)
    //
    // For row-major: the last dimension changes most often
    // For column-major: the first dimension changes most often
    // For slice-major: a middle dimension changes most often

    // Find the dimension that changes in the most consecutive steps
    let threshold = (num_deltas + 1) / 2; // >50% of steps must show this pattern

    // Find the single fastest-changing dimension
    let (max_dim, max_changes) = changing_dim_counts
        .iter()
        .enumerate()
        .max_by_key(|(_, &c)| c)
        .unwrap();

    if *max_changes < threshold {
        return SweepDirection::Random;
    }

    // Check that other dimensions change less frequently (at most half as often)
    let others_max = changing_dim_counts
        .iter()
        .enumerate()
        .filter(|(d, _)| *d != max_dim)
        .map(|(_, &c)| c)
        .max()
        .unwrap_or(0);

    // The fast axis should dominate
    if others_max > 0 && *max_changes < others_max * 2 {
        return SweepDirection::Random;
    }

    if max_dim == ndims - 1 {
        SweepDirection::RowMajor
    } else if max_dim == 0 {
        SweepDirection::ColumnMajor
    } else {
        SweepDirection::SliceMajor(max_dim)
    }
}

/// Predict the next `count` chunk coordinates based on the detected sweep direction.
///
/// Extrapolates from the last entry in `history` using the average step
/// observed along the sweep dimension.
pub fn predict_next(
    history: &[ChunkCoord],
    direction: SweepDirection,
    count: usize,
) -> Vec<ChunkCoord> {
    if history.len() < 2 || count == 0 {
        return Vec::new();
    }

    let ndims = history[0].len();
    let sweep_dim = match direction {
        SweepDirection::RowMajor => ndims.saturating_sub(1),
        SweepDirection::ColumnMajor => 0,
        SweepDirection::SliceMajor(d) => d.min(ndims.saturating_sub(1)),
        SweepDirection::Random => return Vec::new(),
    };

    // Compute average step along the sweep dimension from recent history
    let mut total_step: i64 = 0;
    let mut step_count: usize = 0;
    for i in 1..history.len() {
        let prev = history[i - 1][sweep_dim] as i64;
        let curr = history[i][sweep_dim] as i64;
        let diff = curr - prev;
        if diff != 0 {
            total_step += diff;
            step_count += 1;
        }
    }

    if step_count == 0 {
        return Vec::new();
    }

    let avg_step = total_step / step_count as i64;
    if avg_step == 0 {
        return Vec::new();
    }

    let last = history.last().unwrap();
    let mut predictions = Vec::with_capacity(count);

    for i in 1..=count {
        let mut coord = last.clone();
        let new_val = last[sweep_dim] as i64 + avg_step * i as i64;
        if new_val < 0 {
            break;
        }
        coord[sweep_dim] = new_val as u64;
        predictions.push(coord);
    }

    predictions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_row_major_2d() {
        // Sweeping along dim 1 (columns) — row-major order
        let history = vec![
            vec![0, 0],
            vec![0, 10],
            vec![0, 20],
            vec![0, 30],
        ];
        assert_eq!(detect_sweep(&history, 2), SweepDirection::RowMajor);
    }

    #[test]
    fn detect_column_major_2d() {
        // Sweeping along dim 0 (rows) — column-major order
        let history = vec![
            vec![0, 0],
            vec![10, 0],
            vec![20, 0],
            vec![30, 0],
        ];
        assert_eq!(detect_sweep(&history, 2), SweepDirection::ColumnMajor);
    }

    #[test]
    fn detect_row_major_3d() {
        // In 3D, row-major means the last dim (dim 2) changes fastest
        let history = vec![
            vec![0, 0, 0],
            vec![0, 0, 4],
            vec![0, 0, 8],
            vec![0, 0, 12],
        ];
        assert_eq!(detect_sweep(&history, 3), SweepDirection::RowMajor);
    }

    #[test]
    fn detect_column_major_3d() {
        // In 3D, column-major means dim 0 changes fastest
        let history = vec![
            vec![0, 0, 0],
            vec![4, 0, 0],
            vec![8, 0, 0],
            vec![12, 0, 0],
        ];
        assert_eq!(detect_sweep(&history, 3), SweepDirection::ColumnMajor);
    }

    #[test]
    fn detect_slice_major_3d() {
        // Middle dim (dim 1) changes fastest — SliceMajor(1)
        let history = vec![
            vec![0, 0, 0],
            vec![0, 4, 0],
            vec![0, 8, 0],
            vec![0, 12, 0],
        ];
        assert_eq!(detect_sweep(&history, 3), SweepDirection::SliceMajor(1));
    }

    #[test]
    fn detect_random_too_few_entries() {
        let history = vec![vec![0, 0], vec![0, 10]];
        assert_eq!(detect_sweep(&history, 2), SweepDirection::Random);
    }

    #[test]
    fn detect_random_pattern() {
        // Coordinates jumping around unpredictably
        let history = vec![
            vec![0, 0],
            vec![30, 20],
            vec![10, 0],
            vec![0, 30],
            vec![20, 10],
        ];
        assert_eq!(detect_sweep(&history, 2), SweepDirection::Random);
    }

    #[test]
    fn detect_random_empty() {
        assert_eq!(detect_sweep(&[], 2), SweepDirection::Random);
    }

    #[test]
    fn predict_row_major_2d() {
        let history = vec![
            vec![0, 0],
            vec![0, 10],
            vec![0, 20],
        ];
        let predictions = predict_next(&history, SweepDirection::RowMajor, 3);
        assert_eq!(predictions.len(), 3);
        assert_eq!(predictions[0], vec![0, 30]);
        assert_eq!(predictions[1], vec![0, 40]);
        assert_eq!(predictions[2], vec![0, 50]);
    }

    #[test]
    fn predict_column_major_2d() {
        let history = vec![
            vec![0, 0],
            vec![10, 0],
            vec![20, 0],
        ];
        let predictions = predict_next(&history, SweepDirection::ColumnMajor, 2);
        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0], vec![30, 0]);
        assert_eq!(predictions[1], vec![40, 0]);
    }

    #[test]
    fn predict_random_returns_empty() {
        let history = vec![vec![0, 0], vec![10, 20]];
        let predictions = predict_next(&history, SweepDirection::Random, 3);
        assert!(predictions.is_empty());
    }

    #[test]
    fn predict_too_few_history() {
        let history = vec![vec![0, 0]];
        let predictions = predict_next(&history, SweepDirection::RowMajor, 3);
        assert!(predictions.is_empty());
    }

    #[test]
    fn predict_slice_major_3d() {
        let history = vec![
            vec![0, 0, 0],
            vec![0, 4, 0],
            vec![0, 8, 0],
        ];
        let predictions = predict_next(&history, SweepDirection::SliceMajor(1), 2);
        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0], vec![0, 12, 0]);
        assert_eq!(predictions[1], vec![0, 16, 0]);
    }

    #[test]
    fn detect_and_predict_roundtrip() {
        let history = vec![
            vec![0, 0, 0],
            vec![0, 0, 8],
            vec![0, 0, 16],
            vec![0, 0, 24],
        ];
        let direction = detect_sweep(&history, 3);
        assert_eq!(direction, SweepDirection::RowMajor);

        let predicted = predict_next(&history, direction, 2);
        assert_eq!(predicted, vec![vec![0, 0, 32], vec![0, 0, 40]]);
    }

    #[test]
    fn false_positive_avoidance_alternating() {
        // Alternating pattern should not be detected as a sweep
        let history = vec![
            vec![0, 0],
            vec![10, 10],
            vec![0, 0],
            vec![10, 10],
            vec![0, 0],
        ];
        // Both dims change equally — should be Random
        assert_eq!(detect_sweep(&history, 2), SweepDirection::Random);
    }

    #[test]
    fn false_positive_avoidance_diagonal() {
        // Diagonal traversal — both dims change every step
        let history = vec![
            vec![0, 0],
            vec![10, 10],
            vec![20, 20],
            vec![30, 30],
        ];
        assert_eq!(detect_sweep(&history, 2), SweepDirection::Random);
    }
}
