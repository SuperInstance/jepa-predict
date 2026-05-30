//! jepa-predict: JEPA prediction engine with VectorDb, Prediction, Surprise types,
//! and conservation tracking.

use std::collections::HashMap;

/// A vector database for storing and querying state embeddings.
#[derive(Debug, Clone, Default)]
pub struct VectorDb {
    vectors: Vec<Vec<f32>>,
    ids: Vec<String>,
}

impl VectorDb {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a named vector into the database.
    pub fn insert(&mut self, id: impl Into<String>, vector: Vec<f32>) {
        self.ids.push(id.into());
        self.vectors.push(vector);
    }

    /// Number of stored vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Retrieve a vector by index.
    pub fn get(&self, index: usize) -> Option<&Vec<f32>> {
        self.vectors.get(index)
    }

    /// Retrieve an id by index.
    pub fn get_id(&self, index: usize) -> Option<&str> {
        self.ids.get(index).map(|s| s.as_str())
    }

    /// Find the nearest neighbor by Euclidean distance.
    pub fn nearest(&self, query: &[f32]) -> Option<(usize, f32)> {
        let mut best = None;
        let mut best_dist = f32::MAX;
        for (i, v) in self.vectors.iter().enumerate() {
            if v.len() != query.len() {
                continue;
            }
            let d = euclidean_distance(v, query);
            if d < best_dist {
                best_dist = d;
                best = Some(i);
            }
        }
        best.map(|i| (i, best_dist))
    }

    /// Find k nearest neighbors.
    pub fn knn(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(_, v)| v.len() == query.len())
            .map(|(i, v)| (i, euclidean_distance(v, query)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.into_iter().take(k).collect()
    }

    /// Clear all vectors.
    pub fn clear(&mut self) {
        self.vectors.clear();
        self.ids.clear();
    }

    /// Compute the centroid of all stored vectors.
    pub fn centroid(&self) -> Option<Vec<f32>> {
        if self.vectors.is_empty() {
            return None;
        }
        let dim = self.vectors[0].len();
        let mut sum = vec![0.0f32; dim];
        for v in &self.vectors {
            for (s, x) in sum.iter_mut().zip(v.iter()) {
                *s += x;
            }
        }
        let n = self.vectors.len() as f32;
        for s in &mut sum {
            *s /= n;
        }
        Some(sum)
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

/// A prediction produced by the JEPA engine.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    pub predicted_vector: Vec<f32>,
    pub confidence: f32,
    pub source_index: Option<usize>,
}

impl Prediction {
    pub fn new(predicted_vector: Vec<f32>, confidence: f32) -> Self {
        Self {
            predicted_vector,
            confidence: confidence.clamp(0.0, 1.0),
            source_index: None,
        }
    }

    pub fn with_source(mut self, index: usize) -> Self {
        self.source_index = Some(index);
        self
    }

    /// Compute surprise between prediction and actual outcome.
    pub fn surprise(&self, actual: &[f32]) -> Surprise {
        if self.predicted_vector.len() != actual.len() {
            return Surprise {
                mse: f32::MAX,
                error_vector: vec![],
            };
        }
        let mut error_vector = vec![0.0f32; actual.len()];
        let mut mse = 0.0f32;
        for (i, (p, a)) in self.predicted_vector.iter().zip(actual.iter()).enumerate() {
            let diff = a - p;
            error_vector[i] = diff;
            mse += diff * diff;
        }
        mse /= actual.len() as f32;
        Surprise { mse, error_vector }
    }
}

/// Surprise metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct Surprise {
    pub mse: f32,
    pub error_vector: Vec<f32>,
}

impl Surprise {
    pub fn rmse(&self) -> f32 {
        self.mse.sqrt()
    }

    pub fn is_significant(&self, threshold: f32) -> bool {
        self.mse > threshold
    }
}

/// JEPA prediction engine.
#[derive(Debug, Clone, Default)]
pub struct JepaEngine {
    db: VectorDb,
    history: Vec<Vec<f32>>,
    max_history: usize,
}

impl JepaEngine {
    pub fn new() -> Self {
        Self {
            db: VectorDb::new(),
            history: Vec::new(),
            max_history: 10,
        }
    }

    pub fn with_capacity(max_history: usize) -> Self {
        Self {
            db: VectorDb::new(),
            history: Vec::new(),
            max_history,
        }
    }

    /// Observe a new state vector.
    pub fn observe(&mut self, vector: Vec<f32>) {
        self.history.push(vector);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Predict the next state by linear extrapolation from history.
    pub fn predict(&self) -> Option<Prediction> {
        if self.history.len() < 2 {
            return None;
        }
        let last = self.history.last()?;
        let prev = self.history.get(self.history.len().saturating_sub(2))?;
        let mut predicted = vec![0.0f32; last.len()];
        for i in 0..last.len() {
            let delta = last[i] - prev[i];
            predicted[i] = last[i] + delta;
        }
        Some(Prediction::new(predicted, 0.8))
    }

    /// Predict using nearest-neighbor lookup in the vector DB.
    pub fn predict_from_db(&self, query: &[f32]) -> Option<Prediction> {
        let (idx, dist) = self.db.nearest(query)?;
        let predicted = self.db.get(idx)?.clone();
        let confidence = (1.0 / (1.0 + dist)).clamp(0.0, 1.0);
        Some(Prediction::new(predicted, confidence).with_source(idx))
    }

    pub fn db(&self) -> &VectorDb {
        &self.db
    }

    pub fn db_mut(&mut self) -> &mut VectorDb {
        &mut self.db
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

/// Tracks conservation of a scalar quantity across predictions.
#[derive(Debug, Clone, Default)]
pub struct ConservationTracker {
    values: Vec<f32>,
    pub expected_total: f32,
}

impl ConservationTracker {
    pub fn new(expected_total: f32) -> Self {
        Self {
            values: Vec::new(),
            expected_total,
        }
    }

    pub fn record(&mut self, value: f32) {
        self.values.push(value);
    }

    pub fn current_total(&self) -> f32 {
        self.values.iter().sum()
    }

    pub fn drift(&self) -> f32 {
        (self.current_total() - self.expected_total).abs()
    }

    pub fn is_conserved(&self, tolerance: f32) -> bool {
        self.drift() <= tolerance
    }

    pub fn count(&self) -> usize {
        self.values.len()
    }

    pub fn mean(&self) -> f32 {
        if self.values.is_empty() {
            0.0
        } else {
            self.current_total() / self.values.len() as f32
        }
    }

    pub fn variance(&self) -> f32 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let sum_sq: f32 = self.values.iter().map(|v| (v - mean).powi(2)).sum();
        sum_sq / self.values.len() as f32
    }

    pub fn reset(&mut self) {
        self.values.clear();
    }
}

/// A predictor that uses simple moving-average forecasting.
#[derive(Debug, Clone, Default)]
pub struct MovingAveragePredictor {
    window: Vec<Vec<f32>>,
    window_size: usize,
}

impl MovingAveragePredictor {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: Vec::new(),
            window_size: window_size.max(1),
        }
    }

    pub fn push(&mut self, vector: Vec<f32>) {
        self.window.push(vector);
        if self.window.len() > self.window_size {
            self.window.remove(0);
        }
    }

    pub fn predict(&self) -> Option<Prediction> {
        if self.window.is_empty() {
            return None;
        }
        let dim = self.window[0].len();
        let mut avg = vec![0.0f32; dim];
        for v in &self.window {
            for (a, x) in avg.iter_mut().zip(v.iter()) {
                *a += x;
            }
        }
        let n = self.window.len() as f32;
        for a in &mut avg {
            *a /= n;
        }
        Some(Prediction::new(avg, 0.7))
    }
}

/// Batch predictor for multiple independent channels.
#[derive(Debug, Clone, Default)]
pub struct BatchPredictor {
    channels: HashMap<String, MovingAveragePredictor>,
    window_size: usize,
}

impl BatchPredictor {
    pub fn new(window_size: usize) -> Self {
        Self {
            channels: HashMap::new(),
            window_size,
        }
    }

    pub fn push(&mut self, channel: impl Into<String>, vector: Vec<f32>) {
        let key = channel.into();
        self.channels
            .entry(key)
            .or_insert_with(|| MovingAveragePredictor::new(self.window_size))
            .push(vector);
    }

    pub fn predict(&self, channel: &str) -> Option<Prediction> {
        self.channels.get(channel)?.predict()
    }

    pub fn channels(&self) -> Vec<&String> {
        self.channels.keys().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_db_insert_and_get() {
        let mut db = VectorDb::new();
        db.insert("a", vec![1.0, 2.0, 3.0]);
        assert_eq!(db.len(), 1);
        assert_eq!(db.get(0), Some(&vec![1.0, 2.0, 3.0]));
        assert_eq!(db.get_id(0), Some("a"));
    }

    #[test]
    fn test_vector_db_nearest() {
        let mut db = VectorDb::new();
        db.insert("a", vec![0.0, 0.0]);
        db.insert("b", vec![10.0, 0.0]);
        let (idx, dist) = db.nearest(&[1.0, 0.0]).unwrap();
        assert_eq!(idx, 0);
        assert!((dist - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vector_db_knn() {
        let mut db = VectorDb::new();
        db.insert("a", vec![0.0, 0.0]);
        db.insert("b", vec![0.9, 0.0]);
        db.insert("c", vec![10.0, 0.0]);
        let knn = db.knn(&[0.5, 0.0], 2);
        assert_eq!(knn.len(), 2);
        assert_eq!(knn[0].0, 1); // b is closest
    }

    #[test]
    fn test_vector_db_centroid() {
        let mut db = VectorDb::new();
        db.insert("a", vec![0.0, 0.0]);
        db.insert("b", vec![2.0, 4.0]);
        let c = db.centroid().unwrap();
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_vector_db_clear() {
        let mut db = VectorDb::new();
        db.insert("a", vec![1.0]);
        db.clear();
        assert!(db.is_empty());
    }

    #[test]
    fn test_prediction_new() {
        let p = Prediction::new(vec![1.0, 2.0], 0.5);
        assert_eq!(p.predicted_vector, vec![1.0, 2.0]);
        assert_eq!(p.confidence, 0.5);
    }

    #[test]
    fn test_prediction_clamps_confidence() {
        let p = Prediction::new(vec![], 1.5);
        assert_eq!(p.confidence, 1.0);
        let p2 = Prediction::new(vec![], -0.5);
        assert_eq!(p2.confidence, 0.0);
    }

    #[test]
    fn test_prediction_with_source() {
        let p = Prediction::new(vec![1.0], 0.5).with_source(3);
        assert_eq!(p.source_index, Some(3));
    }

    #[test]
    fn test_surprise_mse() {
        let p = Prediction::new(vec![0.0, 2.0], 1.0);
        let s = p.surprise(&[0.0, 0.0]);
        assert!((s.mse - 2.0).abs() < 1e-5);
        assert_eq!(s.error_vector, vec![0.0, -2.0]);
    }

    #[test]
    fn test_surprise_rmse() {
        let p = Prediction::new(vec![0.0], 1.0);
        let s = p.surprise(&[4.0]);
        assert!((s.rmse() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_surprise_significant() {
        let p = Prediction::new(vec![0.0], 1.0);
        let s = p.surprise(&[10.0]);
        assert!(s.is_significant(50.0));
        assert!(!s.is_significant(200.0));
    }

    #[test]
    fn test_jepa_observe_and_predict() {
        let mut engine = JepaEngine::new();
        engine.observe(vec![0.0, 0.0]);
        engine.observe(vec![1.0, 1.0]);
        let pred = engine.predict().unwrap();
        assert_eq!(pred.predicted_vector, vec![2.0, 2.0]);
    }

    #[test]
    fn test_jepa_predict_not_enough_history() {
        let engine = JepaEngine::new();
        assert!(engine.predict().is_none());
    }

    #[test]
    fn test_jepa_predict_from_db() {
        let mut engine = JepaEngine::new();
        engine.db_mut().insert("a", vec![0.0, 0.0]);
        engine.db_mut().insert("b", vec![1.0, 1.0]);
        let pred = engine.predict_from_db(&[0.9, 0.9]).unwrap();
        assert_eq!(pred.source_index, Some(1));
    }

    #[test]
    fn test_jepa_history_management() {
        let mut engine = JepaEngine::with_capacity(3);
        engine.observe(vec![0.0]);
        engine.observe(vec![1.0]);
        engine.observe(vec![2.0]);
        engine.observe(vec![3.0]);
        assert_eq!(engine.history_len(), 3);
    }

    #[test]
    fn test_conservation_tracker() {
        let mut ct = ConservationTracker::new(10.0);
        ct.record(3.0);
        ct.record(4.0);
        ct.record(3.0);
        assert!((ct.current_total() - 10.0).abs() < 1e-5);
        assert!(ct.is_conserved(1e-4));
    }

    #[test]
    fn test_conservation_tracker_drift() {
        let mut ct = ConservationTracker::new(10.0);
        ct.record(5.0);
        assert!((ct.drift() - 5.0).abs() < 1e-5);
        assert!(!ct.is_conserved(1e-4));
    }

    #[test]
    fn test_conservation_stats() {
        let mut ct = ConservationTracker::new(0.0);
        ct.record(2.0);
        ct.record(4.0);
        ct.record(6.0);
        assert!((ct.mean() - 4.0).abs() < 1e-5);
        assert!(ct.variance() > 0.0);
    }

    #[test]
    fn test_moving_average_predictor() {
        let mut p = MovingAveragePredictor::new(2);
        p.push(vec![0.0, 0.0]);
        p.push(vec![2.0, 4.0]);
        let pred = p.predict().unwrap();
        assert_eq!(pred.predicted_vector, vec![1.0, 2.0]);
    }

    #[test]
    fn test_batch_predictor() {
        let mut bp = BatchPredictor::new(2);
        bp.push("ch1", vec![1.0]);
        bp.push("ch1", vec![3.0]);
        bp.push("ch2", vec![10.0]);
        let pred = bp.predict("ch1").unwrap();
        assert_eq!(pred.predicted_vector, vec![2.0]);
        let pred2 = bp.predict("ch2");
        assert!(pred2.is_none() || pred2.unwrap().predicted_vector == vec![10.0]);
    }
}
