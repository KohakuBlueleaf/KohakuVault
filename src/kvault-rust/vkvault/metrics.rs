//! Similarity metrics for vector search

use crate::vector_utils::VectorType;

/// Similarity metric for vector search
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMetric {
    Cosine,
    L2,
    L1,
    Hamming,
}

impl SimilarityMetric {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(SimilarityMetric::Cosine),
            "l2" | "euclidean" => Ok(SimilarityMetric::L2),
            "l1" | "manhattan" => Ok(SimilarityMetric::L1),
            "hamming" => Ok(SimilarityMetric::Hamming),
            _ => Err(format!("Unknown metric '{}'. Expected: cosine, l2, l1, or hamming", s)),
        }
    }

    pub fn to_str(self) -> &'static str {
        match self {
            SimilarityMetric::Cosine => "cosine",
            SimilarityMetric::L2 => "L2",
            SimilarityMetric::L1 => "L1",
            SimilarityMetric::Hamming => "hamming",
        }
    }

    pub fn is_compatible_with(&self, vec_type: VectorType) -> bool {
        match self {
            SimilarityMetric::Cosine => vec_type == VectorType::Float32,
            SimilarityMetric::L2 | SimilarityMetric::L1 => {
                vec_type == VectorType::Float32 || vec_type == VectorType::Int8
            }
            SimilarityMetric::Hamming => vec_type == VectorType::Bit,
        }
    }
}
