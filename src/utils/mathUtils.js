/**
 * Mathematical utilities for attention mechanism calculations
 * Following the transformer architecture from "Attention is All You Need"
 */

/**
 * Matrix multiplication: A × B
 * @param {number[][]} A - Matrix A (m × n)
 * @param {number[][]} B - Matrix B (n × p)
 * @returns {number[][]} Result matrix (m × p)
 */
export const matMul = (A, B) => {
  if (!A || !B || !A.length || !B.length || !A[0] || !B[0]) return [];

  const result = Array.from({ length: A.length }, () =>
    Array.from({ length: B[0].length }, () => 0)
  );

  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < B[0].length; j++) {
      for (let k = 0; k < B.length; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return result.map(row => row.map(val => Math.round(val * 100) / 100));
};

/**
 * Transpose a matrix
 * @param {number[][]} matrix - Input matrix
 * @returns {number[][]} Transposed matrix
 */
export const transpose = (matrix) => {
  if (!matrix || !matrix.length || !matrix[0]) return [];
  return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
};

/**
 * Softmax function with numerical stability
 * Converts scores to probability distribution
 * @param {number[]} arr - Input array of scores
 * @returns {number[]} Probability distribution summing to 1
 */
export const softmax = (arr) => {
  if (!arr || !arr.length) return [];

  // Subtract max for numerical stability
  const maxVal = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - maxVal));
  const sum = exp.reduce((a, b) => a + b, 0);

  return exp.map(x => Math.round((x / sum) * 1000) / 1000);
};

/**
 * Compute dot product of two vectors
 * @param {number[]} a - First vector
 * @param {number[]} b - Second vector
 * @returns {number} Dot product
 */
export const dotProduct = (a, b) => {
  if (!a || !b || a.length !== b.length) return 0;
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
};

/**
 * Compute vector magnitude (L2 norm)
 * @param {number[]} vector - Input vector
 * @returns {number} Magnitude
 */
export const magnitude = (vector) => {
  if (!vector || !vector.length) return 0;
  return Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
};

/**
 * Compute cosine similarity between two vectors
 * @param {number[]} a - First vector
 * @param {number[]} b - Second vector
 * @returns {number} Cosine similarity (-1 to 1)
 */
export const cosineSimilarity = (a, b) => {
  const dot = dotProduct(a, b);
  const magA = magnitude(a);
  const magB = magnitude(b);

  if (magA === 0 || magB === 0) return 0;
  return dot / (magA * magB);
};

/**
 * Generate embeddings for tokens
 * Educational simplified version with meaningful patterns
 * @param {string[]} tokens - Array of word tokens
 * @param {number} dModel - Embedding dimension
 * @returns {number[][]} Embedding matrix
 */
export const generateEmbeddings = (tokens, dModel = 4) => {
  return tokens.map((token, i) => {
    const embedding = [];

    for (let d = 0; d < dModel; d++) {
      // Create embeddings with patterns based on:
      // - Position (i)
      // - Word length
      // - Character codes
      // This creates interpretable patterns for educational purposes
      let value = 0;

      if (d === 0) {
        // Position encoding component
        value = Math.sin(i * 0.5) * 0.8 + 0.2;
      } else if (d === 1) {
        // Another position component
        value = Math.cos(i * 0.3) * 0.6 + 0.4;
      } else if (d === 2) {
        // Word length component
        value = (token.length / 10) * 0.8;
      } else {
        // Character-based component
        value = ((token.charCodeAt(0) % 26) / 26) * 0.8;
      }

      embedding.push(Math.round(value * 100) / 100);
    }

    return embedding;
  });
};

/**
 * Initialize random weight matrix (simplified for education)
 * @param {number} inputDim - Input dimension
 * @param {number} outputDim - Output dimension
 * @param {number} seed - Random seed for reproducibility
 * @returns {number[][]} Weight matrix
 */
export const initWeightMatrix = (inputDim, outputDim, seed = 0) => {
  const weights = [];

  for (let i = 0; i < inputDim; i++) {
    const row = [];
    for (let j = 0; j < outputDim; j++) {
      // Simple pseudo-random with seed for reproducibility
      const value = Math.sin(seed + i * 3 + j * 7) * 0.8;
      row.push(Math.round(value * 100) / 100);
    }
    weights.push(row);
  }

  return weights;
};

/**
 * Compute attention scores (Q × K^T / √d_k)
 * @param {number[][]} Q - Query matrix
 * @param {number[][]} K - Key matrix
 * @param {number} dK - Key dimension for scaling
 * @returns {number[][]} Scaled attention scores
 */
export const computeAttentionScores = (Q, K, dK) => {
  if (!Q || !K || !Q.length || !K.length) return [];

  const KT = transpose(K);
  const scores = matMul(Q, KT);
  const scale = Math.sqrt(dK);

  return scores.map(row =>
    row.map(val => Math.round((val / scale) * 100) / 100)
  );
};

/**
 * Apply softmax to get attention weights
 * @param {number[][]} scores - Attention scores
 * @returns {number[][]} Attention weight matrix
 */
export const computeAttentionWeights = (scores) => {
  if (!scores || !scores.length) return [];
  return scores.map(row => softmax(row));
};

/**
 * Compute final attention output (Attention_weights × V)
 * @param {number[][]} attentionWeights - Attention weight matrix
 * @param {number[][]} V - Value matrix
 * @returns {number[][]} Output matrix
 */
export const computeAttentionOutput = (attentionWeights, V) => {
  return matMul(attentionWeights, V);
};
