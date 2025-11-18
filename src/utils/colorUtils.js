/**
 * Color utilities for 3Blue1Brown-inspired visualizations
 * Using a cohesive color scheme for mathematical concepts
 */

// 3Blue1Brown-inspired color palette
export const COLORS = {
  // Primary colors for Q, K, V (inspired by 3b1b)
  query: {
    main: '#FF6B6B',      // Red/pink for queries
    light: '#FFB3B3',
    dark: '#CC5555',
    glow: 'rgba(255, 107, 107, 0.3)',
  },
  key: {
    main: '#4ECDC4',      // Cyan for keys
    light: '#A7E6E0',
    dark: '#3CA39C',
    glow: 'rgba(78, 205, 196, 0.3)',
  },
  value: {
    main: '#95E1D3',      // Mint green for values
    light: '#C9F0E8',
    dark: '#6BB3A3',
    glow: 'rgba(149, 225, 211, 0.3)',
  },

  // Attention-specific colors
  attention: {
    high: '#FFD93D',      // Yellow for high attention
    medium: '#FFA07A',    // Orange for medium
    low: '#B8B8D1',       // Light purple for low
    background: '#F8F9FA',
  },

  // Mathematical elements
  math: {
    matrix: '#6C63FF',    // Purple for matrices
    vector: '#FF6B9D',    // Pink for vectors
    scalar: '#45B7D1',    // Blue for scalars
    operation: '#96CEB4', // Green for operations
  },

  // UI colors
  ui: {
    background: '#FAFBFC',
    card: '#FFFFFF',
    border: '#E1E8ED',
    text: '#1F2937',
    textLight: '#6B7280',
    accent: '#3B82F6',
    success: '#10B981',
    warning: '#F59E0B',
  },

  // Gradient presets
  gradients: {
    attention: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    query: 'linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%)',
    key: 'linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%)',
    value: 'linear-gradient(135deg, #95E1D3 0%, #6DD5C3 100%)',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
  },
};

/**
 * Convert attention weight to color
 * @param {number} weight - Attention weight (0 to 1)
 * @param {number} opacity - Opacity override (optional)
 * @returns {string} RGBA color string
 */
export const attentionToColor = (weight, opacity = null) => {
  const alpha = opacity !== null ? opacity : Math.max(0.3, Math.min(1, weight));

  if (weight > 0.7) {
    return `rgba(255, 217, 61, ${alpha})`; // High attention - yellow
  } else if (weight > 0.4) {
    return `rgba(255, 160, 122, ${alpha})`; // Medium attention - orange
  } else {
    return `rgba(184, 184, 209, ${alpha})`; // Low attention - light purple
  }
};

/**
 * Get color for matrix cell based on value
 * @param {number} value - Matrix value
 * @param {number} maxAbs - Maximum absolute value in matrix
 * @returns {string} RGB color string
 */
export const matrixCellColor = (value, maxAbs = 1) => {
  const normalized = maxAbs > 0 ? value / maxAbs : 0;
  const absNormalized = Math.abs(normalized);

  if (normalized > 0) {
    // Positive values: blue scale
    const intensity = Math.floor(absNormalized * 200 + 55);
    return `rgb(${255 - intensity}, ${255 - intensity}, 255)`;
  } else {
    // Negative values: red scale
    const intensity = Math.floor(absNormalized * 200 + 55);
    return `rgb(255, ${255 - intensity}, ${255 - intensity})`;
  }
};

/**
 * Interpolate between two colors
 * @param {string} color1 - Start color (hex)
 * @param {string} color2 - End color (hex)
 * @param {number} ratio - Interpolation ratio (0 to 1)
 * @returns {string} Interpolated color (hex)
 */
export const interpolateColor = (color1, color2, ratio) => {
  const hex = (color) => {
    const c = color.replace('#', '');
    return [
      parseInt(c.substring(0, 2), 16),
      parseInt(c.substring(2, 4), 16),
      parseInt(c.substring(4, 6), 16),
    ];
  };

  const [r1, g1, b1] = hex(color1);
  const [r2, g2, b2] = hex(color2);

  const r = Math.round(r1 + (r2 - r1) * ratio);
  const g = Math.round(g1 + (g2 - g1) * ratio);
  const b = Math.round(b1 + (b2 - b1) * ratio);

  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
};

/**
 * Get color scale for heatmap
 * @param {number} value - Value between 0 and 1
 * @returns {string} Color string
 */
export const heatmapColor = (value) => {
  // Use a perceptually uniform color scale (viridis-like)
  const clampedValue = Math.max(0, Math.min(1, value));

  if (clampedValue < 0.25) {
    return interpolateColor('#440154', '#31688e', clampedValue * 4);
  } else if (clampedValue < 0.5) {
    return interpolateColor('#31688e', '#35b779', (clampedValue - 0.25) * 4);
  } else if (clampedValue < 0.75) {
    return interpolateColor('#35b779', '#fde724', (clampedValue - 0.5) * 4);
  } else {
    return interpolateColor('#fde724', '#ffffff', (clampedValue - 0.75) * 4);
  }
};

/**
 * Generate rainbow colors for multi-head attention
 * @param {number} index - Head index
 * @param {number} total - Total number of heads
 * @returns {string} Color string
 */
export const rainbowColor = (index, total) => {
  const hue = (index * 360) / total;
  return `hsl(${hue}, 70%, 60%)`;
};
