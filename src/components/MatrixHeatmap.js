import React, { useState, useEffect } from 'react';
import { heatmapColor, attentionToColor } from '../utils/colorUtils';

/**
 * MatrixHeatmap - Beautiful animated heatmap visualization
 * Shows matrices with color-coded values and smooth transitions
 */
const MatrixHeatmap = ({
  matrix,
  rowLabels = null,
  colLabels = null,
  title,
  description,
  isAttention = false,
  highlightRow = null,
  highlightCol = null,
  onCellHover = null,
  onCellClick = null,
  showValues = true,
  animate = true,
  colorScheme = 'default', // 'default', 'attention', 'diverging'
}) => {
  const [hoveredCell, setHoveredCell] = useState(null);
  const [displayMatrix, setDisplayMatrix] = useState([]);
  const [animationProgress, setAnimationProgress] = useState(0);

  // Animate matrix values appearing
  useEffect(() => {
    if (!animate || !matrix || matrix.length === 0) {
      setDisplayMatrix(matrix || []);
      return;
    }

    setAnimationProgress(0);
    const duration = 800;
    const startTime = Date.now();

    const animateValues = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(1, elapsed / duration);
      setAnimationProgress(progress);

      if (progress < 1) {
        requestAnimationFrame(animateValues);
      }
    };

    requestAnimationFrame(animateValues);
  }, [matrix, animate]);

  useEffect(() => {
    if (!matrix || matrix.length === 0) {
      setDisplayMatrix([]);
      return;
    }

    // Apply animation progress
    const animated = matrix.map(row =>
      row.map(val => val * animationProgress)
    );
    setDisplayMatrix(animated);
  }, [matrix, animationProgress]);

  if (!displayMatrix || displayMatrix.length === 0 || !displayMatrix[0]) {
    return (
      <div className="matrix-heatmap">
        {title && <h4 className="text-lg font-semibold mb-2">{title}</h4>}
        <div className="flex items-center justify-center h-32 bg-gray-50 rounded-lg border-2 border-gray-200">
          <p className="text-gray-500">No data to display</p>
        </div>
      </div>
    );
  }

  const rows = displayMatrix.length;
  const cols = displayMatrix[0].length;

  // Find min and max for normalization
  const allValues = displayMatrix.flat();
  const maxVal = Math.max(...allValues);
  const minVal = Math.min(...allValues);
  const maxAbs = Math.max(Math.abs(maxVal), Math.abs(minVal));

  const getCellColor = (value, rowIdx, colIdx) => {
    const originalValue = matrix[rowIdx][colIdx];

    if (isAttention || colorScheme === 'attention') {
      return attentionToColor(originalValue);
    } else if (colorScheme === 'diverging') {
      // Blue for negative, white for zero, red for positive
      if (originalValue > 0) {
        const intensity = originalValue / maxAbs;
        return `rgba(239, 68, 68, ${Math.min(1, intensity * 0.8 + 0.2)})`;
      } else if (originalValue < 0) {
        const intensity = Math.abs(originalValue) / maxAbs;
        return `rgba(59, 130, 246, ${Math.min(1, intensity * 0.8 + 0.2)})`;
      } else {
        return '#FFFFFF';
      }
    } else {
      // Default: use heatmap color scale
      const normalized = maxVal > minVal ? (originalValue - minVal) / (maxVal - minVal) : 0;
      return heatmapColor(normalized);
    }
  };

  const handleCellEnter = (rowIdx, colIdx, value) => {
    setHoveredCell({ row: rowIdx, col: colIdx });
    if (onCellHover) {
      onCellHover(rowIdx, colIdx, value);
    }
  };

  const handleCellLeave = () => {
    setHoveredCell(null);
    if (onCellHover) {
      onCellHover(null, null, null);
    }
  };

  const handleCellClickInternal = (rowIdx, colIdx, value) => {
    if (onCellClick) {
      onCellClick(rowIdx, colIdx, value);
    }
  };

  const isHighlighted = (rowIdx, colIdx) => {
    if (hoveredCell) {
      return hoveredCell.row === rowIdx || hoveredCell.col === colIdx;
    }
    if (highlightRow !== null || highlightCol !== null) {
      return rowIdx === highlightRow || colIdx === highlightCol;
    }
    return false;
  };

  const cellSize = Math.min(60, 400 / Math.max(rows, cols));
  const fontSize = cellSize > 40 ? '14px' : cellSize > 30 ? '12px' : '10px';

  return (
    <div className="matrix-heatmap">
      {title && (
        <h4 className="text-lg font-semibold mb-2 text-gray-800">{title}</h4>
      )}
      {description && (
        <p className="text-sm text-gray-600 mb-3">{description}</p>
      )}

      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          {/* Column labels */}
          {colLabels && (
            <div className="flex" style={{ marginLeft: rowLabels ? `${cellSize + 10}px` : '0' }}>
              {colLabels.map((label, i) => (
                <div
                  key={i}
                  className="font-medium text-center text-gray-700"
                  style={{
                    width: `${cellSize}px`,
                    fontSize: '13px',
                    marginBottom: '5px',
                  }}
                >
                  {label}
                </div>
              ))}
            </div>
          )}

          {/* Matrix grid */}
          <div className="flex flex-col">
            {displayMatrix.map((row, rowIdx) => (
              <div key={rowIdx} className="flex items-center">
                {/* Row label */}
                {rowLabels && rowLabels[rowIdx] && (
                  <div
                    className="font-medium text-gray-700 flex items-center justify-end pr-2"
                    style={{
                      width: `${cellSize}px`,
                      fontSize: '13px',
                    }}
                  >
                    {rowLabels[rowIdx]}
                  </div>
                )}

                {/* Row cells */}
                <div className="flex">
                  {row.map((value, colIdx) => {
                    const originalValue = matrix[rowIdx][colIdx];
                    const highlighted = isHighlighted(rowIdx, colIdx);
                    const isHovered = hoveredCell?.row === rowIdx && hoveredCell?.col === colIdx;

                    return (
                      <div
                        key={colIdx}
                        className="relative transition-all duration-200 cursor-pointer border border-gray-300"
                        style={{
                          width: `${cellSize}px`,
                          height: `${cellSize}px`,
                          backgroundColor: getCellColor(value, rowIdx, colIdx),
                          transform: isHovered ? 'scale(1.1)' : highlighted ? 'scale(1.05)' : 'scale(1)',
                          zIndex: isHovered ? 20 : highlighted ? 10 : 1,
                          boxShadow: isHovered
                            ? '0 8px 16px rgba(0,0,0,0.2)'
                            : highlighted
                            ? '0 4px 8px rgba(0,0,0,0.1)'
                            : 'none',
                          opacity: highlighted || !hoveredCell ? 1 : 0.5,
                        }}
                        onMouseEnter={() => handleCellEnter(rowIdx, colIdx, originalValue)}
                        onMouseLeave={handleCellLeave}
                        onClick={() => handleCellClickInternal(rowIdx, colIdx, originalValue)}
                      >
                        {showValues && (
                          <div
                            className="absolute inset-0 flex items-center justify-center font-mono font-semibold"
                            style={{
                              fontSize,
                              color: originalValue > maxAbs * 0.5 ? '#FFFFFF' : '#1F2937',
                            }}
                          >
                            {originalValue.toFixed(2)}
                          </div>
                        )}

                        {/* Highlight overlay */}
                        {isHovered && (
                          <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 bg-gray-900 text-white px-3 py-2 rounded-lg text-xs whitespace-nowrap z-30 shadow-lg">
                            <div className="font-semibold">
                              [{rowIdx}, {colIdx}] = {originalValue.toFixed(4)}
                            </div>
                            {rowLabels && colLabels && (
                              <div className="text-gray-300 mt-1">
                                {rowLabels[rowIdx]} → {colLabels[colIdx]}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Color scale legend */}
      {isAttention && (
        <div className="mt-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
          <div className="text-xs font-semibold text-gray-700 mb-2">Attention Weight Scale</div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-600">Low</span>
            <div className="flex-1 h-6 rounded-lg overflow-hidden flex">
              {[0, 0.25, 0.5, 0.75, 1].map((val, i) => (
                <div
                  key={i}
                  style={{
                    flex: 1,
                    backgroundColor: attentionToColor(val),
                  }}
                />
              ))}
            </div>
            <span className="text-xs text-gray-600">High</span>
          </div>
        </div>
      )}

      {/* Matrix statistics */}
      {hoveredCell === null && (
        <div className="mt-3 text-xs text-gray-600 space-y-1">
          <div>Matrix size: {rows} × {cols}</div>
          <div>Range: [{minVal.toFixed(3)}, {maxVal.toFixed(3)}]</div>
          {isAttention && (
            <div className="text-blue-600 font-medium">
              Each row sums to 1.0 (probability distribution)
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default MatrixHeatmap;
