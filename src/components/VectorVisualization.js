import React, { useEffect, useRef } from 'react';
import { COLORS } from '../utils/colorUtils';
import { magnitude } from '../utils/mathUtils';

/**
 * VectorVisualization - Geometric visualization of vectors
 * Inspired by 3Blue1Brown's visual style
 */
const VectorVisualization = ({
  vectors,
  labels,
  colors,
  title,
  showGrid = true,
  showAxes = true,
  animate = false,
  highlightIndex = null,
  width = 400,
  height = 400,
}) => {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const timeRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !vectors || vectors.length === 0) return;

    const ctx = canvas.getContext('2d');
    const centerX = width / 2;
    const centerY = height / 2;
    const scale = Math.min(width, height) / 6; // Scale for vector display

    const drawGrid = () => {
      ctx.strokeStyle = COLORS.ui.border;
      ctx.lineWidth = 0.5;

      // Draw grid lines
      for (let i = -3; i <= 3; i++) {
        // Vertical lines
        ctx.beginPath();
        ctx.moveTo(centerX + i * scale, 0);
        ctx.lineTo(centerX + i * scale, height);
        ctx.stroke();

        // Horizontal lines
        ctx.beginPath();
        ctx.moveTo(0, centerY + i * scale);
        ctx.lineTo(width, centerY + i * scale);
        ctx.stroke();
      }
    };

    const drawAxes = () => {
      ctx.strokeStyle = COLORS.ui.text;
      ctx.lineWidth = 2;

      // X-axis
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();

      // Y-axis
      ctx.beginPath();
      ctx.moveTo(centerX, 0);
      ctx.lineTo(centerX, height);
      ctx.stroke();

      // Axis labels
      ctx.fillStyle = COLORS.ui.text;
      ctx.font = '14px sans-serif';
      ctx.fillText('x₁', width - 25, centerY - 10);
      ctx.fillText('x₂', centerX + 10, 20);
    };

    const drawVector = (vector, color, label, opacity = 1, pulsePhase = 0) => {
      if (!vector || vector.length < 2) return;

      const [x, y] = vector;
      const endX = centerX + x * scale;
      const endY = centerY - y * scale; // Negative because canvas Y is inverted

      // Draw vector line with glow effect
      ctx.save();

      // Glow
      ctx.shadowBlur = 15;
      ctx.shadowColor = color;

      // Vector line
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.globalAlpha = opacity;

      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(endX, endY);
      ctx.stroke();

      // Arrow head
      const angle = Math.atan2(-y, x);
      const headLength = 15;

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(
        endX - headLength * Math.cos(angle - Math.PI / 6),
        endY + headLength * Math.sin(angle - Math.PI / 6)
      );
      ctx.lineTo(
        endX - headLength * Math.cos(angle + Math.PI / 6),
        endY + headLength * Math.sin(angle + Math.PI / 6)
      );
      ctx.closePath();
      ctx.fill();

      // Pulsing circle at endpoint
      const pulseSize = 6 + Math.sin(pulsePhase) * 2;
      ctx.beginPath();
      ctx.arc(endX, endY, pulseSize, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      ctx.restore();

      // Label
      ctx.fillStyle = color;
      ctx.font = 'bold 14px sans-serif';
      ctx.globalAlpha = 1;
      const labelX = endX + 15;
      const labelY = endY - 15;
      ctx.fillText(label, labelX, labelY);

      // Show vector magnitude
      const mag = magnitude(vector);
      ctx.font = '12px sans-serif';
      ctx.fillStyle = COLORS.ui.textLight;
      ctx.fillText(`|${label}| = ${mag.toFixed(2)}`, labelX, labelY + 15);
    };

    const drawDotProductVisualization = (vec1, vec2, color1, color2) => {
      if (!vec1 || !vec2 || vec1.length < 2 || vec2.length < 2) return;

      const [x1, y1] = vec1;
      const [x2, y2] = vec2;

      // Draw projection
      const dotProd = x1 * x2 + y1 * y2;
      const mag2Squared = x2 * x2 + y2 * y2;

      if (mag2Squared === 0) return;

      const projectionScale = dotProd / mag2Squared;
      const projX = x2 * projectionScale;
      const projY = y2 * projectionScale;

      const projEndX = centerX + projX * scale;
      const projEndY = centerY - projY * scale;

      // Draw projection line (dashed)
      ctx.save();
      ctx.strokeStyle = color1;
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.globalAlpha = 0.5;

      const vec1EndX = centerX + x1 * scale;
      const vec1EndY = centerY - y1 * scale;

      ctx.beginPath();
      ctx.moveTo(vec1EndX, vec1EndY);
      ctx.lineTo(projEndX, projEndY);
      ctx.stroke();

      ctx.restore();

      // Draw right angle indicator
      if (Math.abs(dotProd) < 0.1) {
        const size = 15;
        ctx.strokeStyle = COLORS.ui.textLight;
        ctx.lineWidth = 1;
        ctx.strokeRect(
          centerX - size / 2,
          centerY - size / 2,
          size,
          size
        );
      }
    };

    const draw = (time) => {
      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Background
      ctx.fillStyle = COLORS.ui.background;
      ctx.fillRect(0, 0, width, height);

      // Draw grid and axes
      if (showGrid) drawGrid();
      if (showAxes) drawAxes();

      // Draw vectors
      vectors.forEach((vector, index) => {
        const color = colors ? colors[index] : COLORS.math.vector;
        const label = labels ? labels[index] : `v${index}`;
        const isHighlighted = highlightIndex === index;
        const opacity = highlightIndex === null || isHighlighted ? 1 : 0.3;
        const pulsePhase = animate ? time / 500 : 0;

        drawVector(vector, color, label, opacity, pulsePhase);
      });

      // Draw dot product visualization if we have exactly 2 vectors
      if (vectors.length === 2 && colors && colors.length === 2) {
        drawDotProductVisualization(vectors[0], vectors[1], colors[0], colors[1]);
      }

      if (animate) {
        animationFrameRef.current = requestAnimationFrame(draw);
      }
    };

    if (animate) {
      const animationLoop = (timestamp) => {
        timeRef.current = timestamp;
        draw(timestamp);
      };
      animationFrameRef.current = requestAnimationFrame(animationLoop);
    } else {
      draw(0);
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [vectors, labels, colors, showGrid, showAxes, animate, highlightIndex, width, height]);

  return (
    <div className="vector-visualization">
      {title && (
        <h4 className="text-lg font-semibold mb-3 text-gray-800">{title}</h4>
      )}
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="border-2 border-gray-200 rounded-lg bg-white"
        style={{ maxWidth: '100%', height: 'auto' }}
      />
    </div>
  );
};

export default VectorVisualization;
