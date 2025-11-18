import React, { useEffect, useRef, useState } from 'react';
import { COLORS } from '../utils/colorUtils';
import { dotProduct, magnitude, cosineSimilarity } from '../utils/mathUtils';

/**
 * DotProductVisualization - Shows geometric interpretation of Q·K similarity
 * Visualizes why dot product measures similarity/alignment
 */
const DotProductVisualization = ({
  query,
  key,
  queryLabel = 'Query',
  keyLabel = 'Key',
  animate = true,
}) => {
  const canvasRef = useRef(null);
  const [animationPhase, setAnimationPhase] = useState(0);

  useEffect(() => {
    if (!animate) return;

    const interval = setInterval(() => {
      setAnimationPhase((phase) => (phase + 0.05) % (Math.PI * 2));
    }, 50);

    return () => clearInterval(interval);
  }, [animate]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !query || !key || query.length < 2 || key.length < 2) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const scale = Math.min(width, height) / 5;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Background gradient
    const bgGradient = ctx.createLinearGradient(0, 0, width, height);
    bgGradient.addColorStop(0, '#FAFBFC');
    bgGradient.addColorStop(1, '#F0F4F8');
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, width, height);

    // Draw subtle grid
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
    ctx.lineWidth = 1;
    for (let i = -3; i <= 3; i++) {
      ctx.beginPath();
      ctx.moveTo(centerX + i * scale, 0);
      ctx.lineTo(centerX + i * scale, height);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, centerY + i * scale);
      ctx.lineTo(width, centerY + i * scale);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, height);
    ctx.stroke();

    // Helper function to draw vector
    const drawVector = (vec, color, label, glow = true) => {
      const [x, y] = vec;
      const endX = centerX + x * scale;
      const endY = centerY - y * scale;

      ctx.save();

      if (glow) {
        ctx.shadowBlur = 20;
        ctx.shadowColor = color;
      }

      // Vector line
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(endX, endY);
      ctx.stroke();

      // Arrow head
      const angle = Math.atan2(-y, x);
      const headLength = 18;

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(
        endX - headLength * Math.cos(angle - Math.PI / 7),
        endY + headLength * Math.sin(angle - Math.PI / 7)
      );
      ctx.lineTo(
        endX - headLength * Math.cos(angle + Math.PI / 7),
        endY + headLength * Math.sin(angle + Math.PI / 7)
      );
      ctx.closePath();
      ctx.fill();

      // Endpoint circle with pulse
      const pulseRadius = 7 + Math.sin(animationPhase * 2) * 2;
      ctx.beginPath();
      ctx.arc(endX, endY, pulseRadius, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();

      // Label
      ctx.fillStyle = color;
      ctx.font = 'bold 16px sans-serif';
      ctx.fillText(label, endX + 20, endY - 10);
    };

    // Calculate metrics
    const dot = dotProduct(query, key);
    const qMag = magnitude(query);
    const kMag = magnitude(key);
    const cosSim = cosineSimilarity(query, key);
    const angle = Math.acos(Math.max(-1, Math.min(1, cosSim)));

    // Draw angle arc
    if (Math.abs(angle) > 0.01) {
      const arcRadius = 50;
      const qAngle = Math.atan2(-query[1], query[0]);
      const kAngle = Math.atan2(-key[1], key[0]);

      ctx.strokeStyle = COLORS.attention.medium;
      ctx.fillStyle = `${COLORS.attention.medium}33`;
      ctx.lineWidth = 2;

      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.arc(centerX, centerY, arcRadius, qAngle, kAngle, qAngle > kAngle);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Angle label
      const midAngle = (qAngle + kAngle) / 2;
      const labelX = centerX + Math.cos(midAngle) * (arcRadius + 20);
      const labelY = centerY - Math.sin(midAngle) * (arcRadius + 20);

      ctx.fillStyle = COLORS.ui.text;
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText(`θ = ${(angle * 180 / Math.PI).toFixed(1)}°`, labelX, labelY);
    }

    // Draw projection of query onto key
    const projectionScale = dot / (kMag * kMag);
    const projX = key[0] * projectionScale;
    const projY = key[1] * projectionScale;

    const qEndX = centerX + query[0] * scale;
    const qEndY = centerY - query[1] * scale;
    const projEndX = centerX + projX * scale;
    const projEndY = centerY - projY * scale;

    // Draw projection line (dashed)
    ctx.setLineDash([8, 8]);
    ctx.strokeStyle = COLORS.query.main;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.moveTo(qEndX, qEndY);
    ctx.lineTo(projEndX, projEndY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    // Draw projection point
    ctx.fillStyle = COLORS.math.operation;
    ctx.beginPath();
    ctx.arc(projEndX, projEndY, 6, 0, Math.PI * 2);
    ctx.fill();

    // Draw vectors
    drawVector(key, COLORS.key.main, keyLabel);
    drawVector(query, COLORS.query.main, queryLabel);

    // Draw info panel
    const padding = 20;
    const panelWidth = 220;
    const panelHeight = 140;
    const panelX = width - panelWidth - padding;
    const panelY = padding;

    // Panel background with slight transparency
    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.strokeStyle = COLORS.ui.border;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(panelX, panelY, panelWidth, panelHeight, 10);
    ctx.fill();
    ctx.stroke();

    // Panel content
    ctx.fillStyle = COLORS.ui.text;
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText('Geometric Interpretation', panelX + 10, panelY + 25);

    ctx.font = '13px monospace';
    ctx.fillStyle = COLORS.ui.textLight;

    const metrics = [
      `${queryLabel} · ${keyLabel} = ${dot.toFixed(3)}`,
      `|${queryLabel}| = ${qMag.toFixed(3)}`,
      `|${keyLabel}| = ${kMag.toFixed(3)}`,
      `cos(θ) = ${cosSim.toFixed(3)}`,
      ``,
      dot > 0 ? '✓ Aligned vectors' : dot < 0 ? '✗ Opposite vectors' : '⊥ Orthogonal',
    ];

    metrics.forEach((text, i) => {
      ctx.fillText(text, panelX + 10, panelY + 50 + i * 18);
    });

    // Add explanation at bottom
    ctx.fillStyle = COLORS.ui.text;
    ctx.font = '12px sans-serif';
    const explanation = 'Higher dot product = More similar';
    ctx.fillText(explanation, padding, height - padding);

  }, [query, key, queryLabel, keyLabel, animationPhase]);

  if (!query || !key || query.length < 2 || key.length < 2) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg border-2 border-gray-200">
        <p className="text-gray-500">Invalid vector data</p>
      </div>
    );
  }

  return (
    <div className="dot-product-visualization">
      <canvas
        ref={canvasRef}
        width={600}
        height={400}
        className="border-2 border-gray-200 rounded-lg shadow-lg"
        style={{ maxWidth: '100%', height: 'auto' }}
      />
      <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <p className="text-sm text-blue-900">
          <strong>Key Insight:</strong> The dot product measures how much two vectors point in the same direction.
          In attention, this tells us how relevant the Key is to the Query!
        </p>
      </div>
    </div>
  );
};

export default DotProductVisualization;
