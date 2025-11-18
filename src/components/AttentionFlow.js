import React, { useEffect, useRef, useState } from 'react';
import { COLORS, attentionToColor } from '../utils/colorUtils';

/**
 * AttentionFlow - Beautiful particle-based flow visualization
 * Shows information flowing between tokens based on attention weights
 * Inspired by 3Blue1Brown's animation style
 */
const AttentionFlow = ({
  tokens,
  attentionWeights,
  selectedToken = null,
  animate = true,
  showAllConnections = false,
  width = 800,
  height = 300,
}) => {
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);
  const [hoveredToken, setHoveredToken] = useState(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !tokens || tokens.length === 0) return;

    const ctx = canvas.getContext('2d');
    const padding = 60;
    const tokenY = height / 2;
    const tokenSpacing = (width - 2 * padding) / (tokens.length - 1 || 1);

    // Token positions
    const tokenPositions = tokens.map((_, i) => ({
      x: padding + i * tokenSpacing,
      y: tokenY,
    }));

    // Particle system
    class Particle {
      constructor(fromIdx, toIdx, weight) {
        this.from = tokenPositions[fromIdx];
        this.to = tokenPositions[toIdx];
        this.weight = weight;
        this.progress = Math.random(); // Start at random positions
        this.speed = 0.005 + weight * 0.01; // Faster for higher weights
        this.size = 2 + weight * 4;
        this.opacity = 0.3 + weight * 0.7;

        // Curve control point for smooth arc
        const midX = (this.from.x + this.to.x) / 2;
        const midY = this.from.y - 50 - weight * 50; // Higher arc for stronger attention
        this.controlPoint = { x: midX, y: midY };

        // Color based on weight
        this.color = attentionToColor(weight, this.opacity);
      }

      update() {
        this.progress += this.speed;
        if (this.progress > 1) {
          this.progress = 0; // Loop
        }
      }

      draw(ctx) {
        // Quadratic bezier curve position
        const t = this.progress;
        const oneMinusT = 1 - t;

        const x =
          oneMinusT * oneMinusT * this.from.x +
          2 * oneMinusT * t * this.controlPoint.x +
          t * t * this.to.x;

        const y =
          oneMinusT * oneMinusT * this.from.y +
          2 * oneMinusT * t * this.controlPoint.y +
          t * t * this.to.y;

        // Draw particle with glow
        ctx.save();
        ctx.shadowBlur = 15;
        ctx.shadowColor = this.color;
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(x, y, this.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }

    // Generate particles based on attention weights
    const generateParticles = () => {
      const particles = [];
      const sourceIdx = selectedToken !== null ? selectedToken : (hoveredToken !== null ? hoveredToken : null);

      if (sourceIdx === null && !showAllConnections) {
        return particles;
      }

      if (sourceIdx !== null) {
        // Show connections from selected/hovered token
        const weights = attentionWeights[sourceIdx] || [];
        weights.forEach((weight, targetIdx) => {
          if (weight > 0.05 && targetIdx !== sourceIdx) {
            // Create multiple particles for stronger connections
            const particleCount = Math.ceil(weight * 5);
            for (let i = 0; i < particleCount; i++) {
              particles.push(new Particle(sourceIdx, targetIdx, weight));
            }
          }
        });
      } else if (showAllConnections) {
        // Show all significant connections
        attentionWeights.forEach((weights, fromIdx) => {
          weights.forEach((weight, toIdx) => {
            if (weight > 0.2 && toIdx !== fromIdx) {
              const particleCount = Math.ceil(weight * 3);
              for (let i = 0; i < particleCount; i++) {
                particles.push(new Particle(fromIdx, toIdx, weight));
              }
            }
          });
        });
      }

      return particles;
    };

    // Draw connection curves
    const drawConnections = () => {
      const sourceIdx = selectedToken !== null ? selectedToken : hoveredToken;
      if (sourceIdx === null) return;

      const weights = attentionWeights[sourceIdx] || [];
      weights.forEach((weight, targetIdx) => {
        if (weight > 0.05 && targetIdx !== sourceIdx) {
          const from = tokenPositions[sourceIdx];
          const to = tokenPositions[targetIdx];

          const midX = (from.x + to.x) / 2;
          const midY = from.y - 50 - weight * 50;

          // Draw curve
          ctx.save();
          ctx.strokeStyle = attentionToColor(weight, 0.2);
          ctx.lineWidth = 1 + weight * 3;
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.quadraticCurveTo(midX, midY, to.x, to.y);
          ctx.stroke();
          ctx.restore();
        }
      });
    };

    // Draw tokens
    const drawTokens = () => {
      tokenPositions.forEach((pos, i) => {
        const isSource = i === selectedToken || i === hoveredToken;
        const isTarget =
          (selectedToken !== null && attentionWeights[selectedToken]?.[i] > 0.05) ||
          (hoveredToken !== null && attentionWeights[hoveredToken]?.[i] > 0.05);

        // Token circle
        ctx.save();

        // Glow for active tokens
        if (isSource) {
          ctx.shadowBlur = 25;
          ctx.shadowColor = COLORS.query.main;
        } else if (isTarget) {
          ctx.shadowBlur = 15;
          ctx.shadowColor = COLORS.key.main;
        }

        // Circle
        ctx.fillStyle = isSource
          ? COLORS.query.main
          : isTarget
          ? COLORS.key.light
          : COLORS.ui.border;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, isSource ? 16 : 12, 0, Math.PI * 2);
        ctx.fill();

        // Border
        ctx.strokeStyle = isSource
          ? COLORS.query.dark
          : isTarget
          ? COLORS.key.main
          : COLORS.ui.border;
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.restore();

        // Token label
        ctx.fillStyle = COLORS.ui.text;
        ctx.font = isSource ? 'bold 14px sans-serif' : '13px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(tokens[i], pos.x, pos.y + 35);

        // Attention weight label for targets
        if (isTarget && selectedToken !== null) {
          const weight = attentionWeights[selectedToken][i];
          ctx.fillStyle = COLORS.attention.high;
          ctx.font = 'bold 11px sans-serif';
          ctx.fillText(`${Math.round(weight * 100)}%`, pos.x, pos.y - 25);
        }
      });
    };

    // Main animation loop
    let lastTime = Date.now();
    const animate = (currentTime) => {
      const deltaTime = currentTime - lastTime;
      lastTime = currentTime;

      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Background gradient
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, '#FAFBFC');
      gradient.addColorStop(1, '#F0F4F8');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      // Draw connection curves
      drawConnections();

      // Update and draw particles
      particlesRef.current.forEach((particle) => {
        particle.update();
        particle.draw(ctx);
      });

      // Draw tokens on top
      drawTokens();

      // Title
      if (selectedToken !== null || hoveredToken !== null) {
        const sourceIdx = selectedToken !== null ? selectedToken : hoveredToken;
        ctx.fillStyle = COLORS.ui.text;
        ctx.font = 'bold 16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(
          `"${tokens[sourceIdx]}" is paying attention to:`,
          width / 2,
          30
        );
      } else {
        ctx.fillStyle = COLORS.ui.textLight;
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Click or hover over a token to see its attention flow', width / 2, 30);
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    // Initialize particles
    particlesRef.current = generateParticles();

    // Start animation
    if (animate) {
      animationFrameRef.current = requestAnimationFrame(animate);
    }

    // Cleanup
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [tokens, attentionWeights, selectedToken, hoveredToken, showAllConnections, width, height, animate]);

  // Regenerate particles when selection changes
  useEffect(() => {
    if (!tokens || !attentionWeights) return;

    const padding = 60;
    const tokenSpacing = (width - 2 * padding) / (tokens.length - 1 || 1);
    const tokenY = height / 2;

    const tokenPositions = tokens.map((_, i) => ({
      x: padding + i * tokenSpacing,
      y: tokenY,
    }));

    class Particle {
      constructor(fromIdx, toIdx, weight) {
        this.from = tokenPositions[fromIdx];
        this.to = tokenPositions[toIdx];
        this.weight = weight;
        this.progress = Math.random();
        this.speed = 0.005 + weight * 0.01;
        this.size = 2 + weight * 4;
        this.opacity = 0.3 + weight * 0.7;

        const midX = (this.from.x + this.to.x) / 2;
        const midY = this.from.y - 50 - weight * 50;
        this.controlPoint = { x: midX, y: midY };

        this.color = attentionToColor(weight, this.opacity);
      }

      update() {
        this.progress += this.speed;
        if (this.progress > 1) {
          this.progress = 0;
        }
      }

      draw(ctx) {
        const t = this.progress;
        const oneMinusT = 1 - t;

        const x =
          oneMinusT * oneMinusT * this.from.x +
          2 * oneMinusT * t * this.controlPoint.x +
          t * t * this.to.x;

        const y =
          oneMinusT * oneMinusT * this.from.y +
          2 * oneMinusT * t * this.controlPoint.y +
          t * t * this.to.y;

        ctx.save();
        ctx.shadowBlur = 15;
        ctx.shadowColor = this.color;
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(x, y, this.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }

    const generateParticles = () => {
      const particles = [];
      const sourceIdx = selectedToken !== null ? selectedToken : (hoveredToken !== null ? hoveredToken : null);

      if (sourceIdx === null && !showAllConnections) {
        return particles;
      }

      if (sourceIdx !== null) {
        const weights = attentionWeights[sourceIdx] || [];
        weights.forEach((weight, targetIdx) => {
          if (weight > 0.05 && targetIdx !== sourceIdx) {
            const particleCount = Math.ceil(weight * 5);
            for (let i = 0; i < particleCount; i++) {
              particles.push(new Particle(sourceIdx, targetIdx, weight));
            }
          }
        });
      }

      return particles;
    };

    particlesRef.current = generateParticles();
  }, [selectedToken, hoveredToken, tokens, attentionWeights, showAllConnections, width, height]);

  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas || !tokens) return;

    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) * canvas.width) / rect.width;
    const y = ((e.clientY - rect.top) * canvas.height) / rect.height;

    const padding = 60;
    const tokenSpacing = (width - 2 * padding) / (tokens.length - 1 || 1);
    const tokenY = height / 2;

    // Check if click is near any token
    tokens.forEach((_, i) => {
      const tokenX = padding + i * tokenSpacing;
      const distance = Math.sqrt((x - tokenX) ** 2 + (y - tokenY) ** 2);

      if (distance < 20) {
        // Clicked on token
        if (selectedToken === i) {
          // Deselect
          setHoveredToken(null);
        } else {
          setHoveredToken(i);
        }
      }
    });
  };

  return (
    <div className="attention-flow">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="border-2 border-gray-200 rounded-lg shadow-lg cursor-pointer"
        style={{ maxWidth: '100%', height: 'auto' }}
        onClick={handleCanvasClick}
      />
      <div className="mt-4 text-center text-sm text-gray-600">
        <p>Click on a token to see its attention flow • Particles flow from source to targets</p>
      </div>
    </div>
  );
};

export default AttentionFlow;
