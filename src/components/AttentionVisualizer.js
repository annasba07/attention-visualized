import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, ChevronRight, ChevronDown, Eye, Lightbulb, Zap } from 'lucide-react';

const AttentionVisualizer = () => {
  const [inputText, setInputText] = useState("The cat sat on the mat");
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [expandedSections, setExpandedSections] = useState({});
  const [hoveredToken, setHoveredToken] = useState(null);
  const [selectedToken, setSelectedToken] = useState(null);
  const [showMath, setShowMath] = useState(false);
  const [animationPhase, setAnimationPhase] = useState(0);
  const svgRef = useRef(null);

  // Model dimensions
  const dModel = 4;
  const dK = 2;
  const dV = 2;

  // Parse tokens
  const tokens = inputText.trim().split(' ').filter(t => t.length > 0);
  const seqLen = tokens.length;

  // Initialize embeddings with more meaningful patterns
  const embeddings = tokens.length > 0 ? tokens.map((token, i) => {
    // Create embeddings that relate to word meaning/position
    const base = [
      Math.sin(i * 0.5) * 0.8 + 0.2,
      Math.cos(i * 0.3) * 0.6 + 0.4,
      token.length * 0.1,
      (token.charCodeAt(0) % 10) * 0.1
    ];
    return base.map(val => Math.round(val * 100) / 100);
  }) : [];

  // Weight matrices (simplified for visualization)
  const WQ = [[0.5, -0.3], [0.2, 0.8], [-0.4, 0.6], [0.7, -0.1]];
  const WK = [[0.3, 0.9], [-0.2, 0.4], [0.8, -0.5], [0.1, 0.7]];
  const WV = [[0.6, 0.2], [0.4, -0.8], [-0.3, 0.5], [0.9, 0.1]];

  // Matrix operations
  const matMul = (A, B) => {
    if (!A || !B || !A.length || !B.length || !A[0] || !B[0]) return [];
    const result = Array.from({length: A.length}, () => 
      Array.from({length: B[0].length}, () => 0)
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

  const Q = embeddings.length > 0 ? matMul(embeddings, WQ) : [];
  const K = embeddings.length > 0 ? matMul(embeddings, WK) : [];
  const V = embeddings.length > 0 ? matMul(embeddings, WV) : [];

  const scores = Q.length > 0 && K.length > 0 && K[0] ? matMul(Q, K[0].map((_, i) => K.map(row => row[i]))) : [];
  const scaledScores = scores.length > 0 ? scores.map(row => 
    row.map(val => Math.round(val / Math.sqrt(dK) * 100) / 100)
  ) : [];

  const softmax = (arr) => {
    if (!arr.length) return [];
    const maxVal = Math.max(...arr);
    const exp = arr.map(x => Math.exp(x - maxVal));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(x => Math.round(x / sum * 1000) / 1000);
  };

  const attentionWeights = scaledScores.length > 0 ? scaledScores.map(row => softmax(row)) : [];
  const output = attentionWeights.length > 0 ? matMul(attentionWeights, V) : [];

  const steps = [
    {
      title: "Step 1: Words as Vectors",
      subtitle: "Every word becomes a list of numbers",
      description: "Just like how we might describe a person with height, weight, age, etc., each word gets numbers that capture its 'meaning'",
      component: "embeddings",
      metaphor: "🏷️ Think of this like giving each word a unique ID card with several numbers on it"
    },
    {
      title: "Step 2: Three Questions for Each Word",
      subtitle: "What am I looking for? What do I offer? What do I contribute?",
      description: "Each word gets transformed into three roles: Query (what it wants), Key (what it offers), Value (what it gives)",
      component: "qkv",
      metaphor: "🔍 Like at a networking event: what you're seeking, what you're offering, what you'd share if someone's interested"
    },
    {
      title: "Step 3: Measuring Compatibility",
      subtitle: "How well do words match with each other?",
      description: "We compare what each word is looking for with what every other word offers",
      component: "scores",
      metaphor: "📊 Like a compatibility test - higher scores mean better matches!"
    },
    {
      title: "Step 4: Attention Spotlight",
      subtitle: "Turn compatibility into focus",
      description: "Convert raw scores into a 'spotlight' - each word decides how much attention to pay to every other word",
      component: "attention",
      metaphor: "💡 Like adjusting the brightness of multiple spotlights in a theater"
    },
    {
      title: "Step 5: Gathering Information",
      subtitle: "Each word collects what it needs",
      description: "Using the attention weights, each word gathers information from all the words it's paying attention to",
      component: "output",
      metaphor: "🎯 Like a reporter gathering quotes from different sources, weighted by how relevant each source is"
    }
  ];

  // Animation control
  useEffect(() => {
    let interval;
    if (isPlaying) {
      interval = setInterval(() => {
        setAnimationPhase(phase => (phase + 1) % 4);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  useEffect(() => {
    let stepInterval;
    if (isPlaying) {
      stepInterval = setInterval(() => {
        setCurrentStep(current => current < steps.length - 1 ? current + 1 : 0);
      }, 4000);
    }
    return () => clearInterval(stepInterval);
  }, [isPlaying, steps.length]);

  const generateAttentionStory = (word, allTokens, attentionRow, wordIndex) => {
    if (!attentionRow || attentionRow.length === 0 || !allTokens || allTokens.length === 0) {
      return `"${word}" is still calculating its attention patterns...`;
    }
    
    const sortedAttention = attentionRow
      .map((weight, idx) => ({ 
        weight: weight || 0, 
        idx, 
        token: allTokens[idx] || 'unknown' 
      }))
      .filter(item => item.token !== 'unknown')
      .sort((a, b) => b.weight - a.weight);
    
    if (sortedAttention.length < 2) {
      return `"${word}" is still calculating its attention patterns...`;
    }
    
    const topAttention = sortedAttention.slice(0, 3);
    
    if (topAttention[0].idx === wordIndex) {
      return `"${word}" is mostly focused on itself (${Math.round(topAttention[0].weight * 100)}%), but also pays some attention to "${topAttention[1].token}" (${Math.round(topAttention[1].weight * 100)}%).`;
    } else {
      return `"${word}" is most interested in "${topAttention[0].token}" (${Math.round(topAttention[0].weight * 100)}% of its attention), followed by "${topAttention[1].token}" (${Math.round(topAttention[1].weight * 100)}%).`;
    }
  };

  // Interactive attention visualization
  const AttentionGraph = ({ weights, tokenList, interactive = true }) => {
    const displayToken = hoveredToken !== null ? hoveredToken : selectedToken;
    
    return (
      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-xl border-2 border-blue-200">
        <h4 className="font-bold text-lg mb-4 flex items-center gap-2">
          <Eye className="text-blue-600" />
          Attention Spotlight
        </h4>
        <p className="text-sm text-gray-600 mb-4">
          {displayToken !== null 
            ? `"${tokenList[displayToken]}" is paying attention to:` 
            : "Hover over a word to see what it pays attention to!"
          }
        </p>
        
        <div className="flex flex-wrap gap-3 mb-6">
          {tokenList.map((token, i) => {
            const attention = displayToken !== null ? (weights[displayToken] && weights[displayToken][i] ? weights[displayToken][i] : 0) : 0;
            const isActive = displayToken === i;
            const isHovered = hoveredToken === i;
            const baseOpacity = displayToken !== null ? Math.max(0.4, Math.min(1, attention + 0.3)) : 1;
            
            return (
              <div
                key={i}
                className={`relative px-4 py-3 rounded-lg border-2 cursor-pointer transition-all duration-200 overflow-hidden ${
                  isActive 
                    ? 'bg-blue-600 text-white border-blue-600 shadow-lg' 
                    : 'bg-white border-gray-300 hover:border-blue-400 hover:shadow-md'
                }`}
                style={{
                  opacity: baseOpacity,
                  transform: isActive ? 'scale(1.05)' : isHovered ? 'scale(1.02)' : 'scale(1)'
                }}
                onMouseEnter={() => {
                  if (interactive) {
                    setHoveredToken(i);
                  }
                }}
                onMouseLeave={() => {
                  if (interactive) {
                    setHoveredToken(null);
                  }
                }}
                onClick={() => {
                  if (interactive) {
                    setSelectedToken(selectedToken === i ? null : i);
                    setHoveredToken(null);
                  }
                }}
              >
                <div className="font-medium">{token}</div>
                {displayToken !== null && !isActive && attention > 0 && (
                  <div className="text-xs mt-1 opacity-75">
                    {Math.round(attention * 100)}% attention
                  </div>
                )}
                {displayToken !== null && attention > 0 && (
                  <div 
                    className="absolute -top-2 -right-2 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold border-2 border-white"
                    style={{
                      backgroundColor: attention > 0.5 ? '#3b82f6' : '#93c5fd',
                      color: attention > 0.3 ? 'white' : '#1f2937'
                    }}
                  >
                    {Math.round(attention * 10)}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {displayToken !== null && (
          <div className="bg-white p-4 rounded-lg border">
            <h5 className="font-semibold mb-2">Attention Story for "{tokenList[displayToken]}":</h5>
            <p className="text-sm text-gray-700">
              {generateAttentionStory(tokenList[displayToken], tokenList, weights[displayToken] || [], displayToken)}
            </p>
          </div>
        )}
      </div>
    );
  };

  // Animated attention flow
  const AttentionFlow = ({ weights, tokenList }) => {
    const canvasRef = useRef(null);
    
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas || !weights || !weights.length || selectedToken === null) return;
      
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      
      // Clear canvas
      ctx.clearRect(0, 0, width, height);
      
      // Draw flowing connections
      const tokenPositions = tokenList.map((_, i) => ({
        x: (width / (tokenList.length + 1)) * (i + 1),
        y: height / 2
      }));
      
      const selectedWeights = weights[selectedToken];
      if (selectedWeights && selectedWeights.length > 0) {
        selectedWeights.forEach((weight, targetIdx) => {
          if (weight > 0.05 && targetIdx !== selectedToken && targetIdx < tokenPositions.length) {
            const start = tokenPositions[selectedToken];
            const end = tokenPositions[targetIdx];
            
            // Animated flowing line
            const phase = (animationPhase * Math.PI) / 2;
            
            ctx.beginPath();
            ctx.strokeStyle = `rgba(59, 130, 246, ${Math.min(1, weight * 2)})`;
            ctx.lineWidth = Math.max(2, weight * 8);
            
            // Create flowing effect
            const midX = (start.x + end.x) / 2;
            const midY = (start.y + end.y) / 2 - 30 * Math.sin(phase + weight);
            
            ctx.moveTo(start.x, start.y);
            ctx.quadraticCurveTo(midX, midY, end.x, end.y);
            ctx.stroke();
            
            // Add flowing particles
            const t = (Math.sin(phase + weight) + 1) / 2;
            const particleX = start.x + (end.x - start.x) * t;
            const particleY = start.y + (midY - start.y) * 2 * t * (1 - t);
            
            ctx.beginPath();
            ctx.fillStyle = '#3b82f6';
            ctx.arc(particleX, particleY, Math.max(2, weight * 6), 0, Math.PI * 2);
            ctx.fill();
            
            // Add attention value label
            ctx.fillStyle = '#1f2937';
            ctx.font = '12px sans-serif';
            ctx.fillText(`${Math.round(weight * 100)}%`, end.x + 5, end.y - 10);
          }
        });
      }
    }, [weights, tokenList, selectedToken, animationPhase]);
    
    return (
      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="font-semibold mb-2 flex items-center gap-2">
          <Zap className="text-yellow-500" />
          Attention Flow Animation
        </h4>
        <p className="text-sm text-gray-600 mb-3">
          {selectedToken !== null 
            ? `Showing attention flow from "${tokenList[selectedToken]}"` 
            : "Click on a word above to see animated attention flow!"
          }
        </p>
        <canvas 
          ref={canvasRef} 
          width={600} 
          height={150}
          className="border rounded bg-white w-full"
          style={{ maxWidth: '100%' }}
        />
      </div>
    );
  };

  const SimpleMatrix = ({ matrix, title, description, colorCode = false }) => {
    if (!matrix || !matrix.length || !matrix[0]) {
      return (
        <div className="bg-white border-2 border-gray-200 rounded-lg p-4">
          <h4 className="font-semibold mb-2">{title}</h4>
          {description && <p className="text-sm text-gray-600 mb-3">{description}</p>}
          <p className="text-gray-500 italic">No data to display</p>
        </div>
      );
    }

    return (
      <div className="bg-white border-2 border-gray-200 rounded-lg p-4">
        <h4 className="font-semibold mb-2">{title}</h4>
        {description && <p className="text-sm text-gray-600 mb-3">{description}</p>}
        <div className="grid gap-2" style={{gridTemplateColumns: `repeat(${matrix[0]?.length || 1}, 1fr)`}}>
          {matrix.map((row, i) => 
            row && row.map((val, j) => {
              const intensity = colorCode ? Math.abs(val) : 0;
              return (
                <div
                  key={`${i}-${j}`}
                  className="px-2 py-2 text-center rounded font-mono text-sm border"
                  style={{
                    backgroundColor: colorCode 
                      ? `rgba(59, 130, 246, ${Math.min(1, intensity / 2)})` 
                      : 'white',
                    color: colorCode && intensity > 1 ? 'white' : 'black'
                  }}
                >
                  {typeof val === 'number' ? val.toFixed(2) : val}
                </div>
              );
            })
          )}
        </div>
      </div>
    );
  };

  const StepContent = () => {
    const step = steps[currentStep];
    
    return (
      <div className="space-y-6">
        {/* Metaphor card */}
        <div className="bg-gradient-to-r from-purple-100 to-pink-100 p-4 rounded-lg border-2 border-purple-200">
          <div className="flex items-center gap-2 mb-2">
            <Lightbulb className="text-purple-600" />
            <h4 className="font-semibold text-purple-800">Think of it like this:</h4>
          </div>
          <p className="text-purple-700">{step.metaphor}</p>
        </div>

        {/* Step-specific content */}
        {step.component === "embeddings" && (
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-4">
              <h4 className="font-semibold text-lg">Our Words:</h4>
              <div className="flex flex-wrap gap-2">
                {tokens.map((token, i) => (
                  <span key={i} className="px-4 py-2 bg-blue-100 text-blue-800 rounded-lg font-medium">
                    {token}
                  </span>
                ))}
              </div>
              {showMath && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h5 className="font-semibold mb-2">📐 The Math:</h5>
                  <p className="text-sm mb-2">Each word gets converted to a {dModel}-dimensional vector:</p>
                  <div className="font-mono text-sm bg-white p-2 rounded border">
                    embedding[i] = [sin(i×0.5)×0.8+0.2, cos(i×0.3)×0.6+0.4, word_length×0.1, char_code×0.1]
                  </div>
                </div>
              )}
            </div>
            {showMath && embeddings.length > 0 && (
              <SimpleMatrix 
                matrix={embeddings.map((emb, i) => [tokens[i], ...emb])}
                title="Word Vectors"
                description="Each word as 4 numbers"
                colorCode={true}
              />
            )}
          </div>
        )}

        {step.component === "qkv" && (
          <div>
            <div className="grid md:grid-cols-3 gap-4 mb-4">
              <div className="bg-red-50 p-4 rounded-lg border-2 border-red-200">
                <h4 className="font-semibold text-red-800 mb-2">🔍 Query: "What am I looking for?"</h4>
                <p className="text-sm text-red-700">Each word asks: "What information do I need from other words?"</p>
              </div>
              <div className="bg-green-50 p-4 rounded-lg border-2 border-green-200">
                <h4 className="font-semibold text-green-800 mb-2">🔑 Key: "What do I offer?"</h4>
                <p className="text-sm text-green-700">Each word advertises: "Here's what I can provide!"</p>
              </div>
              <div className="bg-blue-50 p-4 rounded-lg border-2 border-blue-200">
                <h4 className="font-semibold text-blue-800 mb-2">💎 Value: "Here's my contribution"</h4>
                <p className="text-sm text-blue-700">Each word says: "If you pay attention to me, here's what I'll give you!"</p>
              </div>
            </div>
            {showMath && Q.length > 0 && (
              <div className="space-y-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h5 className="font-semibold mb-2">📐 The Math:</h5>
                  <div className="space-y-2 text-sm">
                    <p><strong>Query (Q):</strong> Q = Embeddings × W_Q</p>
                    <p><strong>Key (K):</strong> K = Embeddings × W_K</p>
                    <p><strong>Value (V):</strong> V = Embeddings × W_V</p>
                    <p className="text-gray-600">Each embedding vector gets multiplied by learned weight matrices to create the three different "views"</p>
                  </div>
                </div>
                <div className="grid md:grid-cols-3 gap-4">
                  <SimpleMatrix matrix={Q.map((q, i) => [tokens[i], ...q])} title="Queries (Q)" colorCode={true} />
                  <SimpleMatrix matrix={K.map((k, i) => [tokens[i], ...k])} title="Keys (K)" colorCode={true} />
                  <SimpleMatrix matrix={V.map((v, i) => [tokens[i], ...v])} title="Values (V)" colorCode={true} />
                </div>
              </div>
            )}
          </div>
        )}

        {step.component === "scores" && (
          <div className="space-y-4">
            <div className="bg-yellow-50 p-4 rounded-lg border-2 border-yellow-200">
              <h4 className="font-semibold text-yellow-800 mb-2">🎯 Compatibility Matching</h4>
              <p className="text-yellow-700">We're checking: "How well does what word A is looking for match with what word B offers?"</p>
            </div>
            {showMath && (
              <div className="space-y-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h5 className="font-semibold mb-2">📐 The Math:</h5>
                  <div className="space-y-2 text-sm">
                    <p><strong>Step 1:</strong> Compute raw scores: Scores = Q × K^T</p>
                    <p><strong>Step 2:</strong> Scale by √d_k: Scaled = Scores ÷ √{dK}</p>
                    <p className="text-gray-600">We transpose K so each query can be compared with every key</p>
                  </div>
                </div>
                {scores.length > 0 && (
                  <SimpleMatrix 
                    matrix={scores}
                    title="Raw Compatibility Scores (Q × K^T)"
                    description="Higher = better match"
                    colorCode={true}
                  />
                )}
                {scaledScores.length > 0 && (
                  <SimpleMatrix 
                    matrix={scaledScores}
                    title={`Scaled Scores (÷ √${dK})`}
                    description="Prevents very large values that could cause problems"
                    colorCode={true}
                  />
                )}
              </div>
            )}
          </div>
        )}

        {step.component === "attention" && (
          <div className="space-y-4">
            {showMath && scaledScores.length > 0 && (
              <div className="bg-gray-50 p-4 rounded-lg mb-4">
                <h5 className="font-semibold mb-2">📐 The Math:</h5>
                <div className="space-y-2 text-sm">
                  <p><strong>Softmax Formula:</strong> softmax(x_i) = e^x_i / Σ(e^x_j)</p>
                  <p><strong>What it does:</strong> Converts raw scores into probabilities that sum to 1</p>
                  <p className="text-gray-600">This ensures each word has a probability distribution over all other words</p>
                </div>
              </div>
            )}
            <AttentionGraph weights={attentionWeights} tokenList={tokens} />
            <AttentionFlow weights={attentionWeights} tokenList={tokens} />
            <div className="bg-blue-50 p-4 rounded-lg border-2 border-blue-200">
              <h4 className="font-semibold text-blue-800 mb-2">👆 Try This!</h4>
              <p className="text-blue-700">Hover over the words above to see what each one pays attention to, then click to lock the view!</p>
            </div>
          </div>
        )}

        {step.component === "output" && (
          <div className="space-y-4">
            <div className="bg-green-50 p-4 rounded-lg border-2 border-green-200">
              <h4 className="font-semibold text-green-800 mb-2">🎉 Information Gathering Complete!</h4>
              <p className="text-green-700">Each word now contains not just its own information, but a weighted blend of information from all the words it paid attention to!</p>
            </div>
            {showMath && output.length > 0 && (
              <div className="space-y-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h5 className="font-semibold mb-2">📐 The Math:</h5>
                  <div className="space-y-2 text-sm">
                    <p><strong>Final Output:</strong> Output = Attention_Weights × V</p>
                    <p><strong>What it does:</strong> Each word gets a weighted combination of all Value vectors</p>
                    <p className="text-gray-600">The attention weights determine how much each word contributes to the final representation</p>
                  </div>
                </div>
                <SimpleMatrix 
                  matrix={output.map((out, i) => [tokens[i], ...out])}
                  title="Enhanced Word Representations"
                  description="Each word now knows about the context around it"
                  colorCode={true}
                />
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
      <div className="bg-white rounded-2xl shadow-xl p-8">
        {tokens.length === 0 ? (
          // Empty state
          <div className="text-center py-16">
            <h1 className="text-4xl font-bold mb-4 text-gray-800">
              How AI Pays Attention
            </h1>
            <p className="text-xl text-gray-600 mb-6">
              Discover how AI models like ChatGPT understand which words are important to each other
            </p>
            <div className="mb-6">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="px-4 py-3 border-2 border-gray-300 rounded-lg w-full max-w-md text-lg focus:border-blue-500 focus:outline-none"
                placeholder="Type some words to get started..."
                autoFocus
              />
            </div>
            <div className="bg-blue-50 p-6 rounded-lg border-2 border-blue-200 max-w-md mx-auto">
              <h3 className="font-bold text-lg mb-3 text-blue-800">💡 Try These Examples:</h3>
              <div className="space-y-2">
                {[
                  "The cat sat on the mat",
                  "She loves reading books", 
                  "Coffee tastes great in morning",
                  "The dog chased the ball"
                ].map((example, i) => (
                  <button
                    key={i}
                    onClick={() => setInputText(example)}
                    className="w-full text-left p-3 bg-white rounded-lg border-2 border-blue-200 hover:border-blue-400 transition-all"
                  >
                    <span className="font-medium text-blue-800">"{example}"</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          // Main content when we have tokens
          <>
            <div className="mb-8">
              <h1 className="text-4xl font-bold mb-4 text-gray-800">
                How AI Pays Attention
              </h1>
              <p className="text-xl text-gray-600 mb-6">
                Discover how AI models like ChatGPT understand which words are important to each other
              </p>
              
              <div className="flex flex-wrap gap-4 items-center mb-6">
                <input
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  className="px-4 py-3 border-2 border-gray-300 rounded-lg flex-1 max-w-md text-lg focus:border-blue-500 focus:outline-none"
                  placeholder="Try: The cat sat on the mat"
                />
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg flex items-center gap-2 font-medium hover:bg-blue-700 transition-colors"
                >
                  {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                  {isPlaying ? 'Pause Tour' : 'Start Tour'}
                </button>
                <button
                  onClick={() => setShowMath(!showMath)}
                  className={`px-4 py-3 rounded-lg flex items-center gap-2 font-medium transition-all ${
                    showMath ? 'bg-gray-600 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Show Math
                </button>
              </div>

              {/* Progress bar */}
              <div className="w-full bg-gray-200 rounded-full h-3 mb-6">
                <div 
                  className="bg-blue-600 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
                />
              </div>

              {/* Step navigation */}
              <div className="flex flex-wrap gap-2 mb-6">
                {steps.map((step, index) => (
                  <button
                    key={index}
                    onClick={() => setCurrentStep(index)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      index === currentStep 
                        ? 'bg-blue-600 text-white shadow-lg' 
                        : index < currentStep 
                          ? 'bg-green-100 text-green-800 hover:bg-green-200'
                          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    {index + 1}. {step.title.split(':')[0]}
                  </button>
                ))}
              </div>
            </div>

            {/* Current Step Display */}
            <div className="bg-gradient-to-br from-gray-50 to-blue-50 border-2 border-blue-200 rounded-xl p-8 mb-6">
              <div className="flex items-start gap-4 mb-6">
                <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold text-xl flex-shrink-0">
                  {currentStep + 1}
                </div>
                <div className="flex-1">
                  <h2 className="text-2xl font-bold mb-2">{steps[currentStep].title}</h2>
                  <h3 className="text-lg text-blue-600 font-medium mb-2">{steps[currentStep].subtitle}</h3>
                  <p className="text-gray-700">{steps[currentStep].description}</p>
                </div>
              </div>
              
              <StepContent />
            </div>

            {/* Quick tips */}
            <div className="bg-gradient-to-r from-yellow-50 to-orange-50 border-2 border-yellow-200 rounded-xl p-6">
              <h3 className="font-bold text-lg mb-3 text-yellow-800">💡 Try These Examples:</h3>
              <div className="grid md:grid-cols-2 gap-4">
                {[
                  "The red car drove fast",
                  "She loves reading books",
                  "Coffee tastes great in morning",
                  "The dog chased the ball"
                ].map((example, i) => (
                  <button
                    key={i}
                    onClick={() => setInputText(example)}
                    className="text-left p-3 bg-white rounded-lg border-2 border-yellow-200 hover:border-yellow-400 transition-all"
                  >
                    <span className="font-medium text-yellow-800">"{example}"</span>
                  </button>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default AttentionVisualizer;