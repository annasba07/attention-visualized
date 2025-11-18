import React, { useState, useEffect, useMemo } from 'react';
import { Play, Pause, Lightbulb, BookOpen } from 'lucide-react';
import DotProductVisualization from './DotProductVisualization';
import MatrixHeatmap from './MatrixHeatmap';
import InteractiveControls from './InteractiveControls';
import AttentionFlow from './AttentionFlow';
import CalculationBreakdown from './CalculationBreakdown';
import {
  generateEmbeddings,
  initWeightMatrix,
  matMul,
  transpose,
  softmax,
  computeAttentionOutput,
} from '../utils/mathUtils';

/**
 * AttentionVisualizer - Main component (3Blue1Brown-inspired)
 * Beautiful, interactive visualization of transformer attention
 */
const AttentionVisualizer = () => {
  // State
  const [inputText, setInputText] = useState('The cat sat on the mat');
  const [currentStep, setCurrentStep] = useState(0);
  const [showMath, setShowMath] = useState(true);
  const [autoPlay, setAutoPlay] = useState(false);
  const [temperature, setTemperature] = useState(1.0);

  // Model dimensions
  const dModel = 4;
  const dK = 2;
  const dV = 2;

  // Parse tokens
  const tokens = useMemo(() => {
    return inputText.trim().split(' ').filter((t) => t.length > 0);
  }, [inputText]);

  // Compute embeddings and transformations
  const embeddings = useMemo(() => generateEmbeddings(tokens, dModel), [tokens, dModel]);

  const WQ = useMemo(() => initWeightMatrix(dModel, dK, 1), [dModel, dK]);
  const WK = useMemo(() => initWeightMatrix(dModel, dK, 2), [dModel, dK]);
  const WV = useMemo(() => initWeightMatrix(dModel, dV, 3), [dModel, dV]);

  const Q = useMemo(() => matMul(embeddings, WQ), [embeddings, WQ]);
  const K = useMemo(() => matMul(embeddings, WK), [embeddings, WK]);
  const V = useMemo(() => matMul(embeddings, WV), [embeddings, WV]);

  const scores = useMemo(() => {
    if (!Q || !K || Q.length === 0 || K.length === 0) return [];
    const KT = transpose(K);
    return matMul(Q, KT);
  }, [Q, K]);

  const scaledScores = useMemo(() => {
    if (!scores || scores.length === 0) return [];
    return scores.map((row) =>
      row.map((val) => Math.round((val / Math.sqrt(dK)) * 100) / 100)
    );
  }, [scores, dK]);

  const attentionWeights = useMemo(() => {
    if (!scaledScores || scaledScores.length === 0) return [];
    return scaledScores.map((row) => {
      // Apply temperature
      const tempScaled = row.map((val) => val / temperature);
      return softmax(tempScaled);
    });
  }, [scaledScores, temperature]);

  const output = useMemo(() => {
    return computeAttentionOutput(attentionWeights, V);
  }, [attentionWeights, V]);

  // Auto-play effect
  useEffect(() => {
    if (!autoPlay) return;

    const interval = setInterval(() => {
      setCurrentStep((s) => (s + 1) % 5); // 5 total steps
    }, 5000);

    return () => clearInterval(interval);
  }, [autoPlay]);

  // Step definitions
  const steps = [
    {
      id: 'embeddings',
      title: 'Step 1: Words → Vectors',
      subtitle: 'Converting language to mathematics',
      description:
        'Each word is represented as a vector of numbers. This encoding captures semantic meaning in a form computers can process.',
      icon: '📊',
    },
    {
      id: 'qkv',
      title: 'Step 2: Query, Key, Value',
      subtitle: 'Three perspectives on each word',
      description:
        'We transform each word into three different "views": what it\'s looking for (Query), what it offers (Key), and what it contributes (Value).',
      icon: '🔑',
    },
    {
      id: 'scores',
      title: 'Step 3: Compatibility Scores',
      subtitle: 'Measuring similarity',
      description:
        'We compute how well each Query matches with each Key using the dot product—a geometric measure of alignment.',
      icon: '🎯',
    },
    {
      id: 'attention',
      title: 'Step 4: Attention Weights',
      subtitle: 'From scores to probabilities',
      description:
        'Softmax converts compatibility scores into a probability distribution, determining how much attention each word pays to others.',
      icon: '💡',
    },
    {
      id: 'output',
      title: 'Step 5: Weighted Combination',
      subtitle: 'Gathering contextual information',
      description:
        'Each word gathers information from others, weighted by attention. The result is context-aware representations.',
      icon: '✨',
    },
  ];

  const currentStepData = steps[currentStep];

  // Calculation breakdowns for each step
  const getCalculationSteps = () => {
    switch (currentStepData.id) {
      case 'embeddings':
        return [
          {
            label: 'Word Tokenization',
            description: 'Split input text into individual tokens',
            example: `Input: "${inputText}" → Tokens: [${tokens.map((t) => `"${t}"`).join(', ')}]`,
            insight: 'Each word becomes a discrete unit we can process independently.',
          },
          {
            label: 'Embedding Generation',
            description: `Map each token to a ${dModel}-dimensional vector`,
            formula: `embedding_i = [sin(i×0.5)×0.8+0.2, cos(i×0.3)×0.6+0.4, length/10, charCode/26]`,
            insight:
              'These vectors encode position and word properties. Real models use learned embeddings.',
          },
        ];

      case 'qkv':
        return [
          {
            label: 'Query Transformation',
            description: 'What is each word looking for?',
            formula: 'Q = Embeddings × W_Q',
            example: embeddings.length > 0 && `Result shape: ${Q.length} × ${Q[0]?.length || 0}`,
            insight: 'Query vectors represent what information each word needs from the context.',
          },
          {
            label: 'Key Transformation',
            description: 'What does each word offer?',
            formula: 'K = Embeddings × W_K',
            example: embeddings.length > 0 && `Result shape: ${K.length} × ${K[0]?.length || 0}`,
            insight: 'Key vectors advertise what information each word can provide.',
          },
          {
            label: 'Value Transformation',
            description: 'What will each word contribute?',
            formula: 'V = Embeddings × W_V',
            example: embeddings.length > 0 && `Result shape: ${V.length} × ${V[0]?.length || 0}`,
            insight:
              'Value vectors contain the actual information to be passed based on attention.',
          },
        ];

      case 'scores':
        return [
          {
            label: 'Dot Product Computation',
            description: 'Measure Query-Key similarity',
            formula: 'Scores = Q × K^T',
            insight:
              'Dot product is large when vectors point in similar directions—high similarity!',
          },
          {
            label: 'Scaling',
            description: 'Normalize by key dimension',
            formula: `Scaled = Scores / √${dK} = Scores / ${Math.sqrt(dK).toFixed(2)}`,
            insight:
              'Scaling prevents very large values that would make softmax too "sharp" and hard to train.',
          },
        ];

      case 'attention':
        return [
          {
            label: 'Temperature Adjustment',
            description: 'Control attention focus',
            formula: `Temp_Scores = Scaled_Scores / ${temperature.toFixed(2)}`,
            insight: 'Lower temperature → sharper attention. Higher temperature → smoother attention.',
          },
          {
            label: 'Softmax Normalization',
            description: 'Convert to probability distribution',
            formula: 'Attention[i,j] = exp(score[i,j]) / Σ_k exp(score[i,k])',
            insight:
              'Each row sums to 1.0, making it a valid probability distribution over all tokens.',
          },
        ];

      case 'output':
        return [
          {
            label: 'Weighted Sum',
            description: 'Combine Values according to attention',
            formula: 'Output = Attention × V',
            insight:
              'Each output vector is a weighted mixture of all Value vectors, with weights from attention.',
          },
          {
            label: 'Contextualization',
            description: 'Words now understand their context',
            insight:
              'Unlike the original embeddings, these outputs incorporate information from the entire sequence!',
          },
        ];

      default:
        return [];
    }
  };

  // Render step content
  const renderStepContent = () => {
    switch (currentStepData.id) {
      case 'embeddings':
        return (
          <div className="space-y-6">
            {/* Token display */}
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-6 rounded-xl border-2 border-blue-200">
              <h4 className="font-semibold text-lg mb-4 text-gray-800">Our Tokens:</h4>
              <div className="flex flex-wrap gap-3">
                {tokens.map((token, i) => (
                  <div
                    key={i}
                    className="px-5 py-3 bg-white rounded-lg border-2 border-blue-300 shadow-md hover:shadow-lg transition-shadow"
                  >
                    <div className="font-bold text-blue-800">{token}</div>
                    <div className="text-xs text-gray-500 mt-1">Token {i}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Matrix visualization */}
            {showMath && embeddings.length > 0 && (
              <MatrixHeatmap
                matrix={embeddings}
                rowLabels={tokens}
                colLabels={['d₀', 'd₁', 'd₂', 'd₃']}
                title="Embedding Matrix"
                description={`Each row is a ${dModel}-dimensional vector representing one token`}
                colorScheme="default"
                showValues={true}
                animate={true}
              />
            )}
          </div>
        );

      case 'qkv':
        return (
          <div className="space-y-6">
            {/* Concept cards */}
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-gradient-to-br from-red-50 to-red-100 p-5 rounded-xl border-2 border-red-200 shadow-md">
                <h4 className="font-bold text-red-800 mb-2 flex items-center gap-2">
                  <span className="text-2xl">🔍</span> Query (Q)
                </h4>
                <p className="text-sm text-red-700">
                  "What am I looking for?"
                  <br />
                  What information does this word need?
                </p>
              </div>
              <div className="bg-gradient-to-br from-cyan-50 to-cyan-100 p-5 rounded-xl border-2 border-cyan-200 shadow-md">
                <h4 className="font-bold text-cyan-800 mb-2 flex items-center gap-2">
                  <span className="text-2xl">🔑</span> Key (K)
                </h4>
                <p className="text-sm text-cyan-700">
                  "What do I offer?"
                  <br />
                  What information can this word provide?
                </p>
              </div>
              <div className="bg-gradient-to-br from-green-50 to-green-100 p-5 rounded-xl border-2 border-green-200 shadow-md">
                <h4 className="font-bold text-green-800 mb-2 flex items-center gap-2">
                  <span className="text-2xl">💎</span> Value (V)
                </h4>
                <p className="text-sm text-green-700">
                  "What will I contribute?"
                  <br />
                  The actual information to pass along
                </p>
              </div>
            </div>

            {/* Matrix visualizations */}
            {showMath && Q.length > 0 && (
              <div className="grid md:grid-cols-3 gap-4">
                <MatrixHeatmap
                  matrix={Q}
                  rowLabels={tokens}
                  title="Query (Q)"
                  colorScheme="default"
                  showValues={true}
                />
                <MatrixHeatmap
                  matrix={K}
                  rowLabels={tokens}
                  title="Key (K)"
                  colorScheme="default"
                  showValues={true}
                />
                <MatrixHeatmap
                  matrix={V}
                  rowLabels={tokens}
                  title="Value (V)"
                  colorScheme="default"
                  showValues={true}
                />
              </div>
            )}

            {/* Vector visualization for first two tokens */}
            {Q.length >= 2 && Q[0].length >= 2 && (
              <div className="bg-white p-6 rounded-xl border-2 border-gray-200 shadow-lg">
                <h4 className="font-semibold text-lg mb-4">
                  Geometric View: Query vs Key for "{tokens[0]}" and "{tokens[1]}"
                </h4>
                <DotProductVisualization
                  query={Q[0].slice(0, 2)}
                  key={K[1].slice(0, 2)}
                  queryLabel={`Q[${tokens[0]}]`}
                  keyLabel={`K[${tokens[1]}]`}
                  animate={true}
                />
              </div>
            )}
          </div>
        );

      case 'scores':
        return (
          <div className="space-y-6">
            <div className="bg-yellow-50 p-6 rounded-xl border-2 border-yellow-200">
              <h4 className="font-semibold text-yellow-900 mb-2 flex items-center gap-2">
                <span className="text-2xl">🎯</span> Computing Compatibility
              </h4>
              <p className="text-yellow-800">
                We multiply each Query with every Key (transposed) to get a compatibility score.
                Higher score = better match!
              </p>
            </div>

            {showMath && scores.length > 0 && (
              <>
                <MatrixHeatmap
                  matrix={scores}
                  rowLabels={tokens}
                  colLabels={tokens}
                  title="Raw Scores (Q × K^T)"
                  description="Each cell shows how much token i's Query matches token j's Key"
                  colorScheme="diverging"
                  showValues={true}
                  animate={true}
                />

                <MatrixHeatmap
                  matrix={scaledScores}
                  rowLabels={tokens}
                  colLabels={tokens}
                  title={`Scaled Scores (÷ √${dK})`}
                  description="Scaling prevents extreme values"
                  colorScheme="diverging"
                  showValues={true}
                  animate={true}
                />
              </>
            )}
          </div>
        );

      case 'attention':
        return (
          <div className="space-y-6">
            {showMath && attentionWeights.length > 0 && (
              <MatrixHeatmap
                matrix={attentionWeights}
                rowLabels={tokens}
                colLabels={tokens}
                title="Attention Weights"
                description="Each row is a probability distribution (sums to 1.0)"
                isAttention={true}
                showValues={true}
                animate={true}
              />
            )}

            <div className="bg-white p-6 rounded-xl border-2 border-gray-200 shadow-lg">
              <h4 className="font-semibold text-lg mb-4 flex items-center gap-2">
                <span className="text-2xl">🌊</span> Attention Flow Visualization
              </h4>
              <AttentionFlow
                tokens={tokens}
                attentionWeights={attentionWeights}
                selectedToken={null}
                animate={true}
                width={800}
                height={300}
              />
            </div>

            <div className="bg-blue-50 p-5 rounded-xl border-2 border-blue-200">
              <p className="text-blue-900">
                <strong>💡 Try This:</strong> Click on a token in the flow visualization above to
                see its attention pattern!
              </p>
            </div>
          </div>
        );

      case 'output':
        return (
          <div className="space-y-6">
            <div className="bg-green-50 p-6 rounded-xl border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2 flex items-center gap-2">
                <span className="text-2xl">✨</span> Context-Aware Representations
              </h4>
              <p className="text-green-800">
                Each word now contains not just its own information, but a weighted blend from all
                words it paid attention to!
              </p>
            </div>

            {showMath && output.length > 0 && (
              <MatrixHeatmap
                matrix={output}
                rowLabels={tokens}
                colLabels={['o₀', 'o₁']}
                title="Output Vectors"
                description="Context-enriched representations after attention"
                colorScheme="default"
                showValues={true}
                animate={true}
              />
            )}

            <div className="bg-purple-50 p-6 rounded-xl border-2 border-purple-200">
              <h4 className="font-semibold text-purple-900 mb-3">🎓 Key Takeaway</h4>
              <p className="text-purple-800 mb-3">
                The attention mechanism allows each word to dynamically gather relevant information
                from the entire sequence. This is how transformers understand context!
              </p>
              <p className="text-sm text-purple-700">
                In real transformers, this process happens with multiple attention heads in
                parallel, and is repeated across many layers.
              </p>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  const exampleSentences = [
    'The cat sat on the mat',
    'She loves reading books',
    'Coffee tastes great in morning',
    'The dog chased the ball',
    'Attention is all you need',
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            How AI Pays Attention
          </h1>
          <p className="text-xl text-gray-700 max-w-3xl mx-auto">
            An interactive journey through the transformer attention mechanism—the breakthrough that
            powers modern AI like ChatGPT
          </p>
        </div>

        {/* Main content */}
        <div className="grid lg:grid-cols-3 gap-6 mb-6">
          {/* Left column - Controls */}
          <div className="lg:col-span-1 space-y-4">
            {/* Input */}
            <div className="bg-white rounded-xl p-6 shadow-lg border-2 border-gray-200">
              <label className="block font-semibold text-gray-800 mb-3">
                <BookOpen className="inline mr-2" size={20} />
                Input Text:
              </label>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none resize-none"
                rows={3}
                placeholder="Type your text here..."
              />

              <div className="mt-4">
                <div className="text-sm font-semibold text-gray-700 mb-2">Quick Examples:</div>
                <div className="space-y-2">
                  {exampleSentences.map((sentence, i) => (
                    <button
                      key={i}
                      onClick={() => setInputText(sentence)}
                      className="w-full text-left px-3 py-2 bg-gray-50 hover:bg-blue-50 rounded-lg border border-gray-200 hover:border-blue-300 transition-all text-sm"
                    >
                      "{sentence}"
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Interactive Controls */}
            <InteractiveControls
              temperature={temperature}
              onTemperatureChange={setTemperature}
              showMath={showMath}
              onShowMathChange={setShowMath}
              autoPlay={autoPlay}
              onAutoPlayChange={setAutoPlay}
              step={currentStep}
              maxSteps={steps.length}
              onStepChange={setCurrentStep}
            />
          </div>

          {/* Right column - Visualization */}
          <div className="lg:col-span-2 space-y-6">
            {/* Progress */}
            <div className="bg-white rounded-xl p-6 shadow-lg border-2 border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-gray-800">
                  {currentStepData.icon} {currentStepData.title}
                </h2>
                <button
                  onClick={() => setAutoPlay(!autoPlay)}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg flex items-center gap-2 hover:bg-blue-700 transition-colors"
                >
                  {autoPlay ? <Pause size={18} /> : <Play size={18} />}
                  {autoPlay ? 'Pause' : 'Play'}
                </button>
              </div>

              <h3 className="text-lg text-blue-600 font-medium mb-2">
                {currentStepData.subtitle}
              </h3>
              <p className="text-gray-700 mb-4">{currentStepData.description}</p>

              {/* Progress bar */}
              <div className="w-full bg-gray-200 rounded-full h-3 mb-4 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500"
                  style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
                />
              </div>

              {/* Step buttons */}
              <div className="flex flex-wrap gap-2">
                {steps.map((step, i) => (
                  <button
                    key={i}
                    onClick={() => setCurrentStep(i)}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                      i === currentStep
                        ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg scale-105'
                        : i < currentStep
                        ? 'bg-green-100 text-green-800 hover:bg-green-200'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    {step.icon} {i + 1}
                  </button>
                ))}
              </div>
            </div>

            {/* Step content */}
            <div className="bg-white rounded-xl p-6 shadow-lg border-2 border-gray-200">
              {renderStepContent()}
            </div>

            {/* Calculation breakdown */}
            {showMath && (
              <CalculationBreakdown
                title={`Mathematical Breakdown: ${currentStepData.title}`}
                steps={getCalculationSteps()}
                defaultExpanded={false}
              />
            )}
          </div>
        </div>

        {/* Footer info */}
        <div className="bg-white rounded-xl p-6 shadow-lg border-2 border-gray-200 text-center">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Lightbulb className="text-yellow-500" />
            <h3 className="font-bold text-gray-800">About This Visualization</h3>
          </div>
          <p className="text-gray-600 text-sm max-w-3xl mx-auto">
            This interactive tool breaks down the self-attention mechanism from the "Attention is All
            You Need" paper. It uses simplified mathematics and beautiful visualizations inspired by
            3Blue1Brown to make transformers intuitive and accessible to everyone.
          </p>
        </div>
      </div>
    </div>
  );
};

export default AttentionVisualizer;
