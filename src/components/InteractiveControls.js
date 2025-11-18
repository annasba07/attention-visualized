import React from 'react';
import { Sliders, Thermometer, Zap } from 'lucide-react';
import { COLORS } from '../utils/colorUtils';

/**
 * InteractiveControls - Parameter controls for exploring attention mechanics
 * Allows users to adjust temperature, see live effects
 */
const InteractiveControls = ({
  temperature = 1.0,
  onTemperatureChange,
  showMath = false,
  onShowMathChange,
  autoPlay = false,
  onAutoPlayChange,
  step = 0,
  maxSteps = 5,
  onStepChange,
}) => {
  return (
    <div className="interactive-controls bg-white border-2 border-gray-200 rounded-xl p-6 shadow-lg">
      <div className="flex items-center gap-2 mb-4">
        <Sliders className="text-blue-600" size={24} />
        <h3 className="text-xl font-bold text-gray-800">Interactive Controls</h3>
      </div>

      <div className="space-y-6">
        {/* Temperature Control */}
        {onTemperatureChange && (
          <div className="control-group">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Thermometer className="text-orange-500" size={20} />
                <label className="font-semibold text-gray-700">
                  Temperature: {temperature.toFixed(2)}
                </label>
              </div>
              <div className="text-sm text-gray-500">
                {temperature < 0.5 ? '❄️ Sharp' : temperature > 1.5 ? '🔥 Smooth' : '⚖️ Balanced'}
              </div>
            </div>

            <input
              type="range"
              min="0.1"
              max="3.0"
              step="0.1"
              value={temperature}
              onChange={(e) => onTemperatureChange(parseFloat(e.target.value))}
              className="w-full h-3 bg-gradient-to-r from-blue-200 via-purple-200 to-red-200 rounded-lg appearance-none cursor-pointer slider"
              style={{
                background: `linear-gradient(to right,
                  ${COLORS.key.main} 0%,
                  ${COLORS.math.operation} 50%,
                  ${COLORS.query.main} 100%)`
              }}
            />

            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>0.1 (Focused)</span>
              <span>1.0 (Standard)</span>
              <span>3.0 (Diffuse)</span>
            </div>

            <div className="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
              <p className="text-sm text-blue-900">
                <strong>💡 What does temperature do?</strong>
                <br />
                Lower temperature makes attention more focused on top matches.
                Higher temperature spreads attention more evenly.
              </p>
            </div>
          </div>
        )}

        {/* Math Toggle */}
        {onShowMathChange !== undefined && (
          <div className="control-group">
            <label className="flex items-center gap-3 cursor-pointer group">
              <input
                type="checkbox"
                checked={showMath}
                onChange={(e) => onShowMathChange(e.target.checked)}
                className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
              />
              <div className="flex-1">
                <div className="font-semibold text-gray-700 group-hover:text-blue-600 transition-colors">
                  Show Mathematical Details
                </div>
                <div className="text-sm text-gray-500">
                  View formulas and matrix calculations
                </div>
              </div>
            </label>
          </div>
        )}

        {/* Auto-play Toggle */}
        {onAutoPlayChange !== undefined && (
          <div className="control-group">
            <label className="flex items-center gap-3 cursor-pointer group">
              <input
                type="checkbox"
                checked={autoPlay}
                onChange={(e) => onAutoPlayChange(e.target.checked)}
                className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
              />
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <Zap className="text-yellow-500" size={18} />
                  <div className="font-semibold text-gray-700 group-hover:text-blue-600 transition-colors">
                    Auto-Play Tutorial
                  </div>
                </div>
                <div className="text-sm text-gray-500">
                  Automatically progress through steps
                </div>
              </div>
            </label>
          </div>
        )}

        {/* Step Navigation */}
        {onStepChange !== undefined && (
          <div className="control-group">
            <div className="flex items-center justify-between mb-3">
              <label className="font-semibold text-gray-700">
                Current Step: {step + 1} / {maxSteps}
              </label>
              <div className="text-sm text-gray-500">
                {Math.round(((step + 1) / maxSteps) * 100)}% Complete
              </div>
            </div>

            <div className="flex gap-2 mb-3">
              <button
                onClick={() => onStepChange(Math.max(0, step - 1))}
                disabled={step === 0}
                className="flex-1 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg font-medium hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                ← Previous
              </button>
              <button
                onClick={() => onStepChange(Math.min(maxSteps - 1, step + 1))}
                disabled={step === maxSteps - 1}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Next →
              </button>
            </div>

            {/* Progress bar */}
            <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
                style={{ width: `${((step + 1) / maxSteps) * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Quick presets */}
        {onTemperatureChange && (
          <div className="control-group">
            <div className="font-semibold text-gray-700 mb-2">Quick Presets:</div>
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => onTemperatureChange(0.3)}
                className="px-3 py-2 bg-blue-100 text-blue-700 rounded-lg text-sm font-medium hover:bg-blue-200 transition-colors"
              >
                🎯 Sharp
              </button>
              <button
                onClick={() => onTemperatureChange(1.0)}
                className="px-3 py-2 bg-purple-100 text-purple-700 rounded-lg text-sm font-medium hover:bg-purple-200 transition-colors"
              >
                ⚖️ Normal
              </button>
              <button
                onClick={() => onTemperatureChange(2.0)}
                className="px-3 py-2 bg-orange-100 text-orange-700 rounded-lg text-sm font-medium hover:bg-orange-200 transition-colors"
              >
                🌊 Smooth
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default InteractiveControls;
