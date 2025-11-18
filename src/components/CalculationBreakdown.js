import React, { useState } from 'react';
import { Calculator, ChevronDown, ChevronRight } from 'lucide-react';
import { COLORS } from '../utils/colorUtils';

/**
 * CalculationBreakdown - Step-by-step calculation visualization
 * Shows progressive math with highlights and explanations
 */
const CalculationBreakdown = ({
  title,
  steps,
  defaultExpanded = false,
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className="calculation-breakdown bg-white border-2 border-gray-200 rounded-xl overflow-hidden shadow-lg">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-6 py-4 flex items-center justify-between bg-gradient-to-r from-purple-50 to-blue-50 hover:from-purple-100 hover:to-blue-100 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Calculator className="text-purple-600" size={24} />
          <h3 className="text-lg font-bold text-gray-800">{title}</h3>
        </div>
        {isExpanded ? (
          <ChevronDown className="text-gray-600" size={24} />
        ) : (
          <ChevronRight className="text-gray-600" size={24} />
        )}
      </button>

      {isExpanded && (
        <div className="p-6 space-y-6">
          {steps.map((step, index) => (
            <CalculationStep key={index} step={step} stepNumber={index + 1} />
          ))}
        </div>
      )}
    </div>
  );
};

const CalculationStep = ({ step, stepNumber }) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      className={`calculation-step transition-all duration-300 ${
        isHovered ? 'transform scale-102' : ''
      }`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Step header */}
      <div className="flex items-start gap-4 mb-3">
        <div
          className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold text-white"
          style={{ backgroundColor: COLORS.math.matrix }}
        >
          {stepNumber}
        </div>
        <div className="flex-1">
          <h4 className="font-semibold text-gray-800 mb-1">{step.label}</h4>
          {step.description && (
            <p className="text-sm text-gray-600">{step.description}</p>
          )}
        </div>
      </div>

      {/* Formula display */}
      {step.formula && (
        <div className="ml-12 mb-3">
          <div className="bg-gray-50 border-2 border-gray-200 rounded-lg p-4 font-mono text-sm">
            <div className="text-gray-800">{step.formula}</div>
          </div>
        </div>
      )}

      {/* Example calculation */}
      {step.example && (
        <div className="ml-12 mb-3">
          <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-4">
            <div className="text-sm font-semibold text-blue-900 mb-2">
              Example:
            </div>
            <div className="font-mono text-sm text-blue-800 space-y-1">
              {Array.isArray(step.example) ? (
                step.example.map((line, i) => <div key={i}>{line}</div>)
              ) : (
                <div>{step.example}</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Visual explanation */}
      {step.visual && (
        <div className="ml-12">
          {step.visual}
        </div>
      )}

      {/* Key insight */}
      {step.insight && (
        <div className="ml-12 mt-3">
          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-r-lg">
            <div className="flex items-start gap-2">
              <span className="text-yellow-600 font-bold">💡</span>
              <p className="text-sm text-yellow-900">{step.insight}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * InlineFormula - Colored inline formula component
 */
export const InlineFormula = ({ children, color = COLORS.math.matrix }) => (
  <span
    className="inline-block px-2 py-1 rounded font-mono text-sm font-semibold"
    style={{
      backgroundColor: `${color}22`,
      color: color,
      border: `1px solid ${color}44`,
    }}
  >
    {children}
  </span>
);

/**
 * MatrixDisplay - Show matrix in calculation
 */
export const MatrixDisplay = ({ matrix, label, color = COLORS.math.matrix }) => {
  if (!matrix || matrix.length === 0) return null;

  return (
    <div className="inline-block">
      {label && (
        <div className="text-xs font-semibold mb-1" style={{ color }}>
          {label}
        </div>
      )}
      <div className="border-2 rounded-lg p-2 bg-white inline-block" style={{ borderColor: color }}>
        {matrix.map((row, i) => (
          <div key={i} className="flex gap-2">
            {row.map((val, j) => (
              <div
                key={j}
                className="w-12 text-center font-mono text-sm"
                style={{ color }}
              >
                {typeof val === 'number' ? val.toFixed(2) : val}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * ComparisonView - Show before/after or step progression
 */
export const ComparisonView = ({ before, after, labels = ['Before', 'After'] }) => (
  <div className="grid grid-cols-2 gap-4">
    <div>
      <div className="text-sm font-semibold text-gray-700 mb-2">{labels[0]}</div>
      <div className="bg-gray-50 rounded-lg p-3 border-2 border-gray-200">
        {before}
      </div>
    </div>
    <div>
      <div className="text-sm font-semibold text-gray-700 mb-2">{labels[1]}</div>
      <div className="bg-green-50 rounded-lg p-3 border-2 border-green-200">
        {after}
      </div>
    </div>
  </div>
);

export default CalculationBreakdown;
