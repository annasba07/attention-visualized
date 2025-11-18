# End-to-End Testing Script with Screenshots

**Application**: Attention Visualizer - 3Blue1Brown Edition
**URL**: http://localhost:3000
**Date**: 2025-11-18

## Pre-Test Setup

1. Open http://localhost:3000 in Chrome or Firefox
2. Open DevTools (F12) and check Console tab - should have NO errors
3. Set browser window to 1920x1080 for consistent screenshots
4. Create a `screenshots/` folder for saving images

---

## Test 1: Initial Load and Layout

### Actions:
- Load the page
- Observe default state

### Expected Results:
- [ ] Beautiful gradient background (blue → purple → pink)
- [ ] Header: "How AI Pays Attention"
- [ ] Default text: "The cat sat on the mat"
- [ ] Left sidebar visible with:
  - Input textarea
  - 5 example sentences
  - Interactive Controls panel
  - Temperature slider at 1.0
  - "Show Mathematical Details" toggle (ON)
  - "Auto-Play Tutorial" toggle (OFF)
  - Step navigation (Step 1 / 5)
- [ ] Right panel showing Step 1 visualization
- [ ] Progress bar at 0%
- [ ] NO console errors

### Screenshots to Capture:
1. `01-initial-load-full-page.png` - Full page view
2. `01-controls-panel.png` - Close-up of Interactive Controls
3. `01-console-clean.png` - DevTools console showing no errors

---

## Test 2: Step 1 - Words → Vectors

### Actions:
- Verify you're on Step 1 (default)
- Examine the visualization

### Expected Results:
- [ ] Title: "Step 1: Words → Vectors"
- [ ] Subtitle: "Converting language to mathematics"
- [ ] Token display showing: ["The", "cat", "sat", "on", "the", "mat"]
- [ ] Each token in white card with shadow
- [ ] Hover over tokens increases shadow (test this)
- [ ] Embedding Matrix heatmap visible (Math is ON)
  - Row labels: token names
  - Column labels: d₀, d₁, d₂, d₃
  - Blue gradient colors
  - Hover shows tooltip with exact value
- [ ] Calculation Breakdown section (expandable)
  - Purple/blue header with calculator icon
  - Click to expand
  - Shows formulas and explanations

### Screenshots to Capture:
4. `02-step1-full.png` - Full Step 1 view
5. `02-step1-tokens.png` - Token display
6. `02-step1-matrix.png` - Embedding matrix
7. `02-step1-matrix-hover.png` - Matrix with hover tooltip visible
8. `02-step1-calculation-expanded.png` - Calculation breakdown expanded

---

## Test 3: Step 2 - Query, Key, Value

### Actions:
- Click "Next →" button OR click "Step 2" in step navigator
- Examine the three concepts

### Expected Results:
- [ ] Title: "Step 2: Query, Key, Value (QKV)"
- [ ] Three concept cards visible:
  - Red card: "Query" - "What am I looking for?"
  - Cyan card: "Key" - "What do I offer?"
  - Green card: "Value" - "What will I contribute?"
- [ ] Gradient backgrounds on cards
- [ ] Hover effects work
- [ ] Three matrix heatmaps (Q, K, V) side by side
  - Different colors for each
  - Interactive hover with tooltips
- [ ] Geometric Visualization (Dot Product canvas)
  - 2D vector diagram
  - Two vectors with arrow heads
  - Glow effects on vectors
  - Angle arc between vectors
  - Dashed projection line
  - Info panel showing:
    - Q · K = [value]
    - |Q| = [magnitude]
    - |K| = [magnitude]
    - cos(θ) = [similarity]
    - Alignment status
  - Smooth pulsing animation

### Screenshots to Capture:
9. `03-step2-full.png` - Full Step 2 view
10. `03-step2-concept-cards.png` - Three QKV concept cards
11. `03-step2-matrices.png` - Q, K, V matrices
12. `03-step2-geometric-viz.png` - Dot product geometric visualization
13. `03-step2-geometric-info.png` - Close-up of info panel

---

## Test 4: Step 3 - Compatibility Scores

### Actions:
- Click "Next →" to advance to Step 3
- Compare raw vs scaled scores

### Expected Results:
- [ ] Title: "Step 3: Compatibility Scores"
- [ ] Yellow info box explaining compatibility matching
- [ ] Raw Scores Matrix visible
  - Title: "Raw Scores (Q × K^T)"
  - Diverging colors (blue positive, red negative)
  - Row and column labels (token names)
  - Hover tooltips work
- [ ] Scaled Scores Matrix visible
  - Title shows "÷√d_k" or similar
  - Values are scaled down from raw scores
  - Same layout as raw scores
- [ ] Verify scaling: pick a cell, check that scaled ≈ raw / 1.41

### Screenshots to Capture:
14. `04-step3-full.png` - Full Step 3 view
15. `04-step3-raw-scores.png` - Raw scores matrix
16. `04-step3-scaled-scores.png` - Scaled scores matrix
17. `04-step3-hover-comparison.png` - Both matrices with hover to compare values

---

## Test 5: Step 4 - Attention Weights (MAIN FEATURE)

### Actions:
- Click "Next →" to advance to Step 4
- Examine attention weights matrix
- Test the attention flow visualization

### Expected Results:
- [ ] Title: "Step 4: Attention Weights"
- [ ] Attention Weights Matrix visible
  - Yellow for high attention
  - Light purple for low attention
  - All values between 0 and 1
  - Each row sums to 1.0 (verify by hovering)
  - Color scale legend at bottom
- [ ] Attention Flow Visualization canvas
  - Gradient background
  - Tokens displayed as circles
  - Initial state: "Click or hover over a token..."

### Interactive Testing:
- [ ] Click on "cat" token:
  - Selected token glows (red/pink pulse)
  - Target tokens glow (cyan/light blue)
  - Bezier curves appear connecting "cat" to other tokens
  - **PARTICLES FLOW** along the curves (this is the star feature!)
  - Particle count varies by attention weight
  - Particles move faster for higher weights
  - Attention percentages shown above target tokens
  - Title updates: "cat is paying attention to:"

- [ ] Click on "the" token:
  - Different attention pattern
  - Different particle flows
  - Different percentages

- [ ] Click same token again:
  - Deselects
  - Returns to "Click or hover..." state

### Screenshots to Capture:
18. `05-step4-full.png` - Full Step 4 view
19. `05-step4-weights-matrix.png` - Attention weights matrix with legend
20. `05-step4-flow-initial.png` - Flow visualization (no selection)
21. `05-step4-flow-cat-selected.png` - Flow with "cat" selected showing particles
22. `05-step4-flow-the-selected.png` - Flow with "the" selected (different pattern)
23. `05-step4-particles-closeup.png` - Close-up of particles flowing

### GIF to Capture (if possible):
- `05-step4-particles-animated.gif` - 5-second recording of particles flowing from "cat"

---

## Test 6: Step 5 - Weighted Combination

### Actions:
- Click "Next →" to advance to Step 5
- Review final output

### Expected Results:
- [ ] Title: "Step 5: Weighted Combination"
- [ ] Green success box
  - Celebration of completion
  - Explanation of context-aware representations
- [ ] Output Matrix heatmap
  - 2 columns (o₀, o₁)
  - Row labels: token names
  - Values show final representations
  - Smooth animation
- [ ] Purple summary box
  - Key takeaway about attention mechanism
  - Mention of multi-head attention
  - Reference to multiple layers
- [ ] Step counter shows: "5 / 5"
- [ ] Progress bar at 100%

### Screenshots to Capture:
24. `06-step5-full.png` - Full Step 5 view
25. `06-step5-output-matrix.png` - Output matrix close-up
26. `06-step5-summary.png` - Summary box

---

## Test 7: Temperature Control

### Actions:
- Return to Step 4 (to see attention weights clearly)
- Test temperature slider

### Test 7a: Sharp Temperature (0.3)
- [ ] Drag slider to 0.3 OR click "Sharp" preset button
- [ ] Display shows: "Temperature: 0.30"
- [ ] Status shows: "Sharp"
- [ ] Observe attention weights matrix - should be more focused (higher contrast)
- [ ] Click a token in flow visualization
- [ ] Particles should flow to fewer tokens (more focused attention)

**Screenshot**: `07-temp-sharp-0.3.png`

### Test 7b: Normal Temperature (1.0)
- [ ] Drag slider to 1.0 OR click "Normal" preset button
- [ ] Display shows: "Temperature: 1.00"
- [ ] Status shows: "Balanced"
- [ ] Moderate attention distribution

**Screenshot**: `08-temp-normal-1.0.png`

### Test 7c: Smooth Temperature (2.5)
- [ ] Drag slider to 2.5 OR click "Smooth" preset button
- [ ] Display shows: "Temperature: 2.50"
- [ ] Status shows: "Smooth"
- [ ] Attention more evenly distributed (lower contrast)
- [ ] Click a token - particles flow to more tokens (diffuse attention)

**Screenshot**: `09-temp-smooth-2.5.png`

### GIF to Capture:
- `07-temp-slider-effect.gif` - Recording while sliding temperature from 0.3 → 3.0, showing matrix changes

---

## Test 8: Math Toggle

### Actions:
- Toggle "Show Mathematical Details" OFF
- Navigate through steps

### Expected Results:
- [ ] All matrices disappear
- [ ] Calculation breakdowns hide
- [ ] Only intuitive visualizations remain:
  - Token displays
  - Concept cards
  - Flow visualization
  - Geometric visualizations
- [ ] Page looks cleaner, less technical

### Screenshots to Capture:
27. `10-math-off-step1.png` - Step 1 with math OFF
28. `10-math-off-step2.png` - Step 2 with math OFF
29. `10-math-off-step4.png` - Step 4 with math OFF

### Actions (continued):
- Toggle "Show Mathematical Details" ON
- Verify matrices reappear

---

## Test 9: Auto-Play

### Actions:
- Go to Step 1
- Toggle "Auto-Play Tutorial" ON
- Observe

### Expected Results:
- [ ] Every 5 seconds, automatically advances to next step
- [ ] Goes through: Step 1 → 2 → 3 → 4 → 5
- [ ] After Step 5, loops back to Step 1
- [ ] Smooth transitions
- [ ] Click Auto-Play toggle OFF - stops advancing

### GIF to Capture:
- `11-autoplay-sequence.gif` - 30-second recording of auto-play

---

## Test 10: Input Text Changes

### Test 10a: "Attention is all you need"
- [ ] Clear input textarea
- [ ] Type: "Attention is all you need"
- [ ] Press outside textarea or wait for update
- [ ] Tokens update to: ["Attention", "is", "all", "you", "need"]
- [ ] All matrices recalculate (dimensions change to 5×5)
- [ ] Navigate through all steps - no errors
- [ ] Attention patterns are different from default

**Screenshot**: `12-custom-input-attention.png` - Step 4 with this input

### Test 10b: "Hello world"
- [ ] Change input to: "Hello world"
- [ ] Only 2 tokens: ["Hello", "world"]
- [ ] Matrices are 2×2
- [ ] Flow visualization simpler
- [ ] Everything works correctly

**Screenshot**: `13-custom-input-hello.png` - Step 4 with this input

### Test 10c: Edge Cases
Test these inputs and verify no crashes:

- Single word: "Test"
  - Expected: 1 token, 1×1 matrices
- Long sentence: "The cat sat on the mat eating fish"
  - Expected: 8 tokens, 8×8 matrices
- Extra spaces: "The  cat   sat"
  - Expected: Filters to ["The", "cat", "sat"]

**Screenshots**:
- `14-edge-single-word.png`
- `14-edge-long-sentence.png`

---

## Test 11: Step Navigation

### Actions:
- Test Previous/Next buttons
- Test direct step selection
- Test progress bar accuracy

### Expected Results:
- [ ] "← Previous" button disabled on Step 1
- [ ] "Next →" button disabled on Step 5
- [ ] Clicking "← Previous" from Step 3 goes to Step 2
- [ ] Clicking "Next →" from Step 2 goes to Step 3
- [ ] Progress bar updates:
  - Step 1: 20%
  - Step 2: 40%
  - Step 3: 60%
  - Step 4: 80%
  - Step 5: 100%
- [ ] Progress percentage shown: "20% Complete", etc.

**Screenshot**: `15-navigation-progress.png` - Progress bar at different steps

---

## Test 12: Visual Quality Check

### Color Verification:
- [ ] Query elements are red/pink (#FF6B6B)
- [ ] Key elements are cyan (#4ECDC4)
- [ ] Value elements are green (#95E1D3)
- [ ] Gradients are smooth
- [ ] Text has sufficient contrast

### Typography:
- [ ] Headings are bold and clear
- [ ] Body text is readable
- [ ] Math formulas use monospace font
- [ ] Consistent sizing throughout

### Spacing:
- [ ] Generous padding/margins
- [ ] No overlapping elements
- [ ] Clear visual hierarchy
- [ ] Breathing room around components

### Effects:
- [ ] Glow effects visible on vectors
- [ ] Shadows on cards
- [ ] Hover states are obvious
- [ ] Transitions are smooth (not jarring)

---

## Test 13: Performance Check

### Smooth Animations:
- [ ] Go to Step 2 - geometric visualization pulses smoothly
- [ ] Go to Step 4 - particles flow at 60fps (no jank)
- [ ] Hover over matrices - tooltips appear instantly
- [ ] Temperature slider - changes update immediately
- [ ] Step transitions are smooth

### Memory Leak Test:
- [ ] Change steps 20+ times rapidly
  - Observe: No slowdown, animations still smooth
- [ ] Change input text 10+ times
  - Observe: No issues, page responsive
- [ ] Enable auto-play, let run for 5 minutes
  - Observe: Stable performance, no degradation

### Performance Metrics (DevTools):
- [ ] Open DevTools → Performance tab
- [ ] Record while navigating through all steps
- [ ] Check: Frame rate stays above 50fps
- [ ] Check: No long tasks (> 50ms)

---

## Test 14: Browser Console Check

### Throughout ALL tests above:
- [ ] NO errors in console
- [ ] NO warnings in console
- [ ] Only expected output: React DevTools messages

**Screenshot**: `16-console-final.png` - Clean console after all tests

---

## Test 15: Responsive Design (Optional)

### Desktop (1920×1080)
- [ ] Two-column layout
- [ ] All elements visible
- [ ] No horizontal scroll

**Screenshot**: `17-desktop-1920.png`

### Tablet (768px)
- [ ] Resize browser to 768px width
- [ ] Layout stacks vertically
- [ ] All content accessible

**Screenshot**: `18-tablet-768.png`

### Mobile (375px)
- [ ] Resize to 375px width
- [ ] Single column layout
- [ ] Readable text
- [ ] Scrollable

**Screenshot**: `19-mobile-375.png`

---

## Final Checklist

Before declaring success, verify:
- [ ] All 5 steps work perfectly
- [ ] Temperature slider affects attention visibly
- [ ] Particles animate smoothly in Step 4
- [ ] All matrices display correctly with hover tooltips
- [ ] Calculation breakdowns expand/collapse
- [ ] No console errors or warnings
- [ ] Performance is smooth (60fps animations)
- [ ] Design looks beautiful (3Blue1Brown aesthetic)
- [ ] Educational value is clear (non-technical users can understand)

---

## Summary Report Template

```
E2E Test Report - Attention Visualizer
Date: 2025-11-18
Browser: Chrome [version]
Screen Size: 1920×1080

Total Tests: 15
Tests Passed: __/15
Tests Failed: __/15

Issues Found:
1. [Description]
2. [Description]

Features Working Perfectly:
1. Temperature control with live updates
2. Particle flow animation
3. Geometric dot product visualization
4. [etc.]

Screenshots Captured: __/19 (+ __ GIFs)

Overall Score: __/10

Notes:
[Any additional observations]
```

---

## Key Features to Highlight in Screenshots

When selecting your best screenshots for documentation/README:

1. **Particle Flow** (Step 4) - The star feature showing attention visually
2. **Geometric Dot Product** (Step 2) - Shows WHY similarity works
3. **Temperature Control** - Side-by-side comparison at 0.3, 1.0, 2.5
4. **Beautiful Design** - Full page view showing 3Blue1Brown aesthetic
5. **Progressive Learning** - All 5 steps in sequence

---

**Testing Instructions:**
1. Work through each test section in order
2. Check off completed items
3. Capture screenshots as specified
4. Note any issues or unexpected behavior
5. Fill out the summary report at the end

**Estimated Time:** 45-60 minutes for complete thorough testing

Good luck!
