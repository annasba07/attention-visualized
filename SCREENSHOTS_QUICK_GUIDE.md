# Essential Screenshots Guide

## Quick Reference: Must-Have Screenshots

**Total time:** ~15 minutes for essentials only

---

## Critical Screenshots (Top Priority)

### 1. Step 4 - Particle Flow Animation ⭐ MOST IMPORTANT
**File**: `step4-particle-flow.png`
**Action**:
1. Go to Step 4
2. Click on the word "cat"
3. Wait for particles to flow
4. Capture full screen

**Why**: This is the signature feature that makes attention tangible

---

### 2. Step 2 - Geometric Dot Product ⭐
**File**: `step2-geometric-visualization.png`
**Action**:
1. Go to Step 2
2. Scroll to the geometric visualization canvas
3. Capture the vector diagram with angle and info panel

**Why**: Shows the "WHY" behind dot product similarity

---

### 3. Temperature Comparison ⭐
**Files**:
- `temp-sharp-03.png`
- `temp-normal-10.png`
- `temp-smooth-25.png`

**Action**:
1. Go to Step 4
2. Set temperature to 0.3, click "cat", screenshot
3. Set temperature to 1.0, click "cat", screenshot
4. Set temperature to 2.5, click "cat", screenshot

**Why**: Demonstrates interactive parameter exploration

---

### 4. Full Page Overview
**File**: `full-page-overview.png`
**Action**: Capture entire page at Step 1 (default state)

**Why**: Shows overall design and layout

---

### 5. All 5 Steps Sequence
**Files**: `step1.png`, `step2.png`, `step3.png`, `step4.png`, `step5.png`
**Action**: Navigate through each step, capture each one

**Why**: Shows progressive learning flow

---

## Animated GIFs (if possible)

### 1. Particle Flow Animation
**File**: `particles-flowing.gif`
**Duration**: 5 seconds
**Action**:
1. Go to Step 4
2. Click "cat"
3. Record particles flowing along Bezier curves

---

### 2. Temperature Slider Effect
**File**: `temperature-effect.gif`
**Duration**: 10 seconds
**Action**:
1. Go to Step 4
2. Click "cat" to select
3. Slowly drag temperature from 0.3 to 3.0
4. Watch attention weights change

---

## How to Capture Screenshots

### Chrome/Firefox:
1. **Full page**: F12 → Ctrl+Shift+P → "Capture full size screenshot"
2. **Visible area**: F12 → Ctrl+Shift+P → "Capture screenshot"
3. **Specific region**: Use Snipping Tool / Screenshot app

### GIF Recording:
- **Windows**: Use ScreenToGif (free)
- **Mac**: Use QuickTime Player → Screen Recording
- **Linux**: Use Peek or SimpleScreenRecorder
- **Browser**: Chrome extension "Screencastify"

---

## Screenshot Checklist

Priority 1 (Must Have):
- [ ] Full page overview
- [ ] Step 4 with particles flowing
- [ ] Step 2 geometric visualization
- [ ] Temperature at 0.3, 1.0, 2.5
- [ ] All 5 steps

Priority 2 (Nice to Have):
- [ ] Matrix hover tooltips
- [ ] Math toggle OFF comparison
- [ ] Custom input text example
- [ ] Calculation breakdown expanded

Priority 3 (Optional):
- [ ] Responsive views (mobile/tablet)
- [ ] Console showing no errors
- [ ] Auto-play in action

---

## Quick Testing Workflow

**5-Minute Version:**
1. Load page (http://localhost:3000)
2. Screenshot: Full page
3. Click through Step 1-5, screenshot each
4. Go to Step 4, click "cat", screenshot particles
5. Done!

**15-Minute Version:**
1. Above + temperature comparisons
2. Above + geometric visualization
3. Above + one custom input example

**Complete Version:**
See `E2E_TEST_SCRIPT.md`

---

## Screenshot Naming Convention

```
Format: [step]-[feature]-[variant].png

Examples:
- step1-embeddings-default.png
- step2-geometric-vectors.png
- step4-particles-cat-selected.png
- step4-particles-temp-03.png
- full-page-step1.png
- controls-temperature-slider.png
```

---

## Image Specifications

**Resolution**: 1920×1080 browser window
**Format**: PNG for screenshots, GIF for animations
**Compression**: Optimize with TinyPNG or similar
**Location**: Save to `screenshots/` folder

---

## After Capturing Screenshots

1. Review each image for clarity
2. Check that key features are visible
3. Organize by priority
4. Optional: Add annotations (arrows, labels) using:
   - Photoshop / GIMP
   - Excalidraw
   - draw.io

---

## What Makes a Good Screenshot

✅ **Good**:
- Clear, focused on one feature
- All UI elements readable
- Shows the feature in action
- Good lighting/contrast

❌ **Bad**:
- Blurry or low resolution
- Too much empty space
- Feature not clearly visible
- Dark or hard to read

---

Happy screenshot hunting! 📸
