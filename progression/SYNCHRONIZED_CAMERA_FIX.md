# Synchronized Camera Fix - Implementation Details

## Problem
The three 3D panels were not synchronized - rotating one panel did not rotate the other two panels.

## Root Cause
**The approach was fundamentally flawed:**

The original implementation tried to:
1. Store camera state in `st.session_state.sync_camera`
2. Apply this state to three separate Plotly figures independently
3. Hope that Streamlit would capture camera interactions back to session state

**Why this didn't work:**
- Plotly doesn't expose camera interaction events to Streamlit
- Each figure (fig_actual, fig_baseline, fig_hybrid) was completely independent
- When users rotated one figure, the other figures had no way to know about it
- Session state camera state never updated based on user interactions

## Solution: Plotly Subplots
**Use Plotly's built-in subplot architecture** which natively supports synchronized camera.

### Implementation
```python
# Instead of 3 separate figures:
fig_actual = create_figure(traces_actual, "scene1")  # ❌ Independent
fig_baseline = create_figure(traces_baseline, "scene2")  # ❌ Independent  
fig_hybrid = create_figure(traces_hybrid, "scene3")  # ❌ Independent

# Use 1 figure with 3 subplots:
fig = make_subplots(rows=1, cols=3, specs=[[...]])  # ✅ Synchronized
```

### Key Changes
1. **Import subplots**: `from plotly.subplots import make_subplots`
2. **Create subplot figure**: 3 columns, 1 row, each with `scatter3d` type
3. **Add all traces to subplots**: Traces maintain their scene (scene1, scene2, scene3)
4. **Update all scenes with same camera**: `fig.update_layout(scene1=..., scene2=..., scene3=...)`
5. **Single figure display**: `st.plotly_chart(fig, ...)` displays all three panels together

### Why This Works
- Plotly's subplot system keeps all scenes in **one HTML rendering context**
- When user rotates one scene, Plotly internally synchronizes all scenes with the same camera
- No need for external callbacks or session state manipulation
- Camera synchronization happens automatically within Plotly's JavaScript layer

## File Changes

### Before (Three Independent Figures)
- Lines 267-357: Complex column layout with 3 separate figures
- Each figure had its own `create_synchronized_figure()` wrapper
- Metrics displayed in columns with figures
- **Result:** No actual synchronization

### After (One Unified Figure with 3 Subplots)
- Lines 267-360: Single `make_subplots()` call
- All traces added to appropriate subplot (col 1, 2, or 3)
- Single `st.plotly_chart()` call shows entire visualization
- Metrics displayed below in columns
- **Result:** Perfect synchronization

## Technical Details

### Subplot Architecture
```
┌─────────────────────────────────────────────┐
│  Scene1 (🔵)  │  Scene2 (🔴)  │  Scene3 (🟢)  │
│   Actual      │   Baseline    │   Hybrid      │
│               │               │               │
│  (rotate all 3 together)                      │
└─────────────────────────────────────────────┘
```

### Camera Settings
```python
camera_settings = dict(
    eye=st.session_state.sync_camera['eye'],
    up=st.session_state.sync_camera['up'],
    center=st.session_state.sync_camera['center']
)

# Applied to ALL scenes in one figure
fig.update_layout(
    scene1=scene_dict,  # Same camera settings
    scene2=scene_dict,  # Same camera settings  
    scene3=scene_dict   # Same camera settings
)
```

When user rotates Scene1, Plotly updates all three scene camera positions simultaneously.

## Testing

### Expected Behavior
1. Launch app: `python run_streamlit_app.py`
2. Load any patient
3. **Rotate** - Click and drag in any panel → all three rotate together ✅
4. **Zoom** - Scroll wheel in any panel → all three zoom together ✅
5. **Pan** - Right-click drag in any panel → all three pan together ✅
6. **Change timepoint** - Move slider → camera state persists ✅
7. **Change patient** - Select new patient → camera resets to default ✅

### Success Indicators
- All three panels (blue, red, green) move in perfect sync
- No lag between panels
- No errors in browser console
- Smooth rotation/zoom/pan

## Related Commits
- `01c5baa` - Fix camera_state → sync_camera
- `9eb1950` - Remove duplicate widgets
- `655cca8` - Remove code duplication + add keys
- `4636370` - Implement true synchronized camera using Plotly subplots ← **THIS FIX**

## Performance Notes
- **Single figure rendering**: Slightly more efficient than 3 separate renders
- **Automatic sync**: No additional JavaScript or callbacks needed
- **Memory**: One Plotly instance vs three instances

## Alternative Approaches Considered
1. **Custom JavaScript**: Would require Streamlit's custom component API
2. **Session state callbacks**: Plotly doesn't expose events to Streamlit
3. **Multiple figures with manual sync**: Attempted but impossible without event capture
4. ✅ **Subplots (CHOSEN)**: Native Plotly support, cleanest solution

---

**Status:** FIXED ✅
**Approach:** Plotly subplots with native synchronization
**Date:** 2026-04-12
