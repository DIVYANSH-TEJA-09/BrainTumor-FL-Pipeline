# Synchronized Camera Testing Guide

## Overview
The 3D tumor progression visualization app now features **synchronized camera** across all three panels (Actual, Baseline, LSTM Hybrid). When you rotate or zoom any panel, all three panels move together, making clinical comparison easy.

## Quick Start

### 1. Launch the App
```bash
cd progression
python run_streamlit_app.py
```

The app will start at: `http://localhost:8501`

### 2. Test Synchronized Camera

#### Test Case 1: Rotation Synchronization
1. Select any patient from the sidebar (e.g., "MU-Glioma-Post-001")
2. Hover over the left panel (Actual Tumor - blue) with your mouse
3. Click and drag to rotate the view
4. **Expected**: All three panels (left, center, right) rotate together with the same angle and orientation

#### Test Case 2: Zoom Synchronization
1. With the patient still displayed, move your mouse over the center panel (Baseline - red)
2. Scroll up or down with your mouse wheel to zoom in/out
3. **Expected**: All three panels zoom together to show the same magnification level

#### Test Case 3: Pan Synchronization
1. Move to the right panel (LSTM Hybrid - green)
2. Right-click and drag (or use shift+drag depending on Plotly settings) to pan the view
3. **Expected**: All three panels pan together, keeping the same relative positions

### 3. Verify Camera State Persistence

#### Test Case 4: State Persistence
1. Load a patient and rotate all three panels to a specific angle
2. Use the timepoint slider to move to a different timepoint
3. **Expected**: The three panels should maintain the same camera angle/zoom across different timepoints
4. Navigate to a different patient
5. **Expected**: The synchronized camera state persists

### 4. Test Grade Filtering

#### Test Case 5: HGG Patient Viewing
1. In the sidebar, select "Filter by Grade: HGG"
2. Select any HGG patient
3. Rotate the panels - verify all three rotate together
4. **Expected**: Synchronized camera works for HGG patients

#### Test Case 6: LGG Patient Viewing
1. In the sidebar, select "Filter by Grade: LGG"
2. Select any LGG patient
3. Rotate the panels - verify all three rotate together
4. **Expected**: Synchronized camera works for LGG patients

### 5. Edge Cases

#### Test Case 7: Volume Visibility
1. Load a patient with very small tumor volumes
2. Adjust the Tumor Opacity slider to make the tumor more visible
3. Rotate panels
4. **Expected**: Camera synchronization works regardless of opacity settings

#### Test Case 8: Brain Overlay
1. Enable "Show Brain Overlay" checkbox
2. Adjust Brain Opacity slider
3. Rotate the panels
4. **Expected**: Camera remains synchronized when brain overlay is toggled

#### Test Case 9: Multiple Users (Simulated)
1. Open the app in two browser tabs
2. In Tab 1: Load patient "MU-Glioma-Post-001" and rotate panels
3. In Tab 2: Load patient "MU-Glioma-Post-002" and try rotating panels
4. **Expected**: Each tab maintains its own synchronized camera state independently (no cross-contamination)

## Technical Details

### Implementation
The synchronized camera is implemented using Streamlit's session state:

```python
# Initialize in session state (line 31-37)
if 'sync_camera' not in st.session_state:
    st.session_state.sync_camera = dict(
        eye=dict(x=1.6, y=1.0, z=0.8),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0)
    )

# Applied to all three scenes (line 658-662)
camera=dict(
    eye=st.session_state.sync_camera['eye'],
    up=st.session_state.sync_camera['up'],
    center=st.session_state.sync_camera['center']
)
```

### Camera State Dictionary
- **eye**: Camera position (x, y, z coordinates) relative to the scene
- **up**: Up vector defining which direction is "up" in the view
- **center**: Center point of the scene the camera focuses on

### Session State
- Streamlit's `session_state` persists data across app reruns
- Each browser session has its own session state
- Changes to camera are automatically reflected across all three scenes on next render

## Troubleshooting

### ✅ Fixed Issues
- **Duplicate Element ID Error** - Removed duplicate widget definitions that were causing Streamlit to render the same radio/slider/checkbox elements twice

### Panels Not Synchronized
**Problem**: When rotating one panel, other panels don't move

**Solutions**:
1. Check browser console (F12 > Console) for JavaScript errors
2. Refresh the page (Ctrl+R or Cmd+R)
3. Clear browser cache if issue persists
4. Check `streamlit_3d_progression.py` line 659-662 for correct `sync_camera` reference

### Camera Jumps Unexpectedly
**Problem**: Camera position snaps or resets when changing timepoints

**Solutions**:
1. This is normal if interacting with Streamlit widgets - the app reruns
2. The synchronized state should persist after the rerun
3. If it continues to jump, check if any callback is resetting `st.session_state.sync_camera`

### App Crashes on Load
**Problem**: "Prediction data not found" error

**Solutions**:
1. Run the diagnostic check: `python diagnostic_check.py`
2. Verify prediction index exists: `ls streamlit_data/prediction_index.json`
3. If not present, regenerate: `python src/08_generate_viz_data.py`

## Performance Notes

- Rotating all three panels simultaneously may be slower than rotating a single panel
- If you experience lag:
  - Disable "Show Brain Overlay" for faster rendering
  - Increase "Step Size" slider to reduce mesh resolution
  - Close other browser tabs to free up RAM
- Mesh generation is cached, so first load of a patient may be slightly slower

## Related Files

- `streamlit_3d_progression.py` - Main app file with synchronized camera implementation
- `run_streamlit_app.py` - Launcher script
- `diagnostic_check.py` - Diagnostic tool for troubleshooting
- `ENHANCED_3PANEL_GUIDE.md` - Full feature guide
- `TROUBLESHOOT_APP.md` - General troubleshooting guide

## Success Criteria

✅ All three panels visible side-by-side
✅ Rotating any panel rotates all three together
✅ Zooming any panel zooms all three together
✅ Camera state persists across timepoint changes
✅ Camera state persists across patient selection
✅ No errors in browser console
✅ App performs smoothly without lag

If all criteria are met, the synchronized camera feature is working correctly!
