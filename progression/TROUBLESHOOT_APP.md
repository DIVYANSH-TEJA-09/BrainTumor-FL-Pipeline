# How to Run the App - Step by Step

## Step 1: Stop any running Streamlit instances

Press Ctrl+C in the terminal where Streamlit is running.

Or if it's running in background:
```bash
# Kill all Python processes (if needed)
taskkill /F /IM python.exe
```

## Step 2: Open a fresh terminal

Make sure you have a clean terminal window.

## Step 3: Navigate to the progression directory

```bash
cd D:\Major_Project\FL_QPSO_FedAvg\progression
```

## Step 4: Verify data exists

```bash
# Check prediction index file
dir streamlit_data\prediction_index.json

# Or use Python
python diagnostic_check.py
```

Should show:
```
Successfully loaded: 111 patients
```

## Step 5: Clear Streamlit cache (if you see old errors)

```bash
python -m streamlit cache clear
```

## Step 6: Run the app with the launcher

```bash
python run_streamlit_app.py
```

This will:
- Verify the working directory is correct
- Check that data files exist
- Start Streamlit from the correct directory
- Open browser to http://localhost:8501

## Step 7: Verify it loads

You should see:
- "3D Tumor Growth Prediction" title
- Patient selection sidebar
- No error messages

If you still see "Prediction data not found":
- Check terminal output for DEBUG messages
- Make sure you're in the right directory
- Run `python src/08_generate_viz_data.py` to regenerate data

## Alternative: Direct Streamlit command

If the launcher doesn't work, try:
```bash
cd D:\Major_Project\FL_QPSO_FedAvg\progression
streamlit run streamlit_3d_progression.py --logger.level=debug
```

This will show debug output if there are any issues.

## If Still Having Issues

Run this diagnostic:
```bash
python diagnostic_check.py
```

If it says "Successfully loaded", the data is fine and it's a Streamlit caching issue.

Solutions:
1. Clear cache: `python -m streamlit cache clear`
2. Restart terminal completely
3. Try on a different port: `streamlit run streamlit_3d_progression.py --server.port 8502`
4. Check Windows Task Manager - make sure no python.exe is still running
