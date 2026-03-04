# TUMOR TIME TRAVEL: COMPLETE PROGRESSION PREDICTION GUIDE
## Brain Tumor Growth & Shrinkage Forecasting System

**Version:** 1.0  
**Platform:** Kaggle (2x Tesla T4 GPUs)  
**Integration:** Works with FL-QPSO Classification System  
**Timeline:** 3-4 weeks implementation

---

# TABLE OF CONTENTS

1. [Overview & Objectives](#1-overview--objectives)
2. [Understanding Tumor Progression](#2-understanding-tumor-progression)
3. [Datasets for Longitudinal Analysis](#3-datasets-for-longitudinal-analysis)
4. [Implementation Path A: Mathematical Models](#4-implementation-path-a-mathematical-models)
5. [Implementation Path B: Deep Learning](#5-implementation-path-b-deep-learning)
6. [Integration with FL-QPSO System](#6-integration-with-fl-qpso-system)
7. [Clinical Decision Support](#7-clinical-decision-support)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Visualization Dashboard](#9-visualization-dashboard)
10. [Complete Code Implementation](#10-complete-code-implementation)

---

# 1. OVERVIEW & OBJECTIVES

## 1.1 What is Tumor Time Travel?

**Tumor Time Travel** = Predicting future tumor growth or shrinkage based on historical MRI scans.

**Real-World Problem:**
- Doctors need to know: **"Will this tumor grow dangerously in the next 3-6 months?"**
- Current situation: Manual comparison of scans, subjective assessments
- Your solution: **Automated growth prediction + risk alerts**

## 1.2 Clinical Impact

| Use Case | How It Helps |
|----------|-------------|
| **Surgery Planning** | Identify patients needing urgent intervention |
| **Treatment Monitoring** | Track if therapy is working (shrinking vs growing) |
| **Recurrence Detection** | Early warning if tumor returns post-surgery |
| **Resource Allocation** | Prioritize high-risk patients |

## 1.3 What You'll Build

```
Input: Multiple MRI scans from same patient (t1, t2, t3, ...)
  ↓
Segment tumor in each scan
  ↓
Extract tumor volumes + features
  ↓
Fit growth model (exponential, AI-based, etc.)
  ↓
Predict future volumes (t_future)
  ↓
Output: Growth curve + risk alert + recommended action
```

## 1.4 Two Implementation Approaches

### **Path A: Mathematical Models** ⚡ FASTER
- **Time**: 2-3 weeks
- **Complexity**: Medium
- **Accuracy**: Good for smooth growth patterns
- **Best for**: Getting results quickly, interpretable

**Models:**
- Exponential growth: V(t) = V₀ × e^(kt)
- Gompertz model: V(t) = V_max × e^(-ln(V_max/V₀) × e^(-kt))
- Logistic growth: V(t) = V_max / (1 + e^(-k(t-t₀)))

### **Path B: Deep Learning** 🚀 ADVANCED
- **Time**: 4-6 weeks
- **Complexity**: High
- **Accuracy**: Better for complex patterns
- **Best for**: Research publication, advanced predictions

**Models:**
- LSTM (Long Short-Term Memory)
- Transformer for time series
- 3D CNN-LSTM hybrid
- Graph Neural Networks (tumor shape evolution)

---

# 2. UNDERSTANDING TUMOR PROGRESSION

## 2.1 Biology of Tumor Growth

**Key Concepts:**

1. **Exponential Phase**: Early growth, unlimited resources
   - V(t) = V₀ × 2^(t/T_d) where T_d = doubling time

2. **Plateau Phase**: Growth slows due to resource limits
   - Follows Gompertz or logistic curves

3. **Treatment Effects**:
   - **Regression**: Tumor shrinks (good response)
   - **Stable**: No significant change
   - **Progression**: Tumor grows (bad response)

## 2.2 RANO Criteria (Response Assessment)

Standard clinical criteria for gliomas:

| Status | Criteria |
|--------|----------|
| **Complete Response (CR)** | No visible tumor |
| **Partial Response (PR)** | ≥50% decrease in tumor volume |
| **Stable Disease (SD)** | <50% decrease and <25% increase |
| **Progressive Disease (PD)** | ≥25% increase in tumor volume |

Your system should output **RANO status** automatically.

## 2.3 Growth Rate Metrics

**Key metrics to track:**

1. **Absolute Growth Rate (AGR)**:
   ```
   AGR = (V_t2 - V_t1) / Δt  [cm³/month]
   ```

2. **Relative Growth Rate (RGR)**:
   ```
   RGR = (V_t2 - V_t1) / V_t1  [%]
   ```

3. **Doubling Time (T_d)**:
   ```
   T_d = ln(2) / k  [months]
   ```
   where k is exponential growth constant

4. **Velocity of Diametric Expansion (VDE)**:
   ```
   VDE = (D_t2 - D_t1) / Δt  [mm/month]
   ```

---

# 3. DATASETS FOR LONGITUDINAL ANALYSIS

## 3.1 Recommended Datasets

### **Option 1: MU-Glioma-Post (Best for Beginners)**

**Source**: The Cancer Imaging Archive (TCIA)  
**URL**: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230229

**Details:**
- **Patients**: 65 post-treatment glioma patients
- **Timepoints**: 2-6 per patient
- **Modalities**: T1, T1ce, T2, FLAIR
- **Labels**: Tumor progression status (PD/SD/PR)
- **Best for**: Recurrence prediction, treatment response

**Download:**
```bash
# Install TCIA downloader
pip install tcia-utils

# Python script to download
from tcia_utils import nbia
nbia.downloadSeries(
    series_data=nbia.getSeries(collection="MU-Glioma-Post"),
    input_type="df",
    path="/kaggle/working/data/MU-Glioma-Post"
)
```

### **Option 2: LUMIERE**

**Source**: TCIA  
**URL**: https://wiki.cancerimagingarchive.net/display/Public/LUMIERE

**Details:**
- **Patients**: 30+ glioblastoma patients
- **Timepoints**: Multiple pre- and post-treatment
- **Modalities**: T1, T1ce, T2, FLAIR
- **Labels**: Segmentation masks, survival data
- **Best for**: Advanced progression modeling

### **Option 3: UCSD-PTGBM**

**Source**: TCIA  
**Details:**
- **Patients**: 50+ pre-treatment GBM
- **Timepoints**: 2-4 per patient
- **Best for**: Treatment planning, pre-surgery prediction

## 3.2 Dataset Structure

**Expected organization:**
```
data/
├── patient_001/
│   ├── timepoint_01/
│   │   ├── T1.nii.gz
│   │   ├── T1ce.nii.gz
│   │   ├── T2.nii.gz
│   │   ├── FLAIR.nii.gz
│   │   └── segmentation.nii.gz
│   ├── timepoint_02/
│   │   └── [same structure]
│   └── timepoint_03/
│       └── [same structure]
├── patient_002/
│   └── [same structure]
...
```

## 3.3 Data Preprocessing Pipeline

**Step-by-step preprocessing:**

```python
import nibabel as nib
import numpy as np
from scipy import ndimage

def preprocess_longitudinal_scan(scan_path, reference_scan=None):
    """
    Preprocess individual MRI scan for longitudinal analysis
    
    Args:
        scan_path: path to .nii.gz file
        reference_scan: optional baseline scan for registration
    
    Returns:
        preprocessed numpy array
    """
    # Load scan
    img = nib.load(scan_path)
    data = img.get_fdata()
    
    # 1. Skull stripping (if needed)
    # Use bet2 or deep learning-based method
    
    # 2. N4 bias field correction
    # from SimpleITK import N4BiasFieldCorrectionImageFilter
    
    # 3. Intensity normalization
    data = (data - np.mean(data)) / np.std(data)
    
    # 4. Registration to baseline (if reference provided)
    if reference_scan is not None:
        # Use ANTs or SimpleITK for registration
        # This ensures all timepoints are aligned
        pass
    
    # 5. Resample to consistent spacing (e.g., 1x1x1 mm)
    # Use scipy.ndimage.zoom
    
    return data
```

**Key preprocessing steps:**
1. **Co-registration**: Align all timepoints to baseline scan
2. **Intensity normalization**: Ensure consistent brightness
3. **Resampling**: Same voxel spacing across scans
4. **Skull stripping**: Remove non-brain tissue

---

# 4. IMPLEMENTATION PATH A: MATHEMATICAL MODELS

## 4.1 Overview

**Approach**: Use classical growth models to fit tumor volume time series.

**Pros:**
- Fast implementation (1-2 weeks)
- Interpretable parameters
- Works well with limited data (2-3 timepoints)
- Low computational requirements

**Cons:**
- Assumes smooth growth patterns
- Limited to volume-based predictions
- Cannot capture complex spatial changes

## 4.2 Volume Extraction

**Cell 1: Extract Tumor Volumes**

```python
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

def extract_tumor_volume(segmentation_path, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Extract tumor volume from segmentation mask
    
    Args:
        segmentation_path: path to segmentation .nii.gz
        voxel_spacing: (x, y, z) spacing in mm
    
    Returns:
        volume in cm³
    """
    # Load segmentation
    seg = nib.load(segmentation_path)
    seg_data = seg.get_fdata()
    
    # Count tumor voxels (assuming label=1 for tumor)
    tumor_voxels = np.sum(seg_data > 0)
    
    # Calculate voxel volume
    voxel_volume_mm3 = np.prod(voxel_spacing)
    
    # Convert to cm³
    volume_cm3 = (tumor_voxels * voxel_volume_mm3) / 1000.0
    
    return volume_cm3

def extract_all_volumes(data_dir):
    """
    Extract volumes for all patients and timepoints
    
    Returns:
        DataFrame with columns: patient_id, timepoint, days, volume_cm3
    """
    records = []
    
    for patient_dir in sorted(os.listdir(data_dir)):
        patient_path = os.path.join(data_dir, patient_dir)
        
        if not os.path.isdir(patient_path):
            continue
        
        timepoints = sorted([d for d in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, d))])
        
        for i, tp in enumerate(timepoints):
            seg_path = os.path.join(patient_path, tp, 'segmentation.nii.gz')
            
            if os.path.exists(seg_path):
                volume = extract_tumor_volume(seg_path)
                
                # Assuming timepoints are named with dates or scan numbers
                # Extract days from baseline (you'll need to parse this from metadata)
                days_from_baseline = i * 90  # Example: assume 90-day intervals
                
                records.append({
                    'patient_id': patient_dir,
                    'timepoint': i,
                    'days': days_from_baseline,
                    'volume_cm3': volume
                })
    
    df = pd.DataFrame(records)
    return df

# Execute
volume_df = extract_all_volumes('/kaggle/working/data/MU-Glioma-Post')
print(volume_df.head(20))

# Save
volume_df.to_csv('/kaggle/working/results/progression/tumor_volumes.csv', index=False)
```

## 4.3 Mathematical Growth Models

**Cell 2: Define Growth Models**

```python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class TumorGrowthModels:
    """
    Collection of mathematical models for tumor growth
    """
    
    @staticmethod
    def exponential(t, V0, k):
        """
        Exponential growth: V(t) = V0 * exp(k*t)
        
        Args:
            t: time (days)
            V0: initial volume
            k: growth rate constant (1/days)
        """
        return V0 * np.exp(k * t)
    
    @staticmethod
    def gompertz(t, V0, Vmax, k):
        """
        Gompertz growth: V(t) = Vmax * exp(-ln(Vmax/V0) * exp(-k*t))
        
        Args:
            t: time
            V0: initial volume
            Vmax: carrying capacity (maximum volume)
            k: growth rate
        """
        return Vmax * np.exp(-np.log(Vmax / V0) * np.exp(-k * t))
    
    @staticmethod
    def logistic(t, V0, Vmax, k):
        """
        Logistic growth: V(t) = Vmax / (1 + ((Vmax/V0 - 1) * exp(-k*t)))
        
        Args:
            t: time
            V0: initial volume
            Vmax: carrying capacity
            k: growth rate
        """
        return Vmax / (1 + ((Vmax / V0) - 1) * np.exp(-k * t))
    
    @staticmethod
    def linear(t, V0, k):
        """
        Linear growth: V(t) = V0 + k*t
        
        Args:
            t: time
            V0: initial volume
            k: absolute growth rate (cm³/day)
        """
        return V0 + k * t
    
    @staticmethod
    def power_law(t, V0, alpha):
        """
        Power law: V(t) = V0 * t^alpha
        
        Args:
            t: time
            V0: scaling constant
            alpha: power exponent
        """
        return V0 * np.power(t + 1, alpha)  # +1 to avoid t=0 issues

# Test models
models = TumorGrowthModels()
print("✅ Growth models defined")
```

## 4.4 Fit Models to Data

**Cell 3: Model Fitting Function**

```python
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def fit_growth_model(times, volumes, model_func, initial_guess=None, bounds=None):
    """
    Fit growth model to tumor volume data
    
    Args:
        times: array of timepoints (days)
        volumes: array of tumor volumes (cm³)
        model_func: function to fit (from TumorGrowthModels)
        initial_guess: initial parameter guesses
        bounds: parameter bounds
    
    Returns:
        params: fitted parameters
        r2: R² score
        predictions: predicted volumes
    """
    try:
        # Fit model
        if bounds is not None:
            params, _ = curve_fit(model_func, times, volumes, p0=initial_guess, bounds=bounds, maxfev=10000)
        else:
            params, _ = curve_fit(model_func, times, volumes, p0=initial_guess, maxfev=10000)
        
        # Predict
        predictions = model_func(times, *params)
        
        # Calculate R²
        r2 = r2_score(volumes, predictions)
        mae = mean_absolute_error(volumes, predictions)
        rmse = np.sqrt(mean_squared_error(volumes, predictions))
        
        return {
            'params': params,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'predictions': predictions
        }
    
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None

def fit_all_models_for_patient(patient_data):
    """
    Fit all growth models for a single patient
    
    Args:
        patient_data: DataFrame with columns ['days', 'volume_cm3']
    
    Returns:
        Dictionary of results for each model
    """
    times = patient_data['days'].values
    volumes = patient_data['volume_cm3'].values
    
    results = {}
    
    # 1. Exponential model
    try:
        exp_result = fit_growth_model(
            times, volumes,
            TumorGrowthModels.exponential,
            initial_guess=[volumes[0], 0.001],
            bounds=([0, -0.1], [np.inf, 0.1])
        )
        results['exponential'] = exp_result
    except:
        results['exponential'] = None
    
    # 2. Gompertz model
    try:
        gomp_result = fit_growth_model(
            times, volumes,
            TumorGrowthModels.gompertz,
            initial_guess=[volumes[0], volumes[-1] * 2, 0.001],
            bounds=([0, volumes[-1], 0], [np.inf, np.inf, 0.1])
        )
        results['gompertz'] = gomp_result
    except:
        results['gompertz'] = None
    
    # 3. Logistic model
    try:
        log_result = fit_growth_model(
            times, volumes,
            TumorGrowthModels.logistic,
            initial_guess=[volumes[0], volumes[-1] * 2, 0.001],
            bounds=([0, volumes[-1], 0], [np.inf, np.inf, 0.1])
        )
        results['logistic'] = log_result
    except:
        results['logistic'] = None
    
    # 4. Linear model
    try:
        lin_result = fit_growth_model(
            times, volumes,
            TumorGrowthModels.linear,
            initial_guess=[volumes[0], (volumes[-1] - volumes[0]) / (times[-1] - times[0])],
            bounds=([-np.inf, -np.inf], [np.inf, np.inf])
        )
        results['linear'] = lin_result
    except:
        results['linear'] = None
    
    return results

# Test on a single patient
test_patient = volume_df[volume_df['patient_id'] == volume_df['patient_id'].unique()[0]]
test_results = fit_all_models_for_patient(test_patient)

print("Model fitting results:")
for model_name, result in test_results.items():
    if result is not None:
        print(f"{model_name}: R²={result['r2']:.4f}, MAE={result['mae']:.4f} cm³")
```

## 4.5 Predict Future Growth

**Cell 4: Growth Prediction Function**

```python
def predict_future_growth(patient_data, prediction_days=180, model='best'):
    """
    Predict future tumor growth
    
    Args:
        patient_data: DataFrame with patient's volume history
        prediction_days: days into future to predict
        model: 'best', 'exponential', 'gompertz', 'logistic', or 'linear'
    
    Returns:
        Dictionary with predictions and metadata
    """
    times = patient_data['days'].values
    volumes = patient_data['volume_cm3'].values
    
    # Fit all models
    model_results = fit_all_models_for_patient(patient_data)
    
    # Select best model (highest R²)
    if model == 'best':
        best_model_name = None
        best_r2 = -np.inf
        
        for model_name, result in model_results.items():
            if result is not None and result['r2'] > best_r2:
                best_r2 = result['r2']
                best_model_name = model_name
        
        model = best_model_name
    
    # Get selected model results
    selected_result = model_results[model]
    
    if selected_result is None:
        print(f"Model {model} fitting failed")
        return None
    
    # Generate future timepoints
    last_time = times[-1]
    future_times = np.arange(last_time + 30, last_time + prediction_days + 30, 30)  # Monthly predictions
    all_times = np.concatenate([times, future_times])
    
    # Get model function
    model_funcs = {
        'exponential': TumorGrowthModels.exponential,
        'gompertz': TumorGrowthModels.gompertz,
        'logistic': TumorGrowthModels.logistic,
        'linear': TumorGrowthModels.linear
    }
    
    model_func = model_funcs[model]
    
    # Predict
    historical_predictions = model_func(times, *selected_result['params'])
    future_predictions = model_func(future_times, *selected_result['params'])
    
    # Calculate growth metrics
    last_volume = volumes[-1]
    predicted_volume_6mo = future_predictions[5] if len(future_predictions) >= 6 else future_predictions[-1]
    
    absolute_growth = predicted_volume_6mo - last_volume
    relative_growth = (predicted_volume_6mo - last_volume) / last_volume * 100
    
    # Determine RANO status
    if relative_growth >= 25:
        rano_status = "Progressive Disease (PD)"
        risk_level = "HIGH"
        recommendation = "Consider immediate intervention"
    elif relative_growth <= -50:
        rano_status = "Partial Response (PR)"
        risk_level = "LOW"
        recommendation = "Continue current treatment"
    elif -50 < relative_growth < 25:
        rano_status = "Stable Disease (SD)"
        risk_level = "MEDIUM"
        recommendation = "Continue monitoring"
    
    # Calculate doubling time (for exponential model)
    doubling_time = None
    if model == 'exponential' and selected_result['params'][1] > 0:
        k = selected_result['params'][1]
        doubling_time = np.log(2) / k  # days
    
    return {
        'model_used': model,
        'r2_score': selected_result['r2'],
        'mae': selected_result['mae'],
        'historical_times': times,
        'historical_volumes': volumes,
        'historical_predictions': historical_predictions,
        'future_times': future_times,
        'future_predictions': future_predictions,
        'current_volume': last_volume,
        'predicted_6mo_volume': predicted_volume_6mo,
        'absolute_growth_6mo': absolute_growth,
        'relative_growth_6mo': relative_growth,
        'rano_status': rano_status,
        'risk_level': risk_level,
        'recommendation': recommendation,
        'doubling_time_days': doubling_time
    }

# Test prediction
test_patient = volume_df[volume_df['patient_id'] == volume_df['patient_id'].unique()[0]]
prediction = predict_future_growth(test_patient, prediction_days=180)

if prediction:
    print(f"Model used: {prediction['model_used']}")
    print(f"R² score: {prediction['r2_score']:.4f}")
    print(f"Current volume: {prediction['current_volume']:.2f} cm³")
    print(f"Predicted 6-month volume: {prediction['predicted_6mo_volume']:.2f} cm³")
    print(f"Relative growth: {prediction['relative_growth_6mo']:.1f}%")
    print(f"RANO status: {prediction['rano_status']}")
    print(f"Risk level: {prediction['risk_level']}")
    print(f"Recommendation: {prediction['recommendation']}")
    if prediction['doubling_time_days']:
        print(f"Tumor doubling time: {prediction['doubling_time_days']:.1f} days")
```

## 4.6 Visualize Growth Predictions

**Cell 5: Visualization**

```python
def visualize_growth_prediction(prediction, patient_id, save_path=None):
    """
    Create comprehensive growth visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Tumor Growth Analysis - Patient {patient_id}', fontsize=16, fontweight='bold')
    
    # Plot 1: Growth curve with prediction
    ax1 = axes[0, 0]
    ax1.scatter(prediction['historical_times'], prediction['historical_volumes'], 
                c='blue', s=100, label='Observed', zorder=3)
    ax1.plot(prediction['historical_times'], prediction['historical_predictions'], 
             'b--', linewidth=2, label='Model fit', alpha=0.7)
    ax1.plot(prediction['future_times'], prediction['future_predictions'], 
             'r-', linewidth=2, label='Prediction', alpha=0.7)
    ax1.fill_between(prediction['future_times'], 
                     prediction['future_predictions'] * 0.9,
                     prediction['future_predictions'] * 1.1,
                     color='red', alpha=0.2, label='Uncertainty')
    ax1.set_xlabel('Days from Baseline', fontsize=12)
    ax1.set_ylabel('Tumor Volume (cm³)', fontsize=12)
    ax1.set_title(f'Growth Prediction ({prediction["model_used"]} model, R²={prediction["r2_score"]:.3f})', 
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Growth rate over time
    ax2 = axes[0, 1]
    all_times = np.concatenate([prediction['historical_times'], prediction['future_times']])
    all_volumes = np.concatenate([prediction['historical_predictions'], prediction['future_predictions']])
    growth_rates = np.gradient(all_volumes) / np.gradient(all_times)
    
    ax2.plot(all_times, growth_rates, linewidth=2, color='green')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(x=prediction['historical_times'][-1], color='red', linestyle='--', 
                alpha=0.5, label='Current time')
    ax2.set_xlabel('Days from Baseline', fontsize=12)
    ax2.set_ylabel('Growth Rate (cm³/day)', fontsize=12)
    ax2.set_title('Instantaneous Growth Rate', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Risk assessment
    ax3 = axes[1, 0]
    risk_colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
    risk_color = risk_colors[prediction['risk_level']]
    
    metrics = {
        'Current\nVolume': prediction['current_volume'],
        '6-Month\nPredicted': prediction['predicted_6mo_volume'],
        'Absolute\nGrowth': prediction['absolute_growth_6mo'],
        'Relative\nGrowth (%)': prediction['relative_growth_6mo']
    }
    
    bars = ax3.bar(range(len(metrics)), list(metrics.values()), 
                   color=['blue', 'orange', 'red', 'purple'])
    ax3.set_xticks(range(len(metrics)))
    ax3.set_xticklabels(list(metrics.keys()), rotation=0)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Growth Metrics Summary', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, (bar, value) in enumerate(zip(bars, metrics.values())):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Clinical decision panel
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create text summary
    summary_text = f"""
CLINICAL ASSESSMENT
{'='*50}

RANO Status: {prediction['rano_status']}
Risk Level: {prediction['risk_level']}

MEASUREMENTS:
• Current Volume: {prediction['current_volume']:.2f} cm³
• Predicted 6-Month: {prediction['predicted_6mo_volume']:.2f} cm³
• Absolute Growth: {prediction['absolute_growth_6mo']:.2f} cm³
• Relative Growth: {prediction['relative_growth_6mo']:.1f}%

MODEL PERFORMANCE:
• Model Used: {prediction['model_used']}
• R² Score: {prediction['r2_score']:.4f}
• Mean Absolute Error: {prediction['mae']:.2f} cm³

RECOMMENDATION:
{prediction['recommendation']}
    """
    
    # Add colored box for risk level
    box_props = dict(boxstyle='round', facecolor=risk_color, alpha=0.3)
    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            bbox=box_props, family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Test visualization
visualize_growth_prediction(
    prediction, 
    test_patient['patient_id'].iloc[0],
    save_path='/kaggle/working/results/progression/growth_prediction_sample.png'
)
```

## 4.7 Batch Processing All Patients

**Cell 6: Process All Patients**

```python
def process_all_patients(volume_df, output_dir='/kaggle/working/results/progression'):
    """
    Process all patients and generate predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_predictions = []
    patient_ids = volume_df['patient_id'].unique()
    
    print(f"Processing {len(patient_ids)} patients...")
    
    for patient_id in tqdm(patient_ids):
        patient_data = volume_df[volume_df['patient_id'] == patient_id]
        
        # Need at least 2 timepoints
        if len(patient_data) < 2:
            print(f"Skipping {patient_id}: insufficient data")
            continue
        
        # Predict
        prediction = predict_future_growth(patient_data, prediction_days=180)
        
        if prediction is None:
            continue
        
        # Add patient ID
        prediction['patient_id'] = patient_id
        
        # Visualize
        viz_path = os.path.join(output_dir, f'{patient_id}_growth_prediction.png')
        visualize_growth_prediction(prediction, patient_id, save_path=viz_path)
        
        # Store summary
        all_predictions.append({
            'patient_id': patient_id,
            'model_used': prediction['model_used'],
            'r2_score': prediction['r2_score'],
            'current_volume': prediction['current_volume'],
            'predicted_6mo_volume': prediction['predicted_6mo_volume'],
            'absolute_growth': prediction['absolute_growth_6mo'],
            'relative_growth': prediction['relative_growth_6mo'],
            'rano_status': prediction['rano_status'],
            'risk_level': prediction['risk_level'],
            'recommendation': prediction['recommendation'],
            'doubling_time_days': prediction['doubling_time_days']
        })
    
    # Save summary
    summary_df = pd.DataFrame(all_predictions)
    summary_df.to_csv(os.path.join(output_dir, 'all_patients_summary.csv'), index=False)
    
    print(f"\n✅ Processed {len(all_predictions)} patients")
    print(f"Results saved to {output_dir}")
    
    return summary_df

# Execute
summary_df = process_all_patients(volume_df)

# Display summary statistics
print("\n" + "="*80)
print("COHORT SUMMARY")
print("="*80)
print(f"Total patients analyzed: {len(summary_df)}")
print(f"\nRisk distribution:")
print(summary_df['risk_level'].value_counts())
print(f"\nRANO status distribution:")
print(summary_df['rano_status'].value_counts())
print(f"\nMean predicted 6-month growth: {summary_df['relative_growth'].mean():.1f}%")
print(f"Median predicted 6-month growth: {summary_df['relative_growth'].median():.1f}%")
```

---

# 5. IMPLEMENTATION PATH B: DEEP LEARNING

## 5.1 Overview

**Approach**: Use recurrent or transformer networks to learn complex growth patterns.

**Advantages over Path A:**
- Learn non-linear, complex patterns
- Can incorporate spatial information (not just volume)
- Handle irregular time intervals better
- More accurate for heterogeneous growth

**Models to Try:**

1. **LSTM (Long Short-Term Memory)**
   - Good for time series
   - Handles variable-length sequences
   - Can predict multiple steps ahead

2. **Transformer**
   - Attention mechanism
   - Better for long-range dependencies
   - State-of-the-art for sequences

3. **3D CNN-LSTM Hybrid**
   - Extracts spatial features from 3D scans
   - LSTM for temporal modeling
   - Predicts future 3D volumes

## 5.2 LSTM for Volume Prediction

**Cell 7: LSTM Model**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TumorGrowthLSTM(nn.Module):
    """
    LSTM network for tumor growth prediction
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(TumorGrowthLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch_size, sequence_length, input_size)
        
        Returns:
            predictions: (batch_size, output_size)
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Fully connected
        output = self.fc(last_hidden)
        
        return output

# Test model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TumorGrowthLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)
model = model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("✅ LSTM model created")
```

## 5.3 Dataset for LSTM Training

**Cell 8: Create Dataset**

```python
class TumorSequenceDataset(Dataset):
    """
    Dataset for time series tumor volume prediction
    """
    
    def __init__(self, volume_df, sequence_length=3, prediction_steps=1):
        """
        Args:
            volume_df: DataFrame with patient_id, days, volume_cm3
            sequence_length: number of historical timepoints to use
            prediction_steps: number of future steps to predict
        """
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.sequences = []
        self.targets = []
        
        # Create sequences for each patient
        for patient_id in volume_df['patient_id'].unique():
            patient_data = volume_df[volume_df['patient_id'] == patient_id].sort_values('days')
            
            volumes = patient_data['volume_cm3'].values
            days = patient_data['days'].values
            
            # Need at least sequence_length + prediction_steps
            if len(volumes) < sequence_length + prediction_steps:
                continue
            
            # Create sliding windows
            for i in range(len(volumes) - sequence_length - prediction_steps + 1):
                # Input sequence
                seq = volumes[i:i + sequence_length]
                # Target (next value)
                target = volumes[i + sequence_length:i + sequence_length + prediction_steps]
                
                self.sequences.append(seq)
                self.targets.append(target)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        
        # Normalize
        self.mean = np.mean(self.sequences)
        self.std = np.std(self.sequences)
        
        self.sequences = (self.sequences - self.mean) / (self.std + 1e-8)
        self.targets = (self.targets - self.mean) / (self.std + 1e-8)
        
        print(f"Created {len(self.sequences)} sequences")
        print(f"Sequence shape: {self.sequences.shape}")
        print(f"Target shape: {self.targets.shape}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].reshape(-1, 1)  # (sequence_length, 1)
        target = self.targets[idx]  # (prediction_steps,)
        
        return torch.FloatTensor(seq), torch.FloatTensor(target)
    
    def denormalize(self, normalized_value):
        """Convert normalized value back to original scale"""
        return normalized_value * self.std + self.mean

# Create dataset
dataset = TumorSequenceDataset(volume_df, sequence_length=3, prediction_steps=1)

# Split into train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Train set: {len(train_dataset)} sequences")
print(f"Val set: {len(val_dataset)} sequences")
```

## 5.4 Train LSTM

**Cell 9: Training Loop**

```python
def train_lstm(model, train_loader, val_loader, epochs=100, learning_rate=0.001):
    """
    Train LSTM for tumor growth prediction
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            # Forward
            optimizer.zero_grad()
            predictions = model(sequences)
            
            loss = criterion(predictions.squeeze(), targets.squeeze())
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                
                predictions = model(sequences)
                loss = criterion(predictions.squeeze(), targets.squeeze())
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/kaggle/working/models/lstm_best.pth')
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
    
    return history

# Train
history = train_lstm(model, train_loader, val_loader, epochs=100, learning_rate=0.001)

# Plot training curves
plt.figure(figsize=(10, 6))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('LSTM Training History')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/kaggle/working/results/progression/lstm_training.png', dpi=150)
plt.show()

print("✅ LSTM training completed")
```

## 5.5 LSTM Prediction

**Cell 10: Use LSTM for Prediction**

```python
def predict_with_lstm(model, patient_data, dataset, future_steps=6):
    """
    Use trained LSTM to predict future tumor volumes
    
    Args:
        model: trained LSTM model
        patient_data: DataFrame with patient's volume history
        dataset: TumorSequenceDataset (for normalization)
        future_steps: number of future timepoints to predict
    
    Returns:
        predictions dictionary
    """
    model.eval()
    
    volumes = patient_data['volume_cm3'].values
    days = patient_data['days'].values
    
    # Normalize
    volumes_norm = (volumes - dataset.mean) / (dataset.std + 1e-8)
    
    # Use last sequence_length values as input
    sequence_length = dataset.sequence_length
    input_seq = volumes_norm[-sequence_length:].reshape(1, sequence_length, 1)
    input_seq = torch.FloatTensor(input_seq).to(device)
    
    # Predict iteratively
    predictions_norm = []
    current_seq = input_seq.clone()
    
    with torch.no_grad():
        for _ in range(future_steps):
            # Predict next value
            pred = model(current_seq)
            predictions_norm.append(pred.item())
            
            # Update sequence (sliding window)
            current_seq = torch.cat([current_seq[:, 1:, :], pred.reshape(1, 1, 1)], dim=1)
    
    # Denormalize
    predictions = [dataset.denormalize(p) for p in predictions_norm]
    
    # Create future timepoints (assume monthly intervals)
    last_day = days[-1]
    future_days = [last_day + 30 * (i+1) for i in range(future_steps)]
    
    return {
        'historical_days': days,
        'historical_volumes': volumes,
        'future_days': np.array(future_days),
        'future_volumes': np.array(predictions),
        'model': 'LSTM'
    }

# Test LSTM prediction
test_patient = volume_df[volume_df['patient_id'] == volume_df['patient_id'].unique()[0]]
lstm_prediction = predict_with_lstm(model, test_patient, dataset, future_steps=6)

# Visualize
plt.figure(figsize=(12, 6))
plt.scatter(lstm_prediction['historical_days'], lstm_prediction['historical_volumes'], 
            c='blue', s=100, label='Observed', zorder=3)
plt.plot(lstm_prediction['future_days'], lstm_prediction['future_volumes'], 
         'r-o', linewidth=2, markersize=8, label='LSTM Prediction')
plt.xlabel('Days from Baseline')
plt.ylabel('Tumor Volume (cm³)')
plt.title('LSTM-based Tumor Growth Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/kaggle/working/results/progression/lstm_prediction_sample.png', dpi=150)
plt.show()
```

---

# 6. INTEGRATION WITH FL-QPSO SYSTEM

## 6.1 Complete Pipeline

**How Progression Prediction Integrates with Classification:**

```
Patient MRI Scans (multiple timepoints)
    ↓
┌─────────────────────┐
│ CLASSIFICATION      │  (Your existing FL-QPSO system)
│ - Identifies tumor  │
│   type (Glioma,     │
│   Meningioma, etc.) │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ SEGMENTATION        │
│ - Extract tumor     │
│   boundaries        │
│ - Calculate volume  │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ PROGRESSION         │  (New component you're adding)
│ - Track volume      │
│   changes           │
│ - Predict growth    │
│ - Risk assessment   │
└─────────────────────┘
    ↓
Clinical Decision Support
- Tumor type + Growth prediction → Treatment plan
```

## 6.2 Combined Analysis Function

**Cell 11: Integrated System**

```python
def comprehensive_tumor_analysis(patient_scans, classification_model, segmentation_model, progression_model):
    """
    Complete analysis pipeline combining classification and progression
    
    Args:
        patient_scans: list of scan paths (multiple timepoints)
        classification_model: your FL-QPSO trained classifier
        segmentation_model: tumor segmentation model
        progression_model: trained progression model (LSTM or mathematical)
    
    Returns:
        Complete analysis report
    """
    results = {
        'patient_id': None,
        'num_timepoints': len(patient_scans),
        'tumor_type': None,
        'tumor_type_confidence': None,
        'volumes_over_time': [],
        'growth_prediction': None,
        'risk_assessment': None,
        'recommended_action': None
    }
    
    volumes = []
    days = []
    
    # Process each scan
    for i, scan_path in enumerate(patient_scans):
        # 1. Load scan
        scan = nib.load(scan_path).get_fdata()
        
        # 2. Classify tumor type (use your FL-QPSO model)
        # (Assuming you have preprocessing ready)
        tumor_type, confidence = classify_scan(scan, classification_model)
        
        if i == 0:  # Use first scan's classification
            results['tumor_type'] = tumor_type
            results['tumor_type_confidence'] = confidence
        
        # 3. Segment tumor
        segmentation = segment_tumor(scan, segmentation_model)
        
        # 4. Calculate volume
        volume = calculate_volume(segmentation)
        volumes.append(volume)
        days.append(i * 90)  # Assuming 90-day intervals
    
    results['volumes_over_time'] = volumes
    
    # 5. Growth prediction
    patient_volume_data = pd.DataFrame({
        'days': days,
        'volume_cm3': volumes
    })
    
    growth_pred = predict_future_growth(patient_volume_data, prediction_days=180)
    results['growth_prediction'] = growth_pred
    
    # 6. Risk assessment combining type and growth
    risk_factors = {
        'glioma': 3,  # High risk
        'meningioma': 1,  # Low risk
        'pituitary': 2  # Medium risk
    }
    
    type_risk = risk_factors.get(tumor_type.lower(), 2)
    growth_risk = 3 if growth_pred['risk_level'] == 'HIGH' else (2 if growth_pred['risk_level'] == 'MEDIUM' else 1)
    
    combined_risk = (type_risk + growth_risk) / 2
    
    if combined_risk >= 2.5:
        results['risk_assessment'] = 'HIGH RISK'
        results['recommended_action'] = 'Immediate surgical consultation recommended'
    elif combined_risk >= 1.5:
        results['risk_assessment'] = 'MODERATE RISK'
        results['recommended_action'] = 'Continue monitoring, consider treatment escalation'
    else:
        results['risk_assessment'] = 'LOW RISK'
        results['recommended_action'] = 'Continue routine monitoring'
    
    return results

# This function integrates your classification with progression prediction
print("✅ Integrated analysis pipeline defined")
```

## 6.3 Federated Learning for Progression Models

**Extending FL to Progression:**

```python
"""
You can also apply federated learning to train the progression model
across multiple hospitals, just like you did for classification!
"""

class ProgressionFederatedClient:
    """
    Client for federated training of progression models
    Similar to your classification client, but for time series
    """
    
    def __init__(self, client_id, train_data, val_data, device='cuda'):
        self.client_id = client_id
        self.train_data = train_data  # Time series data
        self.val_data = val_data
        self.device = device
        self.model = None  # LSTM or Transformer
        
    def train_local(self, epochs=10):
        """Train progression model locally on hospital data"""
        # Similar to your classification training
        # But optimized for time series
        pass
    
    def validate(self):
        """Validate on local longitudinal data"""
        pass

# Combine FL-QPSO with progression:
# 1. Each hospital trains classification model (your current work)
# 2. Each hospital ALSO trains progression model on their longitudinal data
# 3. Both models aggregate at server
# 4. Result: Privacy-preserving system for both classification AND progression
```

---

# 7. CLINICAL DECISION SUPPORT

## 7.1 Risk Stratification System

**Cell 12: Risk Calculator**

```python
class ClinicalRiskCalculator:
    """
    Calculate comprehensive risk score for surgical decision
    """
    
    def __init__(self):
        # Risk weights
        self.weights = {
            'tumor_type': 0.3,
            'growth_rate': 0.4,
            'current_size': 0.2,
            'location': 0.1
        }
    
    def calculate_risk_score(self, tumor_data):
        """
        Calculate risk score from 0-100
        
        Args:
            tumor_data: dict with keys:
                - tumor_type: 'glioma', 'meningioma', 'pituitary'
                - growth_rate: relative growth % per month
                - current_volume: cm³
                - location: 'superficial', 'deep', 'critical'
        
        Returns:
            risk_score: 0-100
            risk_category: 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'
            urgency: days until recommended intervention
        """
        
        # Tumor type risk
        type_scores = {
            'glioma': 80,  # Aggressive
            'meningioma': 30,  # Usually benign
            'pituitary': 40  # Moderate
        }
        type_score = type_scores.get(tumor_data['tumor_type'].lower(), 50)
        
        # Growth rate risk
        growth = tumor_data['growth_rate']
        if growth < 0:  # Shrinking
            growth_score = 0
        elif growth < 10:  # Slow growth
            growth_score = 20
        elif growth < 25:  # Moderate growth
            growth_score = 50
        elif growth < 50:  # Fast growth
            growth_score = 75
        else:  # Very fast growth
            growth_score = 100
        
        # Size risk
        volume = tumor_data['current_volume']
        if volume < 5:
            size_score = 20
        elif volume < 15:
            size_score = 40
        elif volume < 30:
            size_score = 60
        elif volume < 50:
            size_score = 80
        else:
            size_score = 100
        
        # Location risk
        location_scores = {
            'superficial': 30,
            'deep': 60,
            'critical': 90  # Near vital structures
        }
        location_score = location_scores.get(tumor_data.get('location', 'deep'), 60)
        
        # Weighted combination
        risk_score = (
            type_score * self.weights['tumor_type'] +
            growth_score * self.weights['growth_rate'] +
            size_score * self.weights['current_size'] +
            location_score * self.weights['location']
        )
        
        # Risk category
        if risk_score < 30:
            risk_category = 'LOW'
            urgency_days = 365  # Annual follow-up
        elif risk_score < 50:
            risk_category = 'MODERATE'
            urgency_days = 90  # Quarterly follow-up
        elif risk_score < 70:
            risk_category = 'HIGH'
            urgency_days = 30  # Monthly follow-up
        else:
            risk_category = 'CRITICAL'
            urgency_days = 7  # Immediate action
        
        return {
            'risk_score': risk_score,
            'risk_category': risk_category,
            'urgency_days': urgency_days,
            'component_scores': {
                'tumor_type': type_score,
                'growth_rate': growth_score,
                'size': size_score,
                'location': location_score
            }
        }
    
    def generate_recommendation(self, risk_assessment):
        """
        Generate clinical recommendation
        """
        score = risk_assessment['risk_score']
        category = risk_assessment['risk_category']
        urgency = risk_assessment['urgency_days']
        
        if category == 'CRITICAL':
            recommendation = f"""
🚨 URGENT ACTION REQUIRED
Risk Score: {score:.1f}/100 (CRITICAL)

Recommendations:
1. Immediate neurosurgical consultation (within {urgency} days)
2. Consider urgent intervention
3. Repeat imaging in 1-2 weeks
4. Discuss treatment options with patient/family
5. Evaluate for clinical trial eligibility

Justification:
- High-risk tumor characteristics
- Rapid growth rate
- Significant size/location concerns
            """
        
        elif category == 'HIGH':
            recommendation = f"""
⚠️ HIGH PRIORITY
Risk Score: {score:.1f}/100 (HIGH)

Recommendations:
1. Schedule surgical consultation within {urgency} days
2. Consider treatment options (surgery, radiation, etc.)
3. Repeat imaging in 1 month
4. Monitor for symptom progression
5. Optimize medical management

Justification:
- Concerning tumor features
- Active growth detected
- Requires close monitoring
            """
        
        elif category == 'MODERATE':
            recommendation = f"""
⏰ ACTIVE SURVEILLANCE
Risk Score: {score:.1f}/100 (MODERATE)

Recommendations:
1. Clinical follow-up in {urgency} days
2. Repeat imaging in 3 months
3. Monitor symptoms
4. Consider preventive measures
5. Patient education on warning signs

Justification:
- Stable disease with some concerning features
- Continued monitoring appropriate
- Intervention may be needed if progression
            """
        
        else:  # LOW
            recommendation = f"""
✅ ROUTINE MONITORING
Risk Score: {score:.1f}/100 (LOW)

Recommendations:
1. Annual clinical follow-up
2. Repeat imaging in 12 months
3. Symptom surveillance
4. Maintain current management
5. Reassure patient/family

Justification:
- Low-risk tumor characteristics
- Stable or slow growth
- Conservative management appropriate
            """
        
        return recommendation

# Test risk calculator
calculator = ClinicalRiskCalculator()

test_case = {
    'tumor_type': 'glioma',
    'growth_rate': 35,  # 35% growth per 6 months
    'current_volume': 28,  # cm³
    'location': 'deep'
}

risk = calculator.calculate_risk_score(test_case)
recommendation = calculator.generate_recommendation(risk)

print(risk)
print(recommendation)
```

---

# 8. EVALUATION METRICS

## 8.1 Progression Prediction Metrics

**Cell 13: Evaluation Functions**

```python
def evaluate_progression_model(predictions, ground_truth):
    """
    Evaluate progression prediction accuracy
    
    Args:
        predictions: dict of predicted volumes
        ground_truth: dict of actual volumes
    
    Returns:
        metrics dict
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Extract predicted and actual volumes
    pred_volumes = []
    true_volumes = []
    
    for patient_id in predictions.keys():
        if patient_id in ground_truth:
            pred_volumes.append(predictions[patient_id]['future_volumes'])
            true_volumes.append(ground_truth[patient_id]['actual_volumes'])
    
    pred_volumes = np.concatenate(pred_volumes)
    true_volumes = np.concatenate(true_volumes)
    
    # Calculate metrics
    mae = mean_absolute_error(true_volumes, pred_volumes)
    rmse = np.sqrt(mean_squared_error(true_volumes, pred_volumes))
    r2 = r2_score(true_volumes, pred_volumes)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((true_volumes - pred_volumes) / (true_volumes + 1e-8))) * 100
    
    # Direction accuracy (did we predict growth/shrinkage correctly?)
    direction_correct = np.sum(np.sign(pred_volumes) == np.sign(true_volumes))
    direction_accuracy = direction_correct / len(pred_volumes) * 100
    
    metrics = {
        'MAE (cm³)': mae,
        'RMSE (cm³)': rmse,
        'R² Score': r2,
        'MAPE (%)': mape,
        'Direction Accuracy (%)': direction_accuracy
    }
    
    return metrics

def evaluate_clinical_utility(predictions, ground_truth, threshold_months=6):
    """
    Evaluate clinical decision accuracy
    
    Specifically: Did we correctly identify patients needing intervention?
    """
    
    # True positives: Correctly identified high-risk patients
    # False positives: Incorrectly flagged low-risk as high-risk
    # False negatives: Missed high-risk patients
    # True negatives: Correctly identified low-risk patients
    
    tp = fp = tn = fn = 0
    
    for patient_id in predictions.keys():
        if patient_id not in ground_truth:
            continue
        
        pred_risk = predictions[patient_id]['risk_level']
        true_progression = ground_truth[patient_id]['actual_progression']
        
        # High risk prediction
        pred_high_risk = pred_risk in ['HIGH', 'CRITICAL']
        
        # True progression (>25% growth = progressive disease)
        true_high_risk = true_progression > 25
        
        if pred_high_risk and true_high_risk:
            tp += 1
        elif pred_high_risk and not true_high_risk:
            fp += 1
        elif not pred_high_risk and true_high_risk:
            fn += 1
        else:
            tn += 1
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    clinical_metrics = {
        'Sensitivity (%)': sensitivity * 100,
        'Specificity (%)': specificity * 100,
        'PPV (%)': ppv * 100,
        'NPV (%)': npv * 100,
        'Accuracy (%)': accuracy * 100,
        'True Positives': tp,
        'False Positives': fp,
        'True Negatives': tn,
        'False Negatives': fn
    }
    
    return clinical_metrics

print("✅ Evaluation functions defined")
```

---

# 9. VISUALIZATION DASHBOARD

## 9.1 Interactive Dashboard

**Using Streamlit for clinical interface:**

```python
# File: tumor_dashboard.py

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def create_dashboard():
    """
    Create interactive dashboard for tumor progression
    """
    
    st.set_page_config(page_title="Tumor Time Travel", layout="wide")
    
    st.title("🧠 Tumor Time Travel: Growth Prediction System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Patient Selection")
    patient_ids = volume_df['patient_id'].unique()
    selected_patient = st.sidebar.selectbox("Select Patient", patient_ids)
    
    # Get patient data
    patient_data = volume_df[volume_df['patient_id'] == selected_patient]
    
    # Make prediction
    prediction = predict_future_growth(patient_data, prediction_days=180)
    
    if prediction is None:
        st.error("Unable to generate prediction for this patient")
        return
    
    # Layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Current Volume",
            value=f"{prediction['current_volume']:.2f} cm³"
        )
    
    with col2:
        st.metric(
            label="Predicted 6-Month Volume",
            value=f"{prediction['predicted_6mo_volume']:.2f} cm³",
            delta=f"{prediction['relative_growth_6mo']:.1f}%"
        )
    
    with col3:
        risk_color = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🔴'}
        st.metric(
            label="Risk Level",
            value=f"{risk_color[prediction['risk_level']]} {prediction['risk_level']}"
        )
    
    # Growth curve plot
    st.subheader("Growth Trajectory")
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=prediction['historical_times'],
        y=prediction['historical_volumes'],
        mode='markers+lines',
        name='Observed',
        marker=dict(size=10, color='blue'),
        line=dict(color='blue', width=2)
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=prediction['future_times'],
        y=prediction['future_predictions'],
        mode='markers+lines',
        name='Predicted',
        marker=dict(size=10, color='red'),
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Uncertainty
    fig.add_trace(go.Scatter(
        x=np.concatenate([prediction['future_times'], prediction['future_times'][::-1]]),
        y=np.concatenate([
            prediction['future_predictions'] * 1.1,
            (prediction['future_predictions'] * 0.9)[::-1]
        ]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Uncertainty',
        showlegend=True
    ))
    
    fig.update_layout(
        title='Tumor Volume Over Time',
        xaxis_title='Days from Baseline',
        yaxis_title='Volume (cm³)',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Clinical recommendation
    st.subheader("Clinical Assessment")
    
    rec_col1, rec_col2 = st.columns([1, 2])
    
    with rec_col1:
        st.markdown(f"**RANO Status:** {prediction['rano_status']}")
        st.markdown(f"**Model Used:** {prediction['model_used']}")
        st.markdown(f"**Model R²:** {prediction['r2_score']:.4f}")
        
        if prediction['doubling_time_days']:
            st.markdown(f"**Doubling Time:** {prediction['doubling_time_days']:.1f} days")
    
    with rec_col2:
        st.info(f"**Recommendation:** {prediction['recommendation']}")
    
    # Detailed metrics
    with st.expander("📊 Detailed Metrics"):
        metric_df = pd.DataFrame({
            'Metric': [
                'Current Volume (cm³)',
                'Predicted 6-Month Volume (cm³)',
                'Absolute Growth (cm³)',
                'Relative Growth (%)',
                'Model MAE (cm³)',
                'Model RMSE (cm³)'
            ],
            'Value': [
                f"{prediction['current_volume']:.2f}",
                f"{prediction['predicted_6mo_volume']:.2f}",
                f"{prediction['absolute_growth_6mo']:.2f}",
                f"{prediction['relative_growth_6mo']:.1f}",
                f"{prediction['mae']:.2f}",
                f"{np.sqrt(prediction['mae']**2):.2f}"
            ]
        })
        st.table(metric_df)

if __name__ == "__main__":
    create_dashboard()
```

**To run:**
```bash
streamlit run tumor_dashboard.py
```

---

# 10. COMPLETE CODE IMPLEMENTATION

## 10.1 End-to-End Script

**Cell 14: Complete Pipeline**

```python
"""
COMPLETE TUMOR PROGRESSION PIPELINE
Run this cell to execute entire progression analysis
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'data_dir': '/kaggle/working/data/MU-Glioma-Post',
    'output_dir': '/kaggle/working/results/progression',
    'prediction_days': 180,
    'use_model': 'best',  # 'best', 'exponential', 'gompertz', 'lstm'
    'generate_visualizations': True,
    'generate_reports': True
}

print("="*80)
print("TUMOR TIME TRAVEL: COMPLETE PIPELINE")
print("="*80)

# Step 1: Extract volumes
print("\n[1/5] Extracting tumor volumes...")
volume_df = extract_all_volumes(CONFIG['data_dir'])
print(f"✅ Extracted volumes for {len(volume_df['patient_id'].unique())} patients")

# Step 2: Process all patients
print("\n[2/5] Generating growth predictions...")
summary_df = process_all_patients(
    volume_df, 
    output_dir=CONFIG['output_dir']
)
print(f"✅ Processed {len(summary_df)} patients")

# Step 3: Generate cohort statistics
print("\n[3/5] Calculating cohort statistics...")
cohort_stats = {
    'total_patients': len(summary_df),
    'high_risk_count': len(summary_df[summary_df['risk_level'] == 'HIGH']),
    'mean_growth': summary_df['relative_growth'].mean(),
    'median_growth': summary_df['relative_growth'].median(),
    'pd_count': len(summary_df[summary_df['rano_status'].str.contains('Progressive')]),
    'pr_count': len(summary_df[summary_df['rano_status'].str.contains('Response')]),
    'sd_count': len(summary_df[summary_df['rano_status'].str.contains('Stable')])
}

print("Cohort Summary:")
for key, value in cohort_stats.items():
    print(f"  {key}: {value}")

# Step 4: Generate reports
if CONFIG['generate_reports']:
    print("\n[4/5] Generating clinical reports...")
    
    # Risk distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Risk levels
    risk_counts = summary_df['risk_level'].value_counts()
    axes[0].bar(risk_counts.index, risk_counts.values, 
                color=['green', 'orange', 'red'])
    axes[0].set_title('Risk Distribution')
    axes[0].set_ylabel('Number of Patients')
    axes[0].grid(axis='y', alpha=0.3)
    
    # RANO status
    rano_counts = summary_df['rano_status'].value_counts()
    axes[1].bar(range(len(rano_counts)), rano_counts.values, 
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[1].set_xticks(range(len(rano_counts)))
    axes[1].set_xticklabels(rano_counts.index, rotation=45, ha='right')
    axes[1].set_title('RANO Status Distribution')
    axes[1].set_ylabel('Number of Patients')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/cohort_summary.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Reports generated")

# Step 5: Save final results
print("\n[5/5] Saving final results...")
results = {
    'config': CONFIG,
    'cohort_stats': cohort_stats,
    'patient_summary': summary_df.to_dict('records')
}

import json
with open(f"{CONFIG['output_dir']}/final_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ All results saved to {CONFIG['output_dir']}")

print("\n" + "="*80)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nNext steps:")
print(f"1. Review visualizations in {CONFIG['output_dir']}")
print(f"2. Check patient summaries in all_patients_summary.csv")
print(f"3. Integrate with classification system")
print(f"4. Prepare clinical deployment")
```

---

# APPENDIX: QUICK START GUIDE

## For Busy Researchers

**Option 1: Mathematical Models (Fast Track - 2 weeks)**

Execute in order:
1. Cell 1: Extract volumes
2. Cell 2: Define growth models
3. Cell 3: Fit models
4. Cell 4: Predict future
5. Cell 5: Visualize
6. Cell 6: Process all patients

**Option 2: Deep Learning (Advanced - 4 weeks)**

Execute in order:
1. Cells 1-6 (same as Option 1)
2. Cell 7: Define LSTM
3. Cell 8: Create dataset
4. Cell 9: Train LSTM
5. Cell 10: LSTM predictions

**Option 3: Complete System (6 weeks)**

Execute all cells + integration (Cell 11-14)

---

# NEXT STEPS AFTER IMPLEMENTATION

1. **Week 1-2**: Implement Path A (Mathematical Models)
   - Get baseline results quickly
   - Validate on small dataset first

2. **Week 3-4**: Implement Path B (LSTM) if time allows
   - Compare with mathematical models
   - Publish comparison results

3. **Week 5**: Integration
   - Combine with FL-QPSO classification
   - Create unified pipeline

4. **Week 6**: Clinical Deployment
   - Build dashboard
   - User testing
   - Documentation

---

**END OF TUMOR PROGRESSION GUIDE**

You now have everything you need to implement tumor time travel! This guide is completely compatible with your existing FL-QPSO classification system and can be implemented in parallel or sequentially.
