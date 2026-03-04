# FL-QPSO + TUMOR PROGRESSION: INTEGRATION GUIDE
## Combining Classification & Growth Prediction into One System

**Version:** 1.0  
**Platform:** Kaggle (2x Tesla T4 GPUs)

---

# OVERVIEW

This guide shows you **exactly** how to integrate the Tumor Progression Prediction system with your existing FL-QPSO Classification system to create a **complete brain tumor management platform**.

---

# SYSTEM ARCHITECTURE

```
┌──────────────────────────────────────────────────────────────┐
│                    PATIENT DATA INPUT                         │
│              (Multiple MRI scans over time)                   │
└───────────────────────────┬──────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼────────┐                   ┌──────────▼─────────┐
│   MODULE 1:    │                   │    MODULE 2:       │
│ CLASSIFICATION │                   │  SEGMENTATION      │
│ (FL-QPSO)      │                   │  (U-Net / any)     │
│                │                   │                    │
│ Output:        │                   │ Output:            │
│ - Tumor type   │                   │ - Tumor masks      │
│ - Confidence   │                   │ - Volumes          │
└───────┬────────┘                   └──────────┬─────────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            │
                   ┌────────▼────────┐
                   │   MODULE 3:     │
                   │  PROGRESSION    │
                   │  PREDICTION     │
                   │                 │
                   │ Output:         │
                   │ - Growth curve  │
                   │ - Risk level    │
                   │ - Recommendation│
                   └────────┬────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
    ┌───────▼────────┐           ┌─────────▼──────────┐
    │   MODULE 4:    │           │    MODULE 5:       │
    │ RISK FUSION    │           │  CLINICAL OUTPUT   │
    │                │           │                    │
    │ Combines:      │           │ - Report PDF       │
    │ - Type risk    │           │ - Visualizations   │
    │ - Growth risk  │           │ - Action plan      │
    │ - Size risk    │           │ - Follow-up date   │
    └───────┬────────┘           └────────────────────┘
            │
            └────► FINAL DECISION SUPPORT
```

---

# INTEGRATION STRATEGY

## Approach 1: Sequential Pipeline ⭐ RECOMMENDED
**Run classification first, then progression**

```python
patient_scans = load_patient_scans(patient_id)

# Step 1: Classify tumor type
tumor_type, confidence = classify_with_fedavg_or_qpso(patient_scans[0])

# Step 2: Segment all timepoints
volumes = []
for scan in patient_scans:
    mask = segment_tumor(scan)
    volumes.append(calculate_volume(mask))

# Step 3: Predict growth
growth_pred = predict_growth(volumes, days)

# Step 4: Combined decision
final_report = generate_clinical_report(tumor_type, confidence, growth_pred)
```

## Approach 2: Parallel Processing ⚡ FASTER
**Run classification and segmentation simultaneously**

```python
from multiprocessing import Pool

def process_scan(scan):
    classification = classify_tumor(scan)
    segmentation = segment_tumor(scan)
    return classification, segmentation

with Pool(processes=2) as pool:
    results = pool.map(process_scan, patient_scans)

# Then aggregate results
```

## Approach 3: Federated Integration 🔒 MOST SECURE
**Both classification AND progression trained federatively**

```python
# Server coordinates both tasks
for round in range(num_rounds):
    # Classification FL
    classification_weights = aggregate_classification(clients)
    
    # Progression FL (on clients with longitudinal data)
    progression_weights = aggregate_progression(clients_with_longitudinal)
    
    # Send both models back to clients
```

---

# STEP-BY-STEP INTEGRATION

## Step 1: Prepare Combined Dataset

**Cell 1: Dataset Organization**

```python
"""
Organize data for integrated system

Expected structure:
data/
├── patient_001/
│   ├── baseline/
│   │   ├── T1.nii.gz
│   │   ├── T1ce.nii.gz
│   │   ├── T2.nii.gz
│   │   ├── FLAIR.nii.gz
│   │   └── segmentation.nii.gz
│   ├── followup_1/
│   │   └── [same structure]
│   ├── followup_2/
│   │   └── [same structure]
│   └── metadata.json  ← NEW: contains tumor type, dates, etc.
"""

import json
import os

def create_integrated_metadata(data_dir):
    """
    Create metadata files linking classification and progression data
    """
    
    metadata_template = {
        'patient_id': None,
        'tumor_type': None,  # From classification
        'tumor_type_confidence': None,
        'timepoints': []  # For progression
    }
    
    for patient_dir in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient_dir)
        
        if not os.path.isdir(patient_path):
            continue
        
        metadata = metadata_template.copy()
        metadata['patient_id'] = patient_dir
        
        # Collect timepoints
        timepoints = []
        for tp_dir in sorted(os.listdir(patient_path)):
            tp_path = os.path.join(patient_path, tp_dir)
            if os.path.isdir(tp_path):
                timepoints.append({
                    'name': tp_dir,
                    'date': None,  # Parse from DICOM if available
                    'scans_present': os.listdir(tp_path)
                })
        
        metadata['timepoints'] = timepoints
        
        # Save
        with open(os.path.join(patient_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata created for all patients")

# Execute
create_integrated_metadata('/kaggle/working/data/integrated')
```

## Step 2: Combined Data Loader

**Cell 2: Unified Data Loader**

```python
class IntegratedBrainTumorDataset:
    """
    Dataset that provides both classification and progression data
    """
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.patients = []
        
        # Load all patient metadata
        for patient_dir in os.listdir(data_dir):
            patient_path = os.path.join(data_dir, patient_dir)
            metadata_path = os.path.join(patient_path, 'metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    metadata['path'] = patient_path
                    self.patients.append(metadata)
    
    def get_classification_data(self, patient_id, timepoint='baseline'):
        """
        Get data for classification (single scan)
        """
        patient = self._get_patient(patient_id)
        timepoint_path = os.path.join(patient['path'], timepoint)
        
        # Load all modalities
        T1 = nib.load(os.path.join(timepoint_path, 'T1.nii.gz')).get_fdata()
        T1ce = nib.load(os.path.join(timepoint_path, 'T1ce.nii.gz')).get_fdata()
        T2 = nib.load(os.path.join(timepoint_path, 'T2.nii.gz')).get_fdata()
        FLAIR = nib.load(os.path.join(timepoint_path, 'FLAIR.nii.gz')).get_fdata()
        
        # Preprocess
        scan = self._preprocess_for_classification(T1, T1ce, T2, FLAIR)
        
        return scan
    
    def get_progression_data(self, patient_id):
        """
        Get data for progression (all timepoints)
        """
        patient = self._get_patient(patient_id)
        
        volumes = []
        days = []
        
        for i, tp in enumerate(patient['timepoints']):
            seg_path = os.path.join(patient['path'], tp['name'], 'segmentation.nii.gz')
            
            if os.path.exists(seg_path):
                seg = nib.load(seg_path).get_fdata()
                volume = self._calculate_volume(seg)
                volumes.append(volume)
                days.append(i * 90)  # Adjust based on actual dates
        
        return {
            'days': days,
            'volumes': volumes
        }
    
    def _get_patient(self, patient_id):
        for patient in self.patients:
            if patient['patient_id'] == patient_id:
                return patient
        return None
    
    def _preprocess_for_classification(self, T1, T1ce, T2, FLAIR):
        """Preprocess for your FL-QPSO classifier"""
        # Use your existing preprocessing from FL-QPSO guide
        pass
    
    def _calculate_volume(self, segmentation):
        """Calculate tumor volume from segmentation"""
        return np.sum(segmentation > 0) * 1.0  # Adjust voxel spacing

# Create dataset
integrated_dataset = IntegratedBrainTumorDataset('/kaggle/working/data/integrated')
print(f"Loaded {len(integrated_dataset.patients)} patients")
```

## Step 3: Integrated Model Class

**Cell 3: Combined Model Wrapper**

```python
class IntegratedBrainTumorSystem:
    """
    Complete system combining classification and progression
    """
    
    def __init__(self, classification_model, segmentation_model, progression_model, device='cuda'):
        """
        Args:
            classification_model: Your trained FL-QPSO classifier
            segmentation_model: U-Net or similar
            progression_model: LSTM or mathematical model
        """
        self.classification_model = classification_model
        self.segmentation_model = segmentation_model
        self.progression_model = progression_model
        self.device = device
        
        # Risk calculator
        self.risk_calculator = ClinicalRiskCalculator()
    
    def analyze_patient(self, patient_id, dataset, timepoints='all'):
        """
        Complete analysis for one patient
        
        Returns:
            comprehensive_report: dict with all results
        """
        
        print(f"Analyzing patient: {patient_id}")
        
        # ===== PHASE 1: CLASSIFICATION =====
        print("[1/4] Classifying tumor type...")
        classification_scan = dataset.get_classification_data(patient_id, 'baseline')
        
        tumor_type, confidence = self._classify(classification_scan)
        
        print(f"  Tumor type: {tumor_type} (confidence: {confidence:.2%})")
        
        # ===== PHASE 2: SEGMENTATION & VOLUMES =====
        print("[2/4] Extracting tumor volumes...")
        progression_data = dataset.get_progression_data(patient_id)
        
        if len(progression_data['volumes']) < 2:
            print("  ⚠️ Insufficient timepoints for progression analysis")
            return self._generate_classification_only_report(tumor_type, confidence)
        
        print(f"  Found {len(progression_data['volumes'])} timepoints")
        
        # ===== PHASE 3: PROGRESSION PREDICTION =====
        print("[3/4] Predicting tumor growth...")
        
        growth_prediction = self._predict_growth(
            progression_data['days'],
            progression_data['volumes']
        )
        
        print(f"  Predicted 6-month growth: {growth_prediction['relative_growth_6mo']:.1f}%")
        print(f"  RANO status: {growth_prediction['rano_status']}")
        
        # ===== PHASE 4: INTEGRATED RISK ASSESSMENT =====
        print("[4/4] Computing integrated risk score...")
        
        risk_data = {
            'tumor_type': tumor_type,
            'growth_rate': growth_prediction['relative_growth_6mo'] / 6,  # per month
            'current_volume': progression_data['volumes'][-1],
            'location': 'deep'  # You can add location detection
        }
        
        risk_assessment = self.risk_calculator.calculate_risk_score(risk_data)
        
        print(f"  Risk score: {risk_assessment['risk_score']:.1f}/100 ({risk_assessment['risk_category']})")
        
        # ===== GENERATE COMPREHENSIVE REPORT =====
        comprehensive_report = {
            'patient_id': patient_id,
            'classification': {
                'tumor_type': tumor_type,
                'confidence': confidence
            },
            'progression': growth_prediction,
            'risk': risk_assessment,
            'recommendation': self.risk_calculator.generate_recommendation(risk_assessment),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return comprehensive_report
    
    def _classify(self, scan):
        """Run classification model"""
        self.classification_model.eval()
        
        with torch.no_grad():
            scan_tensor = torch.FloatTensor(scan).unsqueeze(0).to(self.device)
            output = self.classification_model(scan_tensor)
            
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            class_names = ['glioma', 'meningioma', 'pituitary']
            tumor_type = class_names[predicted_class]
        
        return tumor_type, confidence
    
    def _predict_growth(self, days, volumes):
        """Run progression model"""
        
        patient_data = pd.DataFrame({
            'days': days,
            'volume_cm3': volumes
        })
        
        # Use your progression prediction function from earlier
        prediction = predict_future_growth(patient_data, prediction_days=180)
        
        return prediction
    
    def _generate_classification_only_report(self, tumor_type, confidence):
        """Generate report when progression data unavailable"""
        return {
            'classification': {
                'tumor_type': tumor_type,
                'confidence': confidence
            },
            'progression': None,
            'note': 'Insufficient timepoints for progression analysis'
        }
    
    def visualize_comprehensive_report(self, report, save_path=None):
        """
        Create comprehensive visualization combining all components
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Classification confidence
        ax1 = fig.add_subplot(gs[0, 0])
        classes = ['Glioma', 'Meningioma', 'Pituitary']
        confidences = [0, 0, 0]
        idx = classes.index(report['classification']['tumor_type'].capitalize())
        confidences[idx] = report['classification']['confidence']
        
        ax1.bar(classes, confidences, color=['red', 'blue', 'green'])
        ax1.set_ylim([0, 1])
        ax1.set_ylabel('Confidence')
        ax1.set_title('Tumor Type Classification', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Growth curve (main plot)
        ax2 = fig.add_subplot(gs[0:2, 1:])
        
        if report['progression'] is not None:
            prog = report['progression']
            
            # Historical
            ax2.scatter(prog['historical_times'], prog['historical_volumes'],
                       c='blue', s=100, label='Observed', zorder=3)
            ax2.plot(prog['historical_times'], prog['historical_predictions'],
                    'b--', linewidth=2, alpha=0.7)
            
            # Predicted
            ax2.plot(prog['future_times'], prog['future_predictions'],
                    'r-', linewidth=2, label='Predicted')
            ax2.fill_between(prog['future_times'],
                            prog['future_predictions'] * 0.9,
                            prog['future_predictions'] * 1.1,
                            color='red', alpha=0.2)
            
            ax2.axvline(x=prog['historical_times'][-1], color='black',
                       linestyle='--', alpha=0.5, label='Current')
            
            ax2.set_xlabel('Days from Baseline', fontsize=12)
            ax2.set_ylabel('Tumor Volume (cm³)', fontsize=12)
            ax2.set_title('Tumor Growth Trajectory', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Risk gauge
        ax3 = fig.add_subplot(gs[1, 0])
        risk_score = report['risk']['risk_score']
        risk_category = report['risk']['risk_category']
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones(100)
        
        ax3 = plt.subplot(gs[1, 0], projection='polar')
        ax3.plot(theta, r, color='gray', linewidth=20, alpha=0.3)
        
        # Color segments
        colors = ['green', 'yellow', 'orange', 'red']
        segments = [0, 30, 50, 70, 100]
        
        for i in range(len(segments)-1):
            theta_seg = np.linspace(segments[i]/100 * np.pi, segments[i+1]/100 * np.pi, 20)
            r_seg = np.ones(20)
            ax3.plot(theta_seg, r_seg, color=colors[i], linewidth=20)
        
        # Risk needle
        risk_angle = (risk_score / 100) * np.pi
        ax3.plot([risk_angle, risk_angle], [0, 1], color='black', linewidth=3)
        ax3.scatter([risk_angle], [1], color='black', s=200, zorder=5)
        
        ax3.set_ylim([0, 1.2])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.text(np.pi/2, 1.4, f'{risk_score:.0f}', ha='center', fontsize=24, fontweight='bold')
        ax3.text(np.pi/2, 1.6, risk_category, ha='center', fontsize=14, fontweight='bold')
        ax3.set_title('Risk Score', fontsize=12, fontweight='bold', pad=20)
        
        # Metrics table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        if report['progression'] is not None:
            table_data = [
                ['Classification', f"{report['classification']['tumor_type'].capitalize()} ({report['classification']['confidence']:.1%})"],
                ['Current Volume', f"{prog['current_volume']:.2f} cm³"],
                ['Predicted 6mo', f"{prog['predicted_6mo_volume']:.2f} cm³"],
                ['Growth Rate', f"{prog['relative_growth_6mo']:.1f}%"],
                ['RANO Status', prog['rano_status']],
                ['Risk Category', f"{risk_category} ({risk_score:.0f}/100)"],
                ['Urgency', f"Follow-up in {report['risk']['urgency_days']} days"],
                ['Recommendation', report['risk']['component_scores']]
            ]
            
            table = ax4.table(cellText=table_data, colLabels=['Metric', 'Value'],
                            cellLoc='left', loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Color code risk row
            if risk_category == 'CRITICAL':
                table[(6, 0)].set_facecolor('#ffcccc')
                table[(6, 1)].set_facecolor('#ffcccc')
            elif risk_category == 'HIGH':
                table[(6, 0)].set_facecolor('#ffe6cc')
                table[(6, 1)].set_facecolor('#ffe6cc')
        
        plt.suptitle(f'Comprehensive Brain Tumor Analysis - Patient {report["patient_id"]}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Initialize system
integrated_system = IntegratedBrainTumorSystem(
    classification_model=fedavg_model,  # or qpso_model from your FL training
    segmentation_model=None,  # Add if you have one
    progression_model=None,  # Mathematical or LSTM
    device=device
)

print("✅ Integrated system initialized")
```

## Step 4: Batch Processing

**Cell 4: Process All Patients**

```python
def process_all_patients_integrated(integrated_system, dataset, output_dir):
    """
    Process all patients through integrated system
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_reports = []
    
    for patient in tqdm(dataset.patients):
        patient_id = patient['patient_id']
        
        try:
            # Analyze patient
            report = integrated_system.analyze_patient(patient_id, dataset)
            
            # Visualize
            viz_path = os.path.join(output_dir, f'{patient_id}_comprehensive.png')
            integrated_system.visualize_comprehensive_report(report, save_path=viz_path)
            
            # Store
            all_reports.append(report)
            
            # Save individual report
            with open(os.path.join(output_dir, f'{patient_id}_report.json'), 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            continue
    
    # Save summary
    summary_df = pd.DataFrame([
        {
            'patient_id': r['patient_id'],
            'tumor_type': r['classification']['tumor_type'],
            'confidence': r['classification']['confidence'],
            'risk_score': r['risk']['risk_score'] if 'risk' in r else None,
            'risk_category': r['risk']['risk_category'] if 'risk' in r else None,
            'urgency_days': r['risk']['urgency_days'] if 'risk' in r else None
        }
        for r in all_reports
    ])
    
    summary_df.to_csv(os.path.join(output_dir, 'integrated_summary.csv'), index=False)
    
    print(f"\n✅ Processed {len(all_reports)} patients")
    print(f"Results saved to {output_dir}")
    
    return all_reports

# Execute
all_reports = process_all_patients_integrated(
    integrated_system,
    integrated_dataset,
    output_dir='/kaggle/working/results/integrated'
)
```

---

# FEDERATED LEARNING FOR BOTH TASKS

## Combined FL Training

**Cell 5: Federated Training for Both Models**

```python
"""
Train both classification and progression federatively
Each hospital contributes to both models
"""

class DualTaskFederatedClient:
    """
    Client that trains both classification and progression models
    """
    
    def __init__(self, client_id, classification_data, progression_data, device='cuda'):
        self.client_id = client_id
        self.classification_data = classification_data
        self.progression_data = progression_data
        self.device = device
        
        self.classification_model = None
        self.progression_model = None
    
    def train_both_tasks(self, epochs=5):
        """
        Train both models locally
        """
        
        # Train classification
        class_weights, class_metrics = self.train_classification(epochs)
        
        # Train progression (only if hospital has longitudinal data)
        if len(self.progression_data) >= 2:
            prog_weights, prog_metrics = self.train_progression(epochs)
        else:
            prog_weights, prog_metrics = None, None
        
        return {
            'classification': (class_weights, class_metrics),
            'progression': (prog_weights, prog_metrics)
        }
    
    def train_classification(self, epochs):
        """Train classification model"""
        # Use your existing FL client training code
        pass
    
    def train_progression(self, epochs):
        """Train progression model"""
        # Train LSTM on time series
        pass

class DualTaskFederatedServer:
    """
    Server that aggregates both classification and progression models
    """
    
    def __init__(self, classification_model, progression_model, clients):
        self.classification_model = classification_model
        self.progression_model = progression_model
        self.clients = clients
    
    def aggregate_round(self):
        """
        One round of federated training for both tasks
        """
        
        # Collect updates from all clients
        class_weights = []
        prog_weights = []
        
        for client in self.clients:
            results = client.train_both_tasks(epochs=5)
            
            # Classification weights (all clients)
            if results['classification'][0] is not None:
                class_weights.append(results['classification'][0])
            
            # Progression weights (only clients with longitudinal data)
            if results['progression'][0] is not None:
                prog_weights.append(results['progression'][0])
        
        # Aggregate classification (FedAvg or QPSO)
        aggregated_class = self.aggregate_classification(class_weights)
        self.classification_model.load_state_dict(aggregated_class)
        
        # Aggregate progression (only if we have updates)
        if len(prog_weights) > 0:
            aggregated_prog = self.aggregate_progression(prog_weights)
            self.progression_model.load_state_dict(aggregated_prog)
        
        return {
            'classification_performance': self.evaluate_classification(),
            'progression_performance': self.evaluate_progression()
        }
    
    def aggregate_classification(self, weights):
        """Use your FedAvg or QPSO aggregation"""
        pass
    
    def aggregate_progression(self, weights):
        """Aggregate progression model weights"""
        # Similar to classification aggregation
        pass

# This enables privacy-preserving training for BOTH tasks!
print("✅ Dual-task federated learning defined")
```

---

# CLINICAL DEPLOYMENT

## Web Interface

**Cell 6: Streamlit Dashboard**

```python
"""
Complete clinical interface combining both systems

File: integrated_dashboard.py
Run with: streamlit run integrated_dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go

def main():
    st.set_page_config(page_title="Brain Tumor Analysis System", layout="wide")
    
    st.title("🧠 Integrated Brain Tumor Management System")
    st.markdown("Classification + Growth Prediction + Risk Assessment")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Patient Selection")
    patient_ids = integrated_dataset.patients
    selected_patient_id = st.sidebar.selectbox(
        "Select Patient",
        [p['patient_id'] for p in patient_ids]
    )
    
    # Analyze button
    if st.sidebar.button("🔬 Analyze Patient", type="primary"):
        with st.spinner("Running comprehensive analysis..."):
            report = integrated_system.analyze_patient(
                selected_patient_id,
                integrated_dataset
            )
            
            st.session_state['current_report'] = report
    
    # Display results
    if 'current_report' in st.session_state:
        report = st.session_state['current_report']
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Tumor Type",
                value=report['classification']['tumor_type'].capitalize(),
                delta=f"{report['classification']['confidence']:.1%} confidence"
            )
        
        with col2:
            if report['progression']:
                st.metric(
                    label="Current Volume",
                    value=f"{report['progression']['current_volume']:.1f} cm³"
                )
        
        with col3:
            if report['progression']:
                st.metric(
                    label="6-Month Prediction",
                    value=f"{report['progression']['predicted_6mo_volume']:.1f} cm³",
                    delta=f"{report['progression']['relative_growth_6mo']:.1f}%"
                )
        
        with col4:
            risk_emoji = {'LOW': '🟢', 'MODERATE': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'}
            st.metric(
                label="Risk Level",
                value=f"{risk_emoji.get(report['risk']['risk_category'], '⚪')} {report['risk']['risk_category']}",
                delta=f"{report['risk']['risk_score']:.0f}/100"
            )
        
        # Tabs for detailed info
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "📈 Growth Analysis",
            "⚠️ Risk Assessment",
            "📋 Clinical Report"
        ])
        
        with tab1:
            st.subheader("Classification Results")
            # Add classification visualization
            
            st.subheader("Tumor Volumes Over Time")
            if report['progression']:
                fig = go.Figure()
                prog = report['progression']
                
                fig.add_trace(go.Scatter(
                    x=prog['historical_times'],
                    y=prog['historical_volumes'],
                    mode='markers+lines',
                    name='Observed'
                ))
                
                fig.add_trace(go.Scatter(
                    x=prog['future_times'],
                    y=prog['future_predictions'],
                    mode='markers+lines',
                    name='Predicted',
                    line=dict(dash='dash')
                ))
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Growth Dynamics")
            # Detailed growth analysis
        
        with tab3:
            st.subheader("Risk Breakdown")
            
            # Risk components
            components = report['risk']['component_scores']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(components.keys()),
                    y=list(components.values()),
                    marker_color=['red', 'orange', 'yellow', 'blue']
                )
            ])
            fig.update_layout(title="Risk Component Scores")
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            st.info(report['recommendation'])
        
        with tab4:
            st.subheader("Clinical Report")
            
            # Generate downloadable PDF
            if st.button("📄 Generate PDF Report"):
                pdf_path = generate_pdf_report(report)
                with open(pdf_path, 'rb') as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name=f"report_{selected_patient_id}.pdf",
                        mime="application/pdf"
                    )

if __name__ == "__main__":
    main()
```

---

# TESTING & VALIDATION

## Integration Tests

**Cell 7: Test Suite**

```python
def test_integrated_system():
    """
    Comprehensive tests for integrated system
    """
    
    print("Running integration tests...")
    
    # Test 1: Classification only (single scan)
    print("\n[Test 1] Classification only...")
    single_scan = integrated_dataset.get_classification_data('patient_001', 'baseline')
    tumor_type, confidence = integrated_system._classify(single_scan)
    assert tumor_type in ['glioma', 'meningioma', 'pituitary']
    assert 0 <= confidence <= 1
    print("  ✅ Classification working")
    
    # Test 2: Progression only (volume time series)
    print("\n[Test 2] Progression only...")
    progression_data = integrated_dataset.get_progression_data('patient_001')
    if len(progression_data['volumes']) >= 2:
        prediction = integrated_system._predict_growth(
            progression_data['days'],
            progression_data['volumes']
        )
        assert 'future_predictions' in prediction
        print("  ✅ Progression working")
    
    # Test 3: Full integration
    print("\n[Test 3] Full integrated analysis...")
    report = integrated_system.analyze_patient('patient_001', integrated_dataset)
    assert 'classification' in report
    assert 'risk' in report
    print("  ✅ Integration working")
    
    # Test 4: Batch processing
    print("\n[Test 4] Batch processing...")
    reports = process_all_patients_integrated(
        integrated_system,
        integrated_dataset,
        output_dir='/kaggle/working/results/test'
    )
    assert len(reports) > 0
    print("  ✅ Batch processing working")
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED ✅")
    print("="*50)

# Run tests
test_integrated_system()
```

---

# PERFORMANCE OPTIMIZATION

## GPU Optimization

```python
"""
Optimize for dual GPU usage
"""

# Split models across GPUs
classification_model = classification_model.to('cuda:0')
progression_model = progression_model.to('cuda:1')

# Parallel processing
def parallel_analysis(patients):
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        
        for patient in patients:
            future = executor.submit(
                integrated_system.analyze_patient,
                patient['patient_id'],
                integrated_dataset
            )
            futures.append(future)
        
        results = [f.result() for f in futures]
    
    return results
```

---

# FINAL CHECKLIST

Before deploying integrated system:

- [ ] Both models trained and saved
- [ ] Dataset organized with metadata
- [ ] Integration tested on sample patients
- [ ] Batch processing working
- [ ] Visualizations generate correctly
- [ ] PDF reports functional
- [ ] Dashboard deployed
- [ ] Performance acceptable (<30s per patient)
- [ ] Error handling robust
- [ ] Documentation complete

---

# TIMELINE

**Week 1**: Integration setup & data organization  
**Week 2**: Test classification → segmentation → progression pipeline  
**Week 3**: Implement risk fusion & reporting  
**Week 4**: Build dashboard & user testing  
**Week 5**: Optimize & deploy

---

**END OF INTEGRATION GUIDE**

You now have everything to combine your FL-QPSO classification with tumor progression prediction into one powerful clinical system! 🚀
