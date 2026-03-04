# TUMOR GROWTH & PROGRESSION: TEAM TASK DIVISION PLAN

Based on the overall project structure, **Person 1** has completed the 3D U-Net Segmentation and the FL-QPSO modules.
The remaining modules focus primarily on **Tumor Time Travel (Progression Prediction)** and **System Integration**. 

Here is the strategic plan to divide the tumor growth forecasting and system integration tasks equally between **Person 2** and **Person 3**.

---

## 📅 STRATEGIC WORKFLOW (4-WEEK PLAN)

### 📌 **PERSON 2: Data Engineering & Mathematical Models (The Baseline)**
**Focus:** Handling the temporal MRI datasets, extracting volumes, and building the baseline mathematical growth models.

**Week 1: Data Preparation & Volume Extraction**
- [ ] Understand the `TUMOR_PROGRESSION_COMPLETE_GUIDE.md` specifications.
- [ ] Download the longitudinal dataset (`MU-Glioma-Post` or `LUMIERE`) from TCIA.
- [ ] Set up the data preprocessing pipeline (Co-registration, resampling).
- [ ] Implement `extract_tumor_volume()` script using the provided segmentation outputs from Module 1.
- [ ] Generate the comprehensive `tumor_volumes.csv` linking patient IDs to timepoints.

**Week 2: Mathematical Path Modeling**
- [ ] Implement TumorGrowthModels class (Exponential, Gompertz, Logistic, Linear).
- [ ] Fit mathematical curves across to the extracted longitudinal datasets for each patient.
- [ ] Calculate standard growth metrics: Absolute Growth Rate (AGR), Relative Growth Rate (RGR), Doubling Time.
- [ ] Produce CSV summary of patient historical trajectories vs predicted future growth (Path A completion).

---

### 📌 **PERSON 3: Deep Learning, Risk Assessment & Overall Integration**
**Focus:** Developing the advanced LSTM predictive models, classifying clinical risk, and plugging all modules together.

**Week 3: Deep Learning (LSTM) & Clinical Risk Evaluator**
- [ ] Build the `TumorGrowthLSTM` model in PyTorch for time-series volume predictions.
- [ ] Sequence the longitudinal data into `TumorSequenceDataset` for deep learning batches.
- [ ] Train and validate the forecasting model against the mathematical curves.
- [ ] Develop the `ClinicalRiskCalculator` assigning RANO Status (Progressive Disease, Stable Disease, Complete Response) based on 6-month growth outlooks.
- [ ] Generate the Risk Alert summaries (CRITICAL / HIGH / MODERATE).

**Week 4: Final Pipeline Integration & Dashboarding**
- [ ] Review `INTEGRATION_GUIDE.md`. 
- [ ] Stitch the FL-QPSO Outputs (from Person 1) sequentially into the Tumor Progression inputs.
- [ ] Finalize `comprehensive_tumor_analysis` script spanning from initial scan → segment → trace volumes → risk report.
- [ ] Connect the progression visualizations (growth curves and bar charts from plotting scripts) into the deployment presentation formats.

---

## 🤝 COLLABORATION POINTS

- **End of Week 2:** Person 2 must hand over the standardized `tumor_volumes.csv` and preprocessing functions to Person 3, allowing Person 3 to train the LSTM models on proper data.
- **End of Week 3:** Person 3 integrates the RANO logic alongside Person 2's mathematical baseline module so that the final prediction function can pick the "Best Fitted Mode".
- **Week 4 Sync:** The entire team (Person 1, 2, and 3) verifies that input arrays produced by the completed `3D U-Net Segmentation` map directly without indexing errors into the temporal volume extractors.
