# Detecting-Human-Stress-with-the-WESAD-Dataset-From-Signal-to-Prediction
# WESAD-Chest Signals Analysis

This project analyzes the WESAD dataset’s chest-worn RespiBAN device data to
detect and classify four affective states: Neutral, Stress, amusement and Meditation

## About this Dataset

**WESAD** (Wearable Stress and Affect Detection) is a publicly-available multimodal dataset designed for wearable stress and emotion research. It was recorded in a lab study with 15 participants using both a wrist-worn and a chest-worn device.

**Sensor Modalities**  
- Blood Volume Pulse (BVP)  
- Electrocardiogram (ECG)  
- Electrodermal Activity (EDA)  
- Electromyogram (EMG)  
- Respiration (RESP)  
- Body Temperature (TEMP)  
- Three-axis Acceleration (ACC)  

**Affective States**  
1. **Baseline (Neutral)** – 20 min reading magazines  
2. **Stress** – Trier Social Stress Test (public speaking + mental math)  
3. **Amusement** – Watching humorous video clips  

**Key Facts**  
- Chest device sampling rate: 700 Hz (ECG, EDA, EMG, RESP, TEMP, ACC)  
- Wrist device sampling rates vary by channel  
- Self-report questionnaires accompany each session  
- Benchmark performance:  
  - 3-class (neutral vs. stress vs. amusement): up to 80% accuracy  
  - 2-class (stress vs. non-stress): up to 93% accuracy  

### Citation  
If you use WESAD in your work, please cite:  

> Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & van Laerhoven, K. (2018).  
> Introducing WESAD, a multimodal dataset for wearable stress and affect detection.  
> *Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI ’18)*, 400–408.  
> https://doi.org/10.1145/3242969.3242985

### Disclaimer  
You may use this data for **scientific, non-commercial purposes** only, provided that you give appropriate credit to the original authors. All rights reserved by the original creators.
- **Device**: RespiBAN (chest)
- **Signals used**:
  - ECG (heart activity, 700 Hz)
  - EDA (skin conductance, 700 Hz)
  - RESP (respiration, 700 Hz)
- **Conditions**:  
  1. Baseline (reading magazines)  
  2. Stress (public speaking + mental math)  
  3. Amusement (funny videos)
  4. Meditation (controlled breathing exercise)

 Methodology

1. **Preprocessing**
   - Drop all rows with `temp == 0`  
   - Biosppy filters on ECG, EDA, Resp  

2. **Window Segmentation**  
   I fixed each window to **60 seconds** (42 000 samples at 700 Hz), and generated three separate segmentations by shifting the window start by:
   - **10 s** (7 000 samples)  
   - **20 s** (14 000 samples)  
   - **30 s** (21 000 samples)

   Each segmentation produces its own feature table.

3. **Feature Extraction**  
   Per window, compute:
   - Time-domain stats: mean, std, median, min, max, skew, kurtosis, Q1/Q3  
   - PSD statistical analysis 
   - HR/HRV interpolated features via R-peak detection

4. **Modeling**  
   - Classifier: **Logistic Regression**  
   - Train/test split: 80/20 stratified by label 
## Results

I evaluated Logistic Regression using 60 s windows with three different step sizes. Here are the binary (stress vs. non-stress) and multi-class (baseline vs. stress vs. amusement) accuracies:

| Window Size | Step Size | Binary Accuracy | Multi-class Accuracy |
| :---------: | :-------: | :-------------: | :------------------: |
| 60 sec      | 30 sec    |      97%        |        80%           |
| 60 sec      | 20 sec    |      95%        |        81%           |
| 60 sec      | 10 sec    |      95%        |        81%           |
## Useful Resources

- GitHub: Stress & Affect Detection example  
  https://github.com/jaganjag/stress_affect_detection  
- GitHub: Springboard WESAD project  
  https://github.com/arsen-movsesyan/springboard_WESAD  
- Guide to Electrodermal Activity (University of Birmingham PDF)  
  https://www.birmingham.ac.uk/Documents/college-les/psych/saal/guide-electrodermal-activity.pdf  
- Choi et al. “Development and evaluation of an ambulatory stress monitor…”  
  http://research.cs.tamu.edu/prism/publications/choi2011ambulatoryStressMonitor.pdf  

---

## References

- Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018).  
   Introducing WESAD, a multimodal dataset for wearable stress and affect detection.  
   *ICMI 2018*, 400–408. https://doi.org/10.1145/3242969.3242985
-  Healey, J. A., & Picard, R. W. (2005). Detecting stress during real-world driving tasks using physiological sensors. *IEEE Transactions on Intelligent Transportation Systems*, 6(2), 156–166.
- “From lab to real-life: A three-stage validation of wearable technology for stress monitoring.” *MethodsX* (2025). https://doi.org/10.1016/j.mex.2025.103205

 
