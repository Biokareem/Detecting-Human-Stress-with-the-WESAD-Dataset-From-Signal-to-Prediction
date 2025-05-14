# %% [markdown]
# # Importing Libraries

# %%
pip install -U scikit-learn imbalanced-learn

# %%
!pip install peakutils


# %%
!pip install biosppy


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
import os
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as pl
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# # Loading Dataset

# %%
DATA_PATH = "../input/wesad-full-dataset/WESAD/" 
OUTPUT_PATH = "/kaggle/working/"  # Store processed files in Kaggle working directory

# Define column names
chest_columns = ['sid', 'acc1', 'acc2', 'acc3', 'ecg', 'emg', 'eda', 'temp', 'resp', 'label']

# List of subject IDs
ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]


# %% [markdown]
# ### Loading one subject

# %%
def pkl_to_chest(filename, subject_id):
    with open(filename, "rb") as f:
        unpickled_df = pickle.load(f, encoding="latin1")  # Use latin1 encoding
    
    chest_acc = unpickled_df["signal"]["chest"]["ACC"]
    chest_ecg = unpickled_df["signal"]["chest"]["ECG"]
    chest_emg = unpickled_df["signal"]["chest"]["EMG"]
    chest_eda = unpickled_df["signal"]["chest"]["EDA"]
    chest_temp = unpickled_df["signal"]["chest"]["Temp"]
    chest_resp = unpickled_df["signal"]["chest"]["Resp"]
    
    lbl = unpickled_df["label"].reshape(unpickled_df["label"].shape[0], 1)
    sid = np.full((lbl.shape[0], 1), subject_id)

    chest_all = np.concatenate((sid, chest_acc, chest_ecg, chest_emg, chest_eda, chest_temp, chest_resp, lbl), axis=1)
    return chest_all

# %% [markdown]
# ### Merging Chest Data

# %%
def merge_chest():
    merged_data = np.empty((0, len(chest_columns)))  # Initialize empty array

    for sid in ids:
        file = os.path.join(DATA_PATH, f'S{sid}', f'S{sid}.pkl')
        print(f"Processing file: {file}")

        if os.path.exists(file):
            subject_data = pkl_to_chest(file, sid)
            merged_data = np.concatenate((merged_data, subject_data), axis=0)
            print(f"Merged data shape so far: {merged_data.shape}")
        else:
            print(f"File {file} not found!")
    merged_df = pd.DataFrame(merged_data, columns=chest_columns)
    merged_df.to_pickle(os.path.join(OUTPUT_PATH, "merged_chest.pkl"))  # Save in Kaggle output directory


# %% [markdown]
# ### Filter the merged chest data

# %%
def filter_chest_data():
    df = pd.read_pickle(os.path.join(OUTPUT_PATH, "merged_chest.pkl"))
    
    df_fltr = df[df["label"].isin([1, 2, 3, 4])]  # Keep only relevant labels
    df_fltr = df_fltr[df_fltr["temp"] > 0]  # Remove invalid temperature readings
    df_fltr["binary_label"] = df_fltr["label"].apply(lambda x: 1 if x == 2 else 0)
    print(f"Merged data shape so far: {df_fltr.shape}")
    df_fltr.to_pickle(os.path.join(OUTPUT_PATH, "merged_chest_fltr.pkl"))  # Save filtered data

# %% [markdown]
# ### **Run the preprocessing pipeline**

# %%
def preprocess():
    merge_chest()
    filter_chest_data()

# %%
preprocess()


# %% [markdown]
# # Load filtered dataset

# %%
filtered_data = pd.read_pickle(os.path.join(OUTPUT_PATH, "merged_chest_fltr.pkl"))
print(f"Filtered Data Shape: {filtered_data.shape}")
filtered_data.head()

# %%
filtered_data.shape

# %%
filtered_data.info()

# %%
filtered_data.describe().T

# %%
print(filtered_data['sid'].unique())


# %%
def information(data):
    d_type=data.dtypes
    n_o_U=data.nunique()
    nulls=data.isnull().sum()

    print(pd.DataFrame({"d_types":d_type,"n_uniques":n_o_U,"n_nuls":nulls},index=data.columns))
    print(f"data have  {data.duplicated().sum()} numbers of duplications ")
    print(f"this data have {data.shape[0]} records and {data.shape[1] }  features")

# %%
information(filtered_data)

# %%
filtered_data["temp"].min()

# %%
filtered_data.groupby(['sid', 'label']).head()


# %%
filtered_data['binary_label'] = filtered_data['binary_label'].astype(float)

# %% [markdown]
# # Data Visualization

# %%
filtered_data["binary_label"].value_counts().plot(kind="bar")

# %%
filtered_data["label"].value_counts().plot(kind="bar")

# %% [markdown]
# 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement,
# 4 = meditation

# %% [markdown]
# 0: not stressed , 1: stressed

# %% [markdown]
# > The dataset is imbalanced, meaning a classifier might be biased towards predicting Baseline more often.

# %%
plt.figure(figsize=(10,8))
sns.heatmap(filtered_data.corr(),cmap='Blues',annot=True) 

# %% [markdown]
# # Preprocessing

# %% [markdown]
# ECG preprocessing

# %%
sampling_rate = 700  # Hz
duration_sec = 10
samples_to_plot = sampling_rate * duration_sec
ecg_signal=filtered_data["ecg"]
# --- Time axis ---
time_axis = np.linspace(0, duration_sec, samples_to_plot)

# --- Plot ECG ---
plt.figure(figsize=(15, 4))
plt.plot(time_axis, ecg_signal[:samples_to_plot], color='navy')
plt.title('ECG Signal (First 10 Seconds)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (ECG - raw units)')
plt.grid(True)
plt.show()

# %%
from biosppy.signals import ecg
ecg_raw= filtered_data["ecg"]
# Assume your ECG signal is in `ecg_raw` and sampling rate is 700 Hz
ecg_processed = ecg.ecg(signal=ecg_raw, sampling_rate=700, show=False)

# The filtered signal (bandpass filter applied)
ecg_filtered = ecg_processed['filtered']

# %%
sampling_rate = 700  # Hz
duration_sec = 10
samples_to_plot = sampling_rate * duration_sec
ecg_signal=ecg_filtered
# --- Time axis ---
time_axis = np.linspace(0, duration_sec, samples_to_plot)
# --- Plot ECG ---
plt.figure(figsize=(15, 4))
plt.plot(time_axis, ecg_signal[:samples_to_plot], color='navy')
plt.title('ECG Signal PREPROCESSED BIOspy (First 10 Seconds)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (ECG - raw units)')
plt.grid(True)
plt.show()

# %% [markdown]
# EDA

# %%
sampling_rate = 700  # Hz
duration_sec = 10
samples_to_plot = sampling_rate * duration_sec
eda_raw=filtered_data['eda']
eda_signal=eda_raw
# --- Time axis ---
time_axis = np.linspace(0, duration_sec, samples_to_plot)

# --- Plot ECG ---
plt.figure(figsize=(15, 4))
plt.plot(time_axis, eda_signal[:samples_to_plot], color='navy')
plt.title('eda first 10 sec')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (Eda - raw units)')
plt.grid(True)
plt.show()

# %%
from biosppy.signals import eda
eda_result = eda.eda(signal=eda_raw, sampling_rate=700, show=False)
processed_eda = eda_result['filtered']


# %%
sampling_rate = 700  # Hz
duration_sec = 10
samples_to_plot = sampling_rate * duration_sec
eda_raw=processed_eda
eda_signal=eda_raw
# --- Time axis ---
time_axis = np.linspace(0, duration_sec, samples_to_plot)

# --- Plot ECG ---
plt.figure(figsize=(15, 4))
plt.plot(time_axis, eda_signal[:samples_to_plot], color='navy')
plt.title('eda first 10 sec')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (Eda - raw units)')
plt.grid(True)
plt.show()

# %%
from biosppy.signals import resp
raw_resp=filtered_data['resp']
resp_result = resp.resp(signal=raw_resp, sampling_rate=700, show=False)
processed_resp = resp_result['filtered']

# %%
sampling_rate = 700  # Hz
duration_sec = 10
samples_to_plot = sampling_rate * duration_sec
resp_signal=raw_resp
# --- Time axis ---
time_axis = np.linspace(0, duration_sec, samples_to_plot)

# --- Plot ECG ---
plt.figure(figsize=(15, 4))
plt.plot(time_axis, resp_signal[:samples_to_plot], color='navy')
plt.title('resp first 10 sec')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (resp - raw units)')
plt.grid(True)
plt.show()

# %%
sampling_rate = 700  # Hz
duration_sec = 10
samples_to_plot = sampling_rate * duration_sec
resp_signal=processed_resp
# --- Time axis ---
time_axis = np.linspace(0, duration_sec, samples_to_plot)

# --- Plot ECG ---
plt.figure(figsize=(15, 4))
plt.plot(time_axis, resp_signal[:samples_to_plot], color='navy')
plt.title('processed_resp first 10 sec')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (processed_resp - raw units)')
plt.grid(True)
plt.show()

# %%
filtered_data['resp_filtered'] = processed_resp
filtered_data['ecg_filtered'] = ecg_filtered
filtered_data['eda_filtered'] = processed_eda

# %%
filtered_data.head()

# %%
import matplotlib.pyplot as plt

# Parameters
fs = 700              # Sampling frequency (Hz)
duration = 10          # Duration to plot (seconds)
samples = fs * duration
ecg_column = 'ecg_filtered'    # Update if your ECG column has a different name
label_column = 'binary_label'

# Filter ECG signals for Normal (1) and Stress (2)
normal_ecg = filtered_data[filtered_data[label_column] == 0][ecg_column].values[:samples]
stress_ecg = filtered_data[filtered_data[label_column] == 1][ecg_column].values[:samples]

# Plot the ECG signals
plt.figure(figsize=(14, 5))
plt.plot(normal_ecg, label='Normal (Baseline)', color='green')
plt.plot(stress_ecg, label='Stress', color='red', alpha=0.7)
plt.title("ECG Signal Comparison: Normal vs Stress")
plt.xlabel("Sample Index")
plt.ylabel("ECG Signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# **The stress condition exhibits higher amplitude peaks and more variability in the signal, indicating increased heart rate and irregularity, which are common physiological responses to stress. The normal baseline shows more consistent and lower amplitude peaks, suggesting a steadier heart rhythm.**

# %%
import matplotlib.pyplot as plt

# Parameters
fs = 700              # Sampling frequency (Hz)
duration = 10       # Duration to plot (seconds)
samples = fs * duration
resp_column = 'resp_filtered'    # Update if your ECG column has a different name
label_column = 'binary_label'

# Filter ECG signals for Normal (1) and Stress (2)
normal_resp = filtered_data[filtered_data[label_column] == 0][resp_column].values[:samples]
stress_resp = filtered_data[filtered_data[label_column] == 1][resp_column].values[:samples]

# Plot the ECG signals
plt.figure(figsize=(14, 5))
plt.plot(normal_resp, label='Normal (Baseline)', color='green')
plt.plot(stress_resp, label='Stress', color='red', alpha=0.7)
plt.title("RESP Signal Comparison: Normal vs Stress")
plt.xlabel("Sample Index")
plt.ylabel("RESP Signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# **The stress condition shows a more irregular and higher amplitude breathing pattern, with sharper peaks and deeper troughs, indicating faster and more uneven respiration typical of stress. The normal baseline has a smoother, more consistent sinusoidal pattern, suggesting steady and regular breathing.**

# %%
import matplotlib.pyplot as plt

# Parameters
fs = 700              # Sampling frequency (Hz)
duration = 1000         # Duration to plot (seconds)
samples = fs * duration
eda_column = 'eda_filtered'    # Update if your ECG column has a different name
label_column = 'binary_label'

# Filter ECG signals for Normal (1) and Stress (2)
normal_eda = filtered_data[filtered_data[label_column] == 0][eda_column].values[:samples]
stress_eda = filtered_data[filtered_data[label_column] == 1][eda_column].values[:samples]

# Plot the ECG signals
plt.figure(figsize=(14, 5))
plt.plot(normal_eda, label='Normal (Baseline)', color='green')
plt.plot(stress_eda, label='Stress', color='red', alpha=0.7)
plt.title("EDA Signal Comparison: Normal vs Stress")
plt.xlabel("Sample Index")
plt.ylabel("EDA Signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# * Baseline (Normal): The green line, representing the baseline EDA signal, shows a gradual decrease over time, starting at around 5 EDA signal and ending at approximately 1.
# * Stress: The red line, representing the EDA signal under stress, initially stays relatively flat at around 2 EDA signal. However, around the sample index of 450000, there is a dramatic and sharp increase, peaking at 12, followed by a gradual decline. This indicates a significant and sudden change in skin conductance during the stress condition.
# * While the baseline signal exhibits a slow decline, the stress signal is characterized by a large and rapid increase, indicating a strong activation of the autonomic nervous system. This is a typical physiological response to a stressor, as increased sweat gland activity leads to higher skin conductance.

# %%
import matplotlib.pyplot as plt

# Parameters
fs = 700              # Sampling frequency (Hz)
duration = 10          # Duration to plot (seconds)
samples = fs * duration
emg_column = 'emg'    # Update if your ECG column has a different name
label_column = 'binary_label'

# Filter ECG signals for Normal (1) and Stress (2)
normal_emg = filtered_data[filtered_data[label_column] == 0][emg_column].values[:samples]
stress_emg = filtered_data[filtered_data[label_column] == 1][emg_column].values[:samples]

# Plot the ECG signals
plt.figure(figsize=(14, 5))
plt.plot(normal_emg, label='Normal (Baseline)', color='green')
plt.plot(stress_emg, label='Stress', color='red', alpha=0.7)
plt.title("EMG Signal Comparison: Normal vs Stress")
plt.xlabel("Sample Index")
plt.ylabel("EMG Signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt

# Parameters
fs = 700              # Sampling frequency (Hz)
duration = 10          # Duration to plot (seconds)
samples = fs * duration
acc_column = 'acc1'    # Update if your ECG column has a different name
label_column = 'binary_label'

# Filter ECG signals for Normal (1) and Stress (2)
normal_acc = filtered_data[filtered_data[label_column] == 0][acc_column].values[:samples]
stress_acc = filtered_data[filtered_data[label_column] == 1][acc_column].values[:samples]

# Plot the ECG signals
plt.figure(figsize=(14, 5))
plt.plot(normal_acc, label='Normal (Baseline)', color='green')
plt.plot(stress_acc, label='Stress', color='red', alpha=0.7)
plt.title("acc1 Signal Comparison: Normal vs Stress")
plt.xlabel("Sample Index")
plt.ylabel("acc1 Signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt

# Parameters
fs = 700              # Sampling frequency (Hz)
duration = 10          # Duration to plot (seconds)
samples = fs * duration
acc2_column = 'acc2'    # Update if your ECG column has a different name
label_column = 'binary_label'

# Filter ECG signals for Normal (1) and Stress (2)
normal_acc2 = filtered_data[filtered_data[label_column] == 0][acc2_column].values[:samples]
stress_acc2 = filtered_data[filtered_data[label_column] == 1][acc2_column].values[:samples]

# Plot the ECG signals
plt.figure(figsize=(14, 5))
plt.plot(normal_acc2, label='Normal (Baseline)', color='green')
plt.plot(stress_acc2, label='Stress', color='red', alpha=0.7)
plt.title("acc Signal Comparison: Normal vs Stress")
plt.xlabel("Sample Index")
plt.ylabel("acc Signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt

# Parameters
fs = 700              # Sampling frequency (Hz)
duration = 10          # Duration to plot (seconds)
samples = fs * duration
acc3_column = 'acc3'    # Update if your ECG column has a different name
label_column = 'binary_label'

# Filter ECG signals for Normal (1) and Stress (2)
normal_acc3 = filtered_data[filtered_data[label_column] == 0][acc3_column].values[:samples]
stress_acc3 = filtered_data[filtered_data[label_column] == 1][acc3_column].values[:samples]

# Plot the ECG signals
plt.figure(figsize=(14, 5))
plt.plot(normal_acc3, label='Normal (Baseline)', color='green')
plt.plot(stress_acc3, label='Stress', color='red', alpha=0.7)
plt.title("acc Signal Comparison: Normal vs Stress")
plt.xlabel("Sample Index")
plt.ylabel("acc Signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt

# Parameters
fs = 700              # Sampling frequency (Hz)
duration = 10          # Duration to plot (seconds)
samples = fs * duration
temp_column = 'temp'    # Update if your ECG column has a different name
label_column = 'binary_label'

# Filter ECG signals for Normal (1) and Stress (2)
normal_temp = filtered_data[filtered_data[label_column] == 0][temp_column].values[:samples]
stress_temp = filtered_data[filtered_data[label_column] == 1][temp_column].values[:samples]

# Plot the ECG signals
plt.figure(figsize=(14, 5))
plt.plot(normal_temp, label='Normal (Baseline)', color='green')
plt.plot(stress_temp, label='Stress', color='red', alpha=0.7)
plt.title("temp Signal Comparison: Normal vs Stress")
plt.xlabel("Sample Index")
plt.ylabel("temp Signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# * Normal signal is saturated from 29 to 29.5
# * Stressed signal is saturated from 31 to 31.5

# %%
import matplotlib.pyplot as plt
from scipy.signal import welch

# Replace these with actual column names
ecg_column = 'ecg_filtered'
label_column = 'binary_label'  # This should contain values like 'stressed' or 'normal'
fs = 700  # Sampling frequency in Hz (adjust to your dataset)

# Split the data
normal_ecg = filtered_data[filtered_data[label_column] == 0][ecg_column].values[:samples]
stress_ecg = filtered_data[filtered_data[label_column] == 1][ecg_column].values[:samples]


# Compute PSD using Welch's method
freq_n, psd_n = welch(normal_ecg, fs=fs)
freq_s, psd_s = welch(stress_ecg, fs=fs)

# Plot both PSDs
plt.figure(figsize=(12, 6))
plt.semilogy(freq_n, psd_n, label='Normal', color='green')
plt.semilogy(freq_s, psd_s, label='Stressed', color='red')
plt.title('Power Spectral Density: Normal vs Stressed ECG')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (V²/Hz)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# In the lower frequency range (below 50 Hz), the stressed ECG signal has slightly higher power than the normal ECG signal. This indicates that there might be more low-frequency activity in the ECG signal during stress.
# 
# As the frequency increases (above 50 Hz), the power of both signals decreases rapidly. The difference between the normal and stressed signals becomes less pronounced at higher frequencies. ( typically because of the notch filter)

# %%
import numpy as np
import matplotlib.pyplot as plt

# Sampling frequency
fs = 700  # adjust this to match your ECG sampling rate

# Get ECG signals
ecg_normal = filtered_data[filtered_data[label_column] == 0][ecg_column].values[:samples]
ecg_stress = filtered_data[filtered_data[label_column] == 1][ecg_column].values[:samples]

# Compute FFT
def compute_fft(signal, fs):
    n = len(signal)
    f = np.fft.rfftfreq(n, d=1/fs)
    fft_values = np.fft.rfft(signal)
    fft_magnitude = np.abs(fft_values) / n
    return f, fft_magnitude

freq_normal, fft_normal = compute_fft(ecg_normal, fs)
freq_stress, fft_stress = compute_fft(ecg_stress, fs)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(freq_normal, fft_normal, label='Normal ECG', color='green')
plt.plot(freq_stress, fft_stress, label='Stressed ECG', color='red')
plt.title('FFT of ECG Signals: Normal vs Stressed')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# Both signals show a prominent peak at lower frequencies, which corresponds to the heart rate.
# The stressed ECG signal has a higher magnitude at the dominant low-frequency peak. This indicates that the heart rate component is more pronounced during stress.
# As the frequency increases, the magnitude of both signals decreases.
# There are some differences in the **mid-frequency range (around 20-50 Hz)**, where the stressed ECG signal shows slightly higher magnitudes. These differences might be related to other physiological changes associated with stress.

# %%
Data = filtered_data.drop(['resp', 'ecg', 'eda','acc1','acc2','acc3','sid','emg','temp'], axis=1)
Data.head()

# %%
Data.head()

# %% [markdown]
# # Feature Extraction

# %%
import numpy as np
from scipy import signal, interpolate
from scipy.stats import skew, kurtosis
from biosppy.signals import ecg  # Needed for ECG processing

def extract_hr_hrv_features_interpolated(ecg_signal, fs=700):
    """
    Extract heart rate and HRV features from an ECG signal and interpolate HR.
    """
    # Process ECG to get R-peaks using biosppy
    ecg_out = ecg.ecg(signal=ecg_signal, sampling_rate=fs, show=False)
    rpeaks = ecg_out['rpeaks']
    
    if len(rpeaks) < 3:  # Not enough peaks for HRV analysis
        return {key: np.nan for key in [
            "hr_mean", "hr_median", "hr_min", "hr_max", "hr_variance", "hr_std",
            "hr_q1", "hr_q3", "mean_rr", "sdnn", "rmssd", "hr_skew", "hr_kurtosis"
        ]}
    
    rr_intervals = np.diff(rpeaks) / fs  # In seconds
    hr_instantaneous = 60 / rr_intervals  # BPM

    hr_times = rpeaks[:-1] + np.diff(rpeaks) / 2
    hr_times = hr_times / fs

    t_interp = np.arange(len(ecg_signal)) / fs

    f_cubic = interpolate.interp1d(
        hr_times, hr_instantaneous, kind='cubic',
        bounds_error=False, fill_value=(hr_instantaneous[0], hr_instantaneous[-1])
    )
    hr_interpolated = f_cubic(t_interp)

    return {
        "hr_mean": np.mean(hr_interpolated),
        "hr_median": np.median(hr_interpolated),
        "hr_min": np.min(hr_interpolated),
        "hr_max": np.max(hr_interpolated),
        "hr_variance": np.var(hr_interpolated),
        "hr_std": np.std(hr_interpolated),
        "hr_q1": np.percentile(hr_interpolated, 25),
        "hr_q3": np.percentile(hr_interpolated, 75),
        "mean_rr": np.mean(rr_intervals),
        "sdnn": np.std(rr_intervals),
        "rmssd": np.sqrt(np.mean(np.square(np.diff(rr_intervals)))),
        "hr_skew": skew(hr_interpolated),
        "hr_kurtosis": kurtosis(hr_interpolated)
    }

# %%
def extract_time_domain_features(signal_data):
    """
    Extract basic time-domain statistics from a signal.
    """
    signal_array = np.asarray(signal_data)

    if len(signal_array) == 0:
        return {key: np.nan for key in [
            'mean', 'median', 'variance', 'std', 'min', 'max',
            'q1', 'q3', 'skew', 'kurtosis'
        ]}
    
    return {
        'mean': np.mean(signal_array),
        'median': np.median(signal_array),
        'variance': np.var(signal_array),
        'std': np.std(signal_array),
        'min': np.min(signal_array),
        'max': np.max(signal_array),
        'q1': np.percentile(signal_array, 25),
        'q3': np.percentile(signal_array, 75),
        'skew': skew(signal_array),
        'kurtosis': kurtosis(signal_array)
    }

# %%
def extract_psd_features(signal_data, fs=700):
    """
    Extract PSD (Power Spectral Density) features using Welch’s method.
    """
    signal_array = np.asarray(signal_data)

    if len(signal_array) < 10:
        return {key: np.nan for key in [
            'psd_mean', 'psd_median', 'psd_peak_freq', 'psd_total_power',
            'psd_variance', 'psd_std', 'psd_skewness', 'psd_kurtosis',
            'psd_q1', 'psd_q3'
        ]}
    
    try:
        nperseg = min(256, len(signal_array))
        f, psd = signal.welch(signal_array, fs, nperseg=nperseg, average='median')

        return {
            'psd_mean': np.nanmean(psd),
            'psd_median': np.nanmedian(psd),
            'psd_peak_freq': f[np.nanargmax(psd)] if np.any(psd > 0) else np.nan,
            'psd_total_power': np.nansum(psd),
            'psd_variance': np.nanvar(psd),
            'psd_std': np.nanstd(psd),
            'psd_skewness': skew(psd, nan_policy='omit'),
            'psd_kurtosis': kurtosis(psd, nan_policy='omit'),
            'psd_q1': np.nanpercentile(psd, 25),
            'psd_q3': np.nanpercentile(psd, 75)
        }

    except Exception as e:
        print(f"PSD calculation failed: {str(e)}")
        return {key: np.nan for key in [
            'psd_mean', 'psd_median', 'psd_peak_freq', 'psd_total_power',
            'psd_variance', 'psd_std', 'psd_skewness', 'psd_kurtosis',
            'psd_q1', 'psd_q3'
        ]}

# %%
def extract_all_features(window_data, fs=700):
    """
    Extract all time-domain and frequency-domain features from ECG, EDA, and RESP signals.
    """
    features = {}

    # ECG
    hr_features = extract_hr_hrv_features_interpolated(window_data['ecg'], fs)
    features.update(hr_features)

    ecg_psd_features = extract_psd_features(window_data['ecg'], fs)
    features.update({f'ecg_{k}': v for k, v in ecg_psd_features.items()})

    # EDA
    eda_time_features = extract_time_domain_features(window_data['eda'])
    features.update({f'eda_{k}': v for k, v in eda_time_features.items()})

    eda_psd_features = extract_psd_features(window_data['eda'], fs)
    features.update({f'eda_{k}': v for k, v in eda_psd_features.items()})

    # RESP
    resp_time_features = extract_time_domain_features(window_data['resp'])
    features.update({f'resp_{k}': v for k, v in resp_time_features.items()})

    resp_psd_features = extract_psd_features(window_data['resp'], fs)
    features.update({f'resp_{k}': v for k, v in resp_psd_features.items()})

    return features

# %% [markdown]
# # window segmentation

# %%
def segment_df_into_windows(df, window_size_sec=10, step_size_sec=5, fs=700):
    """
    Segments a DataFrame into overlapping windows with labels.

    Parameters:
    - df: DataFrame with columns 'ecg_filtered', 'eda_filtered', 'resp_filtered', 'label', 'binary_label'
    - window_size_sec: duration of window in seconds
    - step_size_sec: step size between windows in seconds
    - fs: sampling frequency (Hz)

    Returns:
    - List of dicts with signal windows and associated labels
    """
    window_size = int(window_size_sec * fs)
    step_size = int(step_size_sec * fs)
    num_samples = len(df)
    windows = []

    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        window_df = df.iloc[start:end]

        # Get the most frequent label in the window
        label = window_df['label'].mode()[0]
        binary_label = window_df['binary_label'].mode()[0]

        window = {
            'ecg': window_df['ecg_filtered'].values,
            'eda': window_df['eda_filtered'].values,
            'resp': window_df['resp_filtered'].values,
            'label': label,
            'binary_label': binary_label
        }
        windows.append(window)

    return windows


# %%
Data_60_30=segment_df_into_windows(Data,60,30,700)

# %%
Data_60_10=segment_df_into_windows(Data,60,10,700)

# %%
Data_60_20=segment_df_into_windows(Data,60,20,700)

# %%
# Initialize a list to store all the extracted features
feature_list = []

# Loop through each window
for i in range(len(Data_60_10)):
    features = {}

    # Extract the current window
    window = Data_60_10[i]

    # Get the current window's signals
    window_data = {
        'ecg': window['ecg'],
        'eda': window['eda'],
        'resp': window['resp']
    }

    # Extract features
    all_features = extract_all_features(window_data)
    features.update(all_features)

    # Add labels
    features['label'] = window['label']
    features['binary_label'] = window['binary_label']

    # Store the feature dict
    feature_list.append(features)

# Convert list of dicts to a DataFrame
features_df_60_10 = pd.DataFrame(feature_list)


# %%
# Initialize a list to store all the extracted features
feature_list = []

# Loop through each window
for i in range(len(Data_60_20)):
    features = {}

    # Extract the current window
    window = Data_60_20[i]

    # Get the current window's signals
    window_data = {
        'ecg': window['ecg'],
        'eda': window['eda'],
        'resp': window['resp']
    }

    # Extract features
    all_features = extract_all_features(window_data)
    features.update(all_features)

    # Add labels
    features['label'] = window['label']
    features['binary_label'] = window['binary_label']

    # Store the feature dict
    feature_list.append(features)

# Convert list of dicts to a DataFrame
features_df_60_20 = pd.DataFrame(feature_list)


# %%
# Initialize a list to store all the extracted features
feature_list = []

# Loop through each window
for i in range(len(Data_60_30)):
    features = {}

    # Extract the current window
    window = Data_60_30[i]

    # Get the current window's signals
    window_data = {
        'ecg': window['ecg'],
        'eda': window['eda'],
        'resp': window['resp']
    }

    # Extract features
    all_features = extract_all_features(window_data)
    features.update(all_features)

    # Add labels
    features['label'] = window['label']
    features['binary_label'] = window['binary_label']

    # Store the feature dict
    feature_list.append(features)

# Convert list of dicts to a DataFrame
features_df_60_30 = pd.DataFrame(feature_list)


# %%
information(features_df_60_10)

# %%
information(features_df_60_20)

# %%
information(features_df_60_30)

# %%
features_df_60_30 = features_df_60_30.drop(columns=['resp_psd_peak_freq', 'eda_psd_peak_freq'], inplace=False)
features_df_60_20 = features_df_60_20.drop(columns=['resp_psd_peak_freq', 'eda_psd_peak_freq'], inplace=False)
features_df_60_10 = features_df_60_10.drop(columns=['resp_psd_peak_freq', 'eda_psd_peak_freq'], inplace=False)

# %% [markdown]
# Removing ['resp_psd_peak_freq', 'eda_psd_peak_freq']

# %% [markdown]
# # Correlation Matrix

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = features_df_60_30.corr()

# Set figure size to accommodate the number of features
plt.figure(figsize=(20, 16))

# Create the heatmap with the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
            linewidths=0.5, annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})

# Rotate labels for better visibility
plt.xticks(rotation=90, ha='center')
plt.yticks(rotation=0, ha='right')

# Add title and show the plot
title="Correlation Heatmap of 65 Features"
plt.title("Correlation Heatmap of 65 Features", fontsize=16)
plt.tight_layout()  # Adjust layout to avoid label cut-off
plt.savefig(f'{title.lower().replace(" ", "_")}.png')
plt.show()


# %% [markdown]
# # TSNE Visualizations

# %%
def plot_tsne(X, y, title, labels):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='plasma', s=50, alpha=0.6)
    plt.title(title)
    plt.xlabel('tSNE-1')
    plt.ylabel('tSNE-2')
    plt.legend(handles=scatter.legend_elements()[0], labels=labels, title="Label")
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')  # Save plot
    plt.show()

# %%
from imblearn.under_sampling import RandomUnderSampler

# Set random seed for reproducibility
np.random.seed(42)

# Prepare features (exclude label columns)
feature_cols = [col for col in features_df_60_30.columns if col not in ['label', 'binary_label']]
X = features_df_60_30[feature_cols].values

# Standardize features (to improve t-SNE performance for float data)
scaler = StandardScaler()
X = scaler.fit_transform(X)

y_binary = features_df_60_30['binary_label'].values
y_multi = features_df_60_30['label'].values

# 1. Binary Classification (using binary_label)
# Before Random Undersampling
plot_tsne(X, y_binary, "t-SNE Visualization (Binary - Before Random Undersampling)", labels=['0', '1'])

# Apply Random Undersampling (Binary)
rus_binary = RandomUnderSampler(random_state=42)
X_binary_rus, y_binary_rus = rus_binary.fit_resample(X, y_binary)

# After Random Undersampling (Binary)
plot_tsne(X_binary_rus, y_binary_rus, "t-SNE Visualization (Binary - After Random Undersampling)", labels=['0', '1'])

# 2. Multi-Class Classification (using label)
# Before Random Undersampling
plot_tsne(X, y_multi, "t-SNE Visualization (Multi-Class - Before Random Undersampling)", labels=['0', '1', '2', '3'])

# Apply Random Undersampling (Multi-Class)
rus_multi = RandomUnderSampler(random_state=42)
X_multi_rus, y_multi_rus = rus_multi.fit_resample(X, y_multi)

# After Random Undersampling (Multi-Class)
plot_tsne(X_multi_rus, y_multi_rus, "t-SNE Visualization (Multi-Class - After Random Undersampling)", labels=['0', '1', '2', '3'])

# %%
from imblearn.under_sampling import ClusterCentroids

# Set random seed for reproducibility
np.random.seed(42)
feature_cols = [col for col in features_df_60_30.columns if col not in ['label', 'binary_label']]
X = features_df_60_30[feature_cols].values
y_binary = features_df_60_30['binary_label'].values
y_multi = features_df_60_30['label'].values

# Standardize features (optional, to improve clustering)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

y_binary = features_df_60_30['binary_label'].values
y_multi = features_df_60_30['label'].values

# 1. Binary Classification (using binary_label)
# Before ClusterCentroids
plot_tsne(X, y_binary, "t-SNE Visualization (Binary - Before ClusterCentroids)", labels=['0', '1'])

# Apply ClusterCentroids (Binary)
clustercentroids_binary = ClusterCentroids(random_state=42)
X_binary_cluster, y_binary_cluster = clustercentroids_binary.fit_resample(X, y_binary)

# After ClusterCentroids (Binary)
plot_tsne(X_binary_cluster, y_binary_cluster, "t-SNE Visualization (Binary - After ClusterCentroids)", labels=['0', '1'])

# 2. Multi-Class Classification (using label)
# Before ClusterCentroids
plot_tsne(X, y_multi, "t-SNE Visualization (Multi-Class - Before ClusterCentroids)", labels=['0', '1', '2', '3'])

# Apply ClusterCentroids (Multi-Class)
clustercentroids_multi = ClusterCentroids(random_state=42)
X_multi_cluster, y_multi_cluster = clustercentroids_multi.fit_resample(X, y_multi)

# After ClusterCentroids (Multi-Class)
plot_tsne(X_multi_cluster, y_multi_cluster, "t-SNE Visualization (Multi-Class - After ClusterCentroids)", labels=['0', '1', '2', '3'])

# %%
from imblearn.under_sampling import NearMiss

# Set random seed for reproducibility
np.random.seed(42)

feature_cols = [col for col in features_df_60_30.columns if col not in ['label', 'binary_label']]
X = features_df_60_30[feature_cols].values
y_binary = features_df_60_30['binary_label'].values
y_multi = features_df_60_30['label'].values

# 1. Binary Classification (using binary_label)
# Before NearMiss
plot_tsne(X, y_binary, "t-SNE Visualization (Binary - Before NearMiss)", labels=['0', '1'])

# Apply NearMiss (Binary)
nearmiss_binary = NearMiss(version=1)
X_binary_nearmiss, y_binary_nearmiss = nearmiss_binary.fit_resample(X, y_binary)

# After NearMiss (Binary)
plot_tsne(X_binary_nearmiss, y_binary_nearmiss, "t-SNE Visualization (Binary - After NearMiss)", labels=['0', '1'])

# 2. Multi-Class Classification (using label)
# Before NearMiss
plot_tsne(X, y_multi, "t-SNE Visualization (Multi-Class - Before NearMiss)", labels=['0', '1', '2', '3'])

# Apply NearMiss (Multi-Class)
nearmiss_multi = NearMiss(version=1)
X_multi_nearmiss, y_multi_nearmiss = nearmiss_multi.fit_resample(X, y_multi)

# After NearMiss (Multi-Class)
plot_tsne(X_multi_nearmiss, y_multi_nearmiss, "t-SNE Visualization (Multi-Class - After NearMiss)", labels=['0', '1', '2', '3'])

# %%
from imblearn.combine import SMOTETomek

# Set random seed for reproducibility
np.random.seed(42)

feature_cols = [col for col in features_df_60_30.columns if col not in ['label', 'binary_label']]
X = features_df_60_30[feature_cols].values
y_binary = features_df_60_30['binary_label'].values
y_multi = features_df_60_30['label'].values

# 1. Binary Classification (using binary_label)
# Before SMOTE-Tomek
plot_tsne(X, y_binary, "t-SNE Visualization (Binary - Before SMOTE-Tomek)", labels=['0', '1'])

# Apply SMOTE-Tomek (Binary)
smotetomek_binary = SMOTETomek(random_state=42)
X_binary_smotetomek, y_binary_smotetomek = smotetomek_binary.fit_resample(X, y_binary)

# After SMOTE-Tomek (Binary)
plot_tsne(X_binary_smotetomek, y_binary_smotetomek, "t-SNE Visualization (Binary - After SMOTE-Tomek)", labels=['0', '1'])

# 2. Multi-Class Classification (using label)
# Before SMOTE-Tomek
plot_tsne(X, y_multi, "t-SNE Visualization (Multi-Class - Before SMOTE-Tomek)", labels=['0', '1', '2', '3'])

# Apply SMOTE-Tomek (Multi-Class)
smotetomek_multi = SMOTETomek(random_state=42)
X_multi_smotetomek, y_multi_smotetomek = smotetomek_multi.fit_resample(X, y_multi)

# After SMOTE-Tomek (Multi-Class)
plot_tsne(X_multi_smotetomek, y_multi_smotetomek, "t-SNE Visualization (Multi-Class - After SMOTE-Tomek)", labels=['0', '1', '2', '3'])

# %%
from imblearn.under_sampling import TomekLinks

# Set random seed for reproducibility
np.random.seed(42)
# Prepare features (exclude label columns)
feature_cols = [col for col in features_df_60_30.columns if col not in ['label', 'binary_label']]
X = features_df_60_30[feature_cols].values
y_binary = features_df_60_30['binary_label'].values
y_multi = features_df_60_30['label'].values

# 1. Binary Classification (using binary_label)
# Before Tomek Links
plot_tsne(X, y_binary, "t-SNE Visualization (Binary - Before Tomek Links)", labels=['0', '1'])

# Apply Tomek Links (Binary)
tomek_binary = TomekLinks()
X_binary_tomek, y_binary_tomek = tomek_binary.fit_resample(X, y_binary)

# After Tomek Links (Binary)
plot_tsne(X_binary_tomek, y_binary_tomek, "t-SNE Visualization (Binary - After Tomek Links)", labels=['0', '1'])

# 2. Multi-Class Classification (using label)
# Before Tomek Links
plot_tsne(X, y_multi, "t-SNE Visualization (Multi-Class - Before Tomek Links)", labels=['0', '1', '2', '3'])

# Apply Tomek Links (Multi-Class)
tomek_multi = TomekLinks()
X_multi_tomek, y_multi_tomek = tomek_multi.fit_resample(X, y_multi)

# After Tomek Links (Multi-Class)
plot_tsne(X_multi_tomek, y_multi_tomek, "t-SNE Visualization (Multi-Class - After Tomek Links)", labels=['0', '1', '2', '3'])

# %%
from imblearn.over_sampling import RandomOverSampler

# Set random seed for reproducibility
np.random.seed(42)
# Prepare features (exclude label columns)
feature_cols = [col for col in features_df_60_30.columns if col not in ['label', 'binary_label']]
X = features_df_60_30[feature_cols].values
y_binary = features_df_60_30['binary_label'].values
y_multi = features_df_60_30['label'].values

# 1. Binary Classification (using binary_label)
# Before Random Oversampling
plot_tsne(X, y_binary, "t-SNE Visualization (Binary - Before Random Oversampling)", labels=['0', '1'])

# Apply Random Oversampling (Binary)
random_oversampler_binary = RandomOverSampler(random_state=42)
X_binary_random, y_binary_random = random_oversampler_binary.fit_resample(X, y_binary)

# After Random Oversampling (Binary)
plot_tsne(X_binary_random, y_binary_random, "t-SNE Visualization (Binary - After Random Oversampling)", labels=['0', '1'])

# 2. Multi-Class Classification (using label)
# Before Random Oversampling
plot_tsne(X, y_multi, "t-SNE Visualization (Multi-Class - Before Random Oversampling)", labels=['0', '1', '2', '3'])

# Apply Random Oversampling (Multi-Class)
random_oversampler_multi = RandomOverSampler(random_state=42)
X_multi_random, y_multi_random = random_oversampler_multi.fit_resample(X, y_multi)

# After Random Oversampling (Multi-Class)
plot_tsne(X_multi_random, y_multi_random, "t-SNE Visualization (Multi-Class - After Random Oversampling)", labels=['0', '1', '2', '3'])

# %%
from imblearn.over_sampling import BorderlineSMOTE

# Set random seed for reproducibility
np.random.seed(42)

# Function to plot t-SNE visualization

# Assuming features_df_60_30 is your DataFrame with features and labels
# Prepare features (exclude label columns)
feature_cols = [col for col in features_df_60_30.columns if col not in ['label', 'binary_label']]
X = features_df_60_30[feature_cols].values
y_binary = features_df_60_30['binary_label'].values
y_multi = features_df_60_30['label'].values

# 1. Binary Classification (using binary_label)
# Before Borderline-SMOTE
plot_tsne(X, y_binary, "t-SNE Visualization (Binary - Before Borderline-SMOTE)", labels=['0', '1'])

# Apply Borderline-SMOTE (Binary)
borderline_smote_binary = BorderlineSMOTE(random_state=42)
X_binary_borderline, y_binary_borderline = borderline_smote_binary.fit_resample(X, y_binary)

# After Borderline-SMOTE (Binary)
plot_tsne(X_binary_borderline, y_binary_borderline, "t-SNE Visualization (Binary - After Borderline-SMOTE)", labels=['0', '1'])

# 2. Multi-Class Classification (using label)
# Before Borderline-SMOTE
plot_tsne(X, y_multi, "t-SNE Visualization (Multi-Class - Before Borderline-SMOTE)", labels=['0', '1', '2', '3'])

# Apply Borderline-SMOTE (Multi-Class)
borderline_smote_multi = BorderlineSMOTE(random_state=42)
X_multi_borderline, y_multi_borderline = borderline_smote_multi.fit_resample(X, y_multi)

# After Borderline-SMOTE (Multi-Class)
plot_tsne(X_multi_borderline, y_multi_borderline, "t-SNE Visualization (Multi-Class - After Borderline-SMOTE)", labels=['0', '1', '2', '3'])

# %%
from imblearn.over_sampling import ADASYN

# Set random seed for reproducibility
np.random.seed(42)

# Assuming features_df_60_30 is your DataFrame with features and labels
# Prepare features (exclude label columns)
feature_cols = [col for col in features_df_60_30.columns if col not in ['label', 'binary_label']]
X = features_df_60_30[feature_cols].values
y_binary = features_df_60_30['binary_label'].values
y_multi = features_df_60_30['label'].values

# 1. Binary Classification (using binary_label)
# Before ADASYN
plot_tsne(X, y_binary, "t-SNE Visualization (Binary - Before ADASYN)", labels=['0', '1'])

# Apply ADASYN (Binary)
adasyn_binary = ADASYN(random_state=42)
X_binary_adasyn, y_binary_adasyn = adasyn_binary.fit_resample(X, y_binary)

# After ADASYN (Binary)
plot_tsne(X_binary_adasyn, y_binary_adasyn, "t-SNE Visualization (Binary - After ADASYN)", labels=['0', '1'])

# 2. Multi-Class Classification (using label)
# Before ADASYN
plot_tsne(X, y_multi, "t-SNE Visualization (Multi-Class - Before ADASYN)", labels=['0', '1', '2', '3'])

# Apply ADASYN (Multi-Class)
adasyn_multi = ADASYN(random_state=42)
X_multi_adasyn, y_multi_adasyn = adasyn_multi.fit_resample(X, y_multi)

# After ADASYN (Multi-Class)
plot_tsne(X_multi_adasyn, y_multi_adasyn, "t-SNE Visualization (Multi-Class - After ADASYN)", labels=['0', '1', '2', '3'])

# %%
# Set random seed for reproducibility
np.random.seed(42)
# Prepare features (exclude label columns)
feature_cols = [col for col in features_df_60_30.columns if col not in ['label', 'binary_label']]
X = features_df_60_30[feature_cols].values
y_binary = features_df_60_30['binary_label'].values
y_multi = features_df_60_30['label'].values

# 1. Binary Classification (using binary_label)
# Before SMOTE
plot_tsne(X, y_binary, "t-SNE Visualization (Binary - Before SMOTE)", labels=['0', '1'])

# Apply SMOTE (Binary)
smote_binary = SMOTE(random_state=42)
X_binary_smote, y_binary_smote = smote_binary.fit_resample(X, y_binary)

# After SMOTE (Binary)
plot_tsne(X_binary_smote, y_binary_smote, "t-SNE Visualization (Binary - After SMOTE)", labels=['0', '1'])

# 2. Multi-Class Classification (using label)
# Before SMOTE
plot_tsne(X, y_multi, "t-SNE Visualization (Multi-Class - Before SMOTE)", labels=['0', '1', '2', '3'])

# Apply SMOTE (Multi-Class)
smote_multi = SMOTE(random_state=42)
X_multi_smote, y_multi_smote = smote_multi.fit_resample(X, y_multi)

# After SMOTE (Multi-Class)
plot_tsne(X_multi_smote, y_multi_smote, "t-SNE Visualization (Multi-Class - After SMOTE)", labels=['0', '1', '2', '3'])

# %%
X = features_df_60_30.drop(['label', 'binary_label'], axis=1)
y = features_df_60_30['label'].astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)


# %% [markdown]
# Multi

# %%
# Create a DataFrame for plotting
tsne_df = pd.DataFrame({
    'TSNE-1': X_tsne[:, 0],
    'TSNE-2': X_tsne[:, 1],
    'Label': y
})

plt.figure(figsize=(10, 8))
sns.scatterplot(data=tsne_df, x='TSNE-1', y='TSNE-2', hue='Label', palette='deep')
plt.title('t-SNE Visualization of Feature Space')
plt.show()

# %% [markdown]
# * Label 1.0 (blue) and label 4.0 (red) points are more spread out, with some overlap in certain regions.
# * Label 2.0 (orange) forms a tight cluster on the right side, suggesting strong similarity within this group.
# * Label 3.0 (green) points are scattered but tend to cluster near the center-left.

# %%
X = features_df_60_30.drop(['label', 'binary_label'], axis=1)
y = features_df_60_30['binary_label'].astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
tsne = TSNE(n_components=3, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# %% [markdown]
# Binary

# %%
# Create a DataFrame for plotting
tsne_df = pd.DataFrame({
    'TSNE-1': X_tsne[:, 0],
    'TSNE-2': X_tsne[:, 1],
    'Label': y
})

plt.figure(figsize=(10, 8))
sns.scatterplot(data=tsne_df, x='TSNE-1', y='TSNE-2', hue='Label', palette='deep')
plt.title('t-SNE Visualization of Feature Space')
plt.show()

# %% [markdown]
# # MACHINE LEARNING APPROACH

# %% [markdown]
# # Evaluation metrics

# %%
# Learning curves
def plot_learning_curve(estimator, X, y, filename):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, val_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename)
    plt.show()
def evaluate_classifier(
    model,
    X_train,
    y_train,
    y_train_pred,
    X_test,
    y_test,
    y_test_pred,
    cv_folds=3,
    learning_curve_path='learning_curve.png',
    cm_train_path='Confusion_Matrix_train.png',
    cm_test_path='Confusion_Matrix_test.png'
):
    """
    Evaluates a classifier with classification reports, learning curve, confusion matrices,
    cross-validation scores, and accuracy metrics.
    
    Parameters:
    - model: Trained classifier (e.g., RandomForestClassifier)
    - X_train: Training features (float-based, e.g., WESAD PSD features)
    - y_train_resampled: Resampled training labels (e.g., after Random Undersampling)
    - y_train_pred: Predicted labels for training set
    - X_test: Test features
    - y_test: Test labels
    - y_test_pred: Predicted labels for test set
    - cv_folds: Number of cross-validation folds (default: 3)
    - learning_curve_path: Path to save learning curve plot
    - cm_train_path: Path to save train confusion matrix
    - cm_test_path: Path to save test confusion matrix
    """
    # Classification reports
    print("Train Classification Report:")
    print(classification_report(y_train, y_train_pred))
    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Plot learning curve
    plot_learning_curve(model, X_train, y_train, learning_curve_path)

    # Plot confusion matrices
    plot_confusion_matrix(y_train, y_train_pred, 'Confusion Matrix - Train Set', cm_train_path)
    plot_confusion_matrix(y_test, y_test_pred, 'Confusion Matrix - Test Set', cm_test_path)

    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, n_jobs=-1, scoring='accuracy')
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Score:", np.mean(cv_scores))
    print("Standard Deviation:", np.std(cv_scores))

    # Train and Test Accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)


# %% [markdown]
# # LogisticRegression

# %% [markdown]
# # **60_30**

# %%
X = features_df_60_30.drop(['label', 'binary_label'], axis=1)
y = features_df_60_30['binary_label'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# # T0P 50 MOST IMPORTANT FEATURES

# %%
# Feature selection
selector = SelectKBest(score_func=f_classif, k=50)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_resampled)
X_test_selected = selector.transform(X_test_scaled)

# Get the boolean mask of selected features
selected_features_mask = selector.get_support()

# Get the names of all features
all_feature_names = pd.DataFrame(X_train).columns  
# Get the names of the selected features
feature_scores = selector.scores_
# Plot the feature scores
plt.figure(figsize=(12, 6))
plt.bar(all_feature_names, feature_scores)
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("F-statistic Score")
plt.title("Feature Importance (SelectKBest with F-statistic)")
plt.tight_layout()
plt.show()

# %% [markdown]
# # 1.binary

# %%
# Logistic Regression with balanced class weights
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

# Fit and evaluate
model.fit(X_train_selected, y_train_resampled)
y_train_pred = model.predict(X_train_selected)
y_test_pred = model.predict(X_test_selected)

# %%
evaluate_classifier(
    model,
    X_train_resampled,
    y_train_resampled,
    y_train_pred,
    X_test,
    y_test,
    y_test_pred
)

# %% [markdown]
# saving best model

# %%
filename_model = 'logistic_regression_model.pkl'
filename_scaler = 'scaler.pkl'

pickle.dump(model, open(filename_model, 'wb'))
pickle.dump(scaler, open(filename_scaler, 'wb'))  # Save the scaler

print(f"Model saved to {filename_model}")
print(f"Scaler saved to {filename_scaler}")


# %% [markdown]
# # **Multi**

# %%
X = features_df_60_30.drop(['label', 'binary_label'], axis=1)
y = features_df_60_30['label'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)
# Feature selection
selector = SelectKBest(score_func=f_classif, k=50)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_resampled)
X_test_selected = selector.transform(X_test_scaled)

# %%

# Logistic Regression with balanced class weights
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

# Fit and evaluate
model.fit(X_train_selected, y_train_resampled)
y_train_pred = model.predict(X_train_selected)
y_test_pred = model.predict(X_test_selected)

# %%
evaluate_classifier(
    model,
    X_train_resampled,
    y_train_resampled,
    y_train_pred,
    X_test,
    y_test,
    y_test_pred
)

# %% [markdown]
# # **60_20**

# %%
X = features_df_60_20.drop(['label', 'binary_label'], axis=1)
y = features_df_60_20['binary_label'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=50)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_resampled)
X_test_selected = selector.transform(X_test_scaled)

# %% [markdown]
# # binary

# %%

# Logistic Regression with balanced class weights
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

# Fit and evaluate
model.fit(X_train_selected, y_train_resampled)
y_train_pred = model.predict(X_train_selected)
y_test_pred = model.predict(X_test_selected)

# %%
evaluate_classifier(
    model,
    X_train_resampled,
    y_train_resampled,
    y_train_pred,
    X_test,
    y_test,
    y_test_pred
)

# %% [markdown]
# # **Multi**

# %%
X_m = features_df_60_20.drop(['label', 'binary_label'], axis=1)
y_m = features_df_60_20['label'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X_m, y_m, test_size=0.2, random_state=42, stratify=y_m
)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=50)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_resampled)
X_test_selected = selector.transform(X_test_scaled)

# %%

# Logistic Regression with balanced class weights
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

# Fit and evaluate
model.fit(X_train_selected, y_train_resampled)
y_train_pred = model.predict(X_train_selected)
y_test_pred = model.predict(X_test_selected)

# %%
evaluate_classifier(
    model,
    X_train_resampled,
    y_train_resampled,
    y_train_pred,
    X_test,
    y_test,
    y_test_pred
)

# %%
filename_model = 'logistic_regression_model.pkl'
filename_scaler = 'scaler.pkl'

pickle.dump(model, open(filename_model, 'wb'))
pickle.dump(scaler, open(filename_scaler, 'wb'))  # Save the scaler

print(f"Model saved to {filename_model}")
print(f"Scaler saved to {filename_scaler}")

# %% [markdown]
# # **60_10**

# %%
X = features_df_60_10.drop(['label', 'binary_label'], axis=1)
y = features_df_60_10['binary_label'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=50)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_resampled)
X_test_selected = selector.transform(X_test_scaled)

# %% [markdown]
# # **binary**

# %%
# Logistic Regression with balanced class weights
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

# Fit and evaluate
model.fit(X_train_selected, y_train_resampled)
y_train_pred = model.predict(X_train_selected)
y_test_pred = model.predict(X_test_selected)

# %%
evaluate_classifier(
    model,
    X_train_resampled,
    y_train_resampled,
    y_train_pred,
    X_test,
    y_test,
    y_test_pred
)

# %% [markdown]
# # **Multi**

# %%
X_m = features_df_60_20.drop(['label', 'binary_label'], axis=1)
y_m = features_df_60_20['label'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X_m, y_m, test_size=0.2, random_state=42, stratify=y_m
)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=50)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_resampled)
X_test_selected = selector.transform(X_test_scaled)

# %%
# Logistic Regression with balanced class weights
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

# Fit and evaluate
model.fit(X_train_selected, y_train_resampled)
y_train_pred = model.predict(X_train_selected)
y_test_pred = model.predict(X_test_selected)

# %%
evaluate_classifier(
    model,
    X_train_resampled,
    y_train_resampled,
    y_train_pred,
    X_test,
    y_test,
    y_test_pred
)

# %% [markdown]
# # 60_30 without smote

# %% [markdown]
# binary

# %%
X = features_df_60_30.drop(['label', 'binary_label'], axis=1)
y = features_df_60_30['binary_label'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=50)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# %%
# Logistic Regression with balanced class weights
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

# Fit and evaluate
model.fit(X_train_selected, y_train)
y_train_pred = model.predict(X_train_selected)
y_test_pred = model.predict(X_test_selected)

# %%
evaluate_classifier(
    model,
    X_train_selected,
    y_train,
    y_train_pred,
    X_test,
    y_test,
    y_test_pred
)

# %%



