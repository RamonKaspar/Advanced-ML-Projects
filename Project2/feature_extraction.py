import pandas as pd
import numpy as np
from biosppy.signals import ecg
import neurokit2 as nk
import matplotlib.pyplot as plt

SAMPLING_RATE = 300

""" 
In this script, we test and plot the feature extraction process for a single ECG signal.
This includes: Extraction of R-peaks, extraction of heartbeat templates, and position of characteristic points (P, Q, R, S, T).
"""

# Load data
X_train = pd.read_parquet('data/train.parquet').drop(columns=['id', 'y'])

# We choose one signal at random
random_idx = np.random.randint(0, len(X_train))
signal = X_train.iloc[random_idx].dropna().to_numpy()

# Feature extraction
ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(
    signal=signal, 
    sampling_rate=SAMPLING_RATE, 
    show=False
)
# Get the interesting points
signals, info = nk.ecg_delineate(ecg_cleaned=filtered, rpeaks=rpeaks, sampling_rate=SAMPLING_RATE, show=False)
points = {
    'P': info['ECG_P_Peaks'],
    'Q': info['ECG_Q_Peaks'],
    'R': rpeaks,
    'S': info['ECG_S_Peaks'],
    'T': info['ECG_T_Peaks']
}

# Print information
print(f"\nSignal length: {len(signal)} samples ({len(signal)/SAMPLING_RATE:.1f} seconds)")
print(f"Number of R-peaks detected: {len(rpeaks)}")
print(f"Number of templates extracted: {templates.shape[0]}")
assert templates.shape[0] == len(rpeaks)
print(f"Mean heart rate: {np.mean(heart_rate):.1f} bpm")


# == PLOTTING ==
fig = plt.figure(figsize=(15, 15))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

# Plot 1: R-peaks (marked in red) overlaid on the signal
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(ts, filtered, label='Filtered Signal')
ax1.scatter(ts[rpeaks], filtered[rpeaks], color='red', label='R-peaks')
ax1.set_title(f"ECG Signal with R-peaks ({len(rpeaks)} detected) -- Signal {random_idx}")
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.legend()

# Plot 2: All templates overlaid, draw mean template in red
ax2 = fig.add_subplot(gs[1, :])
for template in templates:
    ax2.plot(templates_ts, template, alpha=0.1, color='blue')
ax2.plot(templates_ts, np.mean(templates, axis=0), color='red', label='Mean Template')
ax2.set_title(f'All Heartbeat Templates ({templates.shape[0]} beats)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude')
ax2.legend()


def plot_heartbeat(ax, template_idx):
    """Plot a single heartbeat template with its interesting points."""
    r_peak = rpeaks[template_idx]   # Get the corresponding R-peak and window around it from the original signal
    # Use same window size as template
    half_window = len(templates_ts) // 2
    start_idx = r_peak - half_window
    end_idx = r_peak + half_window
    
    # Plot the signal within the window
    time_window = ts[start_idx:end_idx]
    signal_window = filtered[start_idx:end_idx]
    ax.plot(time_window, signal_window, label='Signal')
    
    # Plot characteristic points
    colors = {'P': 'green', 'Q': 'orange', 'R': 'red', 'S': 'purple', 'T': 'brown'}
    for point_name, point_indices in points.items():
        point_indices = np.array(point_indices)
        point_indices = point_indices[~np.isnan(point_indices)].astype(int)
        # Find points that fall within our window
        mask = (point_indices >= start_idx) & (point_indices < end_idx)
        if np.any(mask):
            ax.scatter(ts[point_indices[mask]], 
                filtered[point_indices[mask]], 
                color=colors[point_name],
                label=point_name,
                s=100
            )
    ax.set_title(f'Heartbeat (Template {template_idx})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()

# Plot 3 & 4: Two random heartbeat templates (side by side)
random_indices = np.random.choice(len(rpeaks), 2, replace=False)
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[2, 1])

plot_heartbeat(ax3, random_indices[0])
plot_heartbeat(ax4, random_indices[1])

plt.savefig('plots/ecg_template_analysis.pdf')