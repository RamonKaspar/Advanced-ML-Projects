import pandas as pd
import matplotlib.pyplot as plt

# Load the training data
data = pd.read_csv('data/train.csv', index_col='id')
X = data.drop('y', axis=1)
y = data['y']

# Class Distribution
class_counts = y.value_counts().sort_index()
print("Class Distribution:\n", class_counts)

plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.savefig('plots/class_distribution.pdf')

# Signal Lengths
signal_lengths = X.notna().sum(axis=1)
print("\nSignal Lengths:\n", signal_lengths.describe())

plt.figure(figsize=(8, 6))
plt.hist(signal_lengths, bins=50)
plt.title('Signal Length Distribution')
plt.xlabel('Signal Length')
plt.ylabel('Frequency')
plt.savefig('plots/signal_length_distribution.pdf')

# Visualize Sample Signals
num_classes = y.nunique()
fig, axs = plt.subplots(num_classes, 1, figsize=(12, num_classes * 3))

for class_id in range(num_classes):
    sample_id = y[y == class_id].index[0]
    signal = X.loc[sample_id].dropna().values
    axs[class_id].plot(signal)
    axs[class_id].set_title(f'Class {class_id} Sample Signal')
    axs[class_id].set_xlabel('Time')
    axs[class_id].set_ylabel('Amplitude')

plt.tight_layout()
plt.savefig('plots/sample_signals.pdf')
