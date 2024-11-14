import numpy as np
import pandas as pd
from sklearn import svm, metrics, model_selection, preprocessing, impute, ensemble
from biosppy.signals import ecg
import neurokit2 as nk
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import cpu_count
from joblib import Parallel, delayed


RANDOM_STATE = 69
SAMPLING_RATE = 300

# Number of samples to use for training and testing, None to use all samples
SAMPLES = None
# Whether to make predictions on the test set, if False, we use a validation set. If True, we use the 
# full training set to train the model and make predictions on the test set.
PREDICTION = True    

def main():
    # Load data
    print("Loading data...")
    X_train = pd.read_parquet('data/train.parquet').drop(columns=['id', 'y'])
    X_test = pd.read_parquet('data/test.parquet').drop(columns='id')
    y_train = pd.read_parquet('data/train.parquet')['y'].values.ravel()
    
    # Print initial class distribution
    print_class_distribution("Full dataset", y_train, plot=True)
    
    # Create train/test split (90/10)
    if not PREDICTION:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X_train, y_train, test_size=0.1, random_state=RANDOM_STATE,
            stratify=y_train  # Add stratification to maintain class distribution
        ) 
        # Limit samples if specified
        if SAMPLES is not None:
            X_train = X_train.iloc[:SAMPLES]
            y_train = y_train[:SAMPLES]
            X_test = X_test.iloc[:SAMPLES]
            y_test = y_test[:SAMPLES]
    
    # Preprocess data (remove NaNs and filter by minimum length)
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    
    print_class_distribution(f"Training set", y_train, plot=False)
    if not PREDICTION: print_class_distribution(f"Test set", y_test, plot=False)
    
    # Feature extraction with tracking of valid signals
    print("\n== Extracting Features ==")
    X_train_features = extract_features(X_train, parallel=True)
    X_test_features = extract_features(X_test, parallel=True)
    print(f"Shape of training features: {X_train_features.shape}")
    print(f"Shape of test features: {X_test_features.shape}")
    
    # Standardize data
    print("\n== Standardizing Data ==")
    X_train_features, X_test_features = standardize_data(method='standard', X_train=X_train_features, X_test=X_test_features)
    
    # Might have NaN values in features, so we impute them
    print("\n== Imputing NaN values ==")
    num_nan_train = np.isnan(X_train_features).sum()
    num_nan_test = np.isnan(X_test_features).sum()
    num_nan_relative_train = num_nan_train / X_train_features.size
    num_nan_relative_test = num_nan_test / X_test_features.size
    print(f"Number of NaN values in training features: {num_nan_train} ({num_nan_relative_train:.2f}%)")
    print(f"Number of NaN values in test features: {num_nan_test} ({num_nan_relative_test:.2f}%)")
    X_train_features, X_test_features = fill_missing_values(X_train_features, strategy='mean', X_test=X_test_features)
    assert np.isnan(X_train_features).sum() == 0
    assert np.isnan(X_test_features).sum() == 0
    
    # TODO: Feature selection
    
    # Train Histogram Gradient Boosting Classifier
    print("Train Histogram Gradient Boosting Classifier")
    hgb = ensemble.HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        max_iter=1000,
        # verbose=1
    )
    
    # Predict a score using cross-validation
    score = model_selection.cross_val_score(hgb, X_train_features, y_train, cv=5, scoring='f1_micro')
    print(f"Cross-validation score (f1 micro): {score.mean():.4f} (+/- {score.std() * 2:.4f})")
    
    # Train on the full training set
    hgb.fit(X_train_features, y_train)  
    
    if not PREDICTION:
        # Evaluate on the evaluation set
        y_pred_hgb = hgb.predict(X_test_features)
        evaluate_predictions(y_test, y_pred_hgb, "Histogram Gradient Boosting")
    else:
        # Make predictions on the test set
        print("\n== Making Predictions ==")
        X_test_final = X_test_features
        y_pred_final = hgb.predict(X_test_final)
        create_final_submission(y_pred_final, "submission.csv")
    

def print_class_distribution(name, y, plot=False):
    """Print and plot the distribution of classes in a dataset."""
    counts = Counter(y)
    total = len(y)
    print(f"\n{name} class distribution ({len(y)} samples):")
    for i in range(4): 
        count = counts.get(i, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"Class {i}: {count} samples ({percentage:.1f}%)")
    
    if plot:
        plt.figure(figsize=(8, 5))
        class_labels = sorted(counts.keys())
        class_counts = [counts[label] for label in class_labels]
        plt.bar(class_labels, class_counts, color='skyblue', edgecolor='black')
        plt.title(f"{name} Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(class_labels)
        # Add count labels on top of each bar
        for i, count in enumerate(class_counts):
            plt.text(i, count, str(count), ha='center', va='bottom')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{name.lower().replace(' ', '_')}_class_distribution.pdf")


def preprocess_data(X):
    """Preprocess ECG signals."""
    # Convert to numpy arrays and handle NaNs
    X = X.apply(lambda row: row.dropna().to_numpy(dtype='float32'), axis=1)
    X = X.values
    return X


def standardize_data(method: str, X_train, X_test=None):
    """
    Standardize the data using RobustScaler (which is less sensitive to outliers), StandardScaler
    QuantileTransformer, or MinMaxScaler.
    """
    if method == 'standard':
        scaler = preprocessing.StandardScaler()
    elif method == 'robust':
        scaler = preprocessing.RobustScaler()
    elif method == 'quantile': 
        scaler = preprocessing.QuantileTransformer(output_distribution='normal', n_quantiles=100)
    elif method == 'minmax':
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    else:
        raise ValueError(f"Unknown method: '{method}'.")
    
    print(f"Standardizing data using {method} scaler.")
    scaler.fit(X_train)
    X_scaled = scaler.transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_scaled, X_test_scaled
    return X_scaled, None


def extract_features(X, parallel=True):
    """Extract features from ECG signals."""
    if parallel:
        n_cores = cpu_count()
        print(f"Using {n_cores} cores for parallel processing with joblib")
        
        signal_data = list(enumerate(X))
        results = Parallel(n_jobs=n_cores)(
            delayed(extract_features_single_signal)(data)
            for data in tqdm(signal_data, desc="Extracting Features")
        )
        # Sort by index, so we can pair with the correct label
        results.sort(key=lambda x: x[0]) 
        features = [list(statistics.values()) for idx, statistics in results]
        features_array = np.array(features)
        return features_array
    all_features = []
    for signal in tqdm(X):
        idx, statistics = extract_features_single_signal((0, signal))
        all_features.append(list(statistics.values()))
    features_array = np.array(all_features)
    return features_array


def extract_features_single_signal(signal_data):
    """Extract features from a single ECG signal.
    
    Parameters:
        signal_data: Tuple of (index, signal)
    
    Returns:
        Tuple of (index, features) or None if extraction fails
    """
    idx, signal = signal_data
    features = {
        'PR_interval': [], 'QRS_duration': [], 'QT_interval': [], 'ST_segment': [], 
        'P_amplitude': [], 'Q_amplitude': [], 'R_amplitude': [], 'S_amplitude': [], 'T_amplitude': [],
        'RR_interval': [], 'Heart_rate': []
    }
        
    # R-peaks
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(signal=signal, sampling_rate=SAMPLING_RATE, show=False)
    # Get the interesting points
    waves, signals = nk.ecg_delineate(ecg_cleaned=filtered, rpeaks=rpeaks, sampling_rate=SAMPLING_RATE, show=False)
    points = {
        'P': signals['ECG_P_Peaks'],
        'Q': signals['ECG_Q_Peaks'],
        'R': rpeaks,
        'S': signals['ECG_S_Peaks'],
        'T': signals['ECG_T_Peaks']
    }
    
    # RR intervals
    rr_intervals = np.diff(rpeaks) / SAMPLING_RATE
    features['RR_interval'] = rr_intervals
    features['Heart_rate'] = heart_rate
    
    # Iterate through each template and extract features
    for i in range(len(rpeaks)):
        p = points['P'][i] if not np.isnan(points['P'][i]) else None
        q = points['Q'][i] if not np.isnan(points['Q'][i]) else None
        r = rpeaks[i]
        s = points['S'][i] if not np.isnan(points['S'][i]) else None
        t = points['T'][i] if not np.isnan(points['T'][i]) else None
        
        # assert p < q < r < s < t    # Should always hold?
        assert r is not None
        
        features['R_amplitude'].append(filtered[r])
        
        if p is not None:
            features['PR_interval'].append((r - p) / SAMPLING_RATE)
            features['P_amplitude'].append(filtered[int(p)])
            
        if q is not None:
            features['Q_amplitude'].append(filtered[int(q)])
            
        if s is not None:
            features['S_amplitude'].append(filtered[int(s)])
            
        if q is not None and s is not None:
            features['QRS_duration'].append((s - q) / SAMPLING_RATE)

        if t is not None:
            features['T_amplitude'].append(filtered[int(t)])
            
        if q is not None and t is not None:
            features['QT_interval'].append((t - r) / SAMPLING_RATE)
            
        if s is not None and t is not None:
            features['ST_segment'].append((t - s) / SAMPLING_RATE)
        
    statistics = {}
    for feature_name, values in features.items():
        statistics[f'{feature_name}_mean'] = np.mean(values) if len(values) > 0 else np.nan
        statistics[f'{feature_name}_std'] = np.std(values) if len(values) > 0 else np.nan
        statistics[f'{feature_name}_min'] = np.min(values) if len(values) > 0 else np.nan
        statistics[f'{feature_name}_max'] = np.max(values) if len(values) > 0 else np.nan
        statistics[f'{feature_name}_median'] = np.median(values) if len(values) > 0 else np.nan
        statistics[f'{feature_name}_q25'] = np.percentile(values, 25) if len(values) > 0 else np.nan
        statistics[f'{feature_name}_q75'] = np.percentile(values, 75) if len(values) > 0 else np.nan
        statistics[f'{feature_name}_ptp'] = np.ptp(values) if len(values) > 0 else np.nan # Peak-to-peak
        statistics[f'{feature_name}_entropy'] = entropy(values) if len(values) > 0 else np.nan
    
    return idx, statistics


def entropy(signal):
    """Calculate Shannon entropy of a signal."""
    # Normalize signal to probabilities
    signal = np.abs(signal)
    signal = signal / np.sum(signal)
    # Remove zeros to avoid log(0)
    signal = signal[signal > 0]
    return -np.sum(signal * np.log2(signal))


def fill_missing_values(X_train, strategy: str, X_test=None):
    """
    Fill missing values in the datasets according to the strategy provided.
    Strategies available: 'mean', 'median', 'knn', 'most_frequent'.
    Reference: https://scikit-learn.org/stable/modules/impute.html
    """    
    if strategy in ['mean', 'median', 'most_frequent']:
        imputer = impute.SimpleImputer(strategy=strategy)
    elif strategy == 'knn':
        imputer = impute.KNNImputer()
    elif strategy == 'iterative':
        from sklearn.experimental import enable_iterative_imputer # Required to import 
        imputer = impute.IterativeImputer(initial_strategy='median')
    else:
        raise ValueError(F"The strategy {strategy} is invalid.")
    
    print(f"Imputing missing values using {strategy} strategy.")
    imputer.fit(X_train)
    X_train_imputed = imputer.transform(X_train)
    
    # If X_test is provided, impute missing values in X_test
    if X_test is not None:
        X_test_imputed = imputer.transform(X_test)
        return X_train_imputed, X_test_imputed  # use the same statistics as training set

    return X_train_imputed, None


def evaluate_predictions(y_true, y_pred, model_name):
    """Evaluate predictions with multiple metrics."""
    print_class_distribution(f"Predicted ({model_name})", y_pred, plot=True)
    
    print("\nMetrics per class:")
    for class_label in sorted(set(y_true)):
        precision = metrics.precision_score(y_true, y_pred, labels=[class_label], average=None)[0]
        recall = metrics.recall_score(y_true, y_pred, labels=[class_label], average=None)[0]
        f1 = metrics.f1_score(y_true, y_pred, labels=[class_label], average=None)[0]
        print(f"\nClass {class_label}:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
    
    # Overall metrics
    print("\nOverall metrics:")
    print(f"Micro-averaged F1: {metrics.f1_score(y_true, y_pred, average='micro'):.4f}")
    print(f"Macro-averaged F1: {metrics.f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1: {metrics.f1_score(y_true, y_pred, average='weighted'):.4f}")
    
    # Confusion matrix
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"plots/{model_name.lower().replace(' ', '_')}_confusion_matrix.pdf")
    
    
def create_final_submission(y_pred, filename):
    """Create the final submission file."""
    submission = pd.DataFrame({'id': np.arange(len(y_pred)), 'y': y_pred})
    print("Shape of submission:", submission.shape)
    submission.to_csv(filename, index=False)
    print(f"Final submission file '{filename}' created.")

if __name__ == "__main__":
    main()