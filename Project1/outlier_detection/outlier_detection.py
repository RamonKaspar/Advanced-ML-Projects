import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import ensemble, impute, svm, decomposition, preprocessing, neighbors, manifold

# Speed-up for Intel(R) CPUs
from sklearnex import patch_sklearn
patch_sklearn()

# For reproducibility
RANDOM_STATE = 69   

# Keep track of all methods and their results
results = {}

def main():
    X_train = pd.read_csv('data/X_train.csv').drop(columns=['id'])
    y_train = pd.read_csv('data/y_train.csv').drop(columns=['id'])
    X_test = pd.read_csv('data/X_test.csv').drop(columns=['id'])
    y_train = y_train.values.ravel()    # we need a 1D array
    print("Shape Matrices (Input)")
    print("X_train:", X_train.shape, "y_train:", y_train.shape, "X_test:", X_test.shape)
    
    output_dir = 'plots/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot Train/Test data for each imputing and scaling method
    imputation_techniques = ['mean', 'median', 'knn', 'most_frequent']  # TODO: Maybe consider 'iterative'?
    standardize_techniques = ['standard', 'robust', 'quantile', None]
    
    # # Plot PCA for each imputing and scaling method
    fig, axs = plt.subplots(len(imputation_techniques), len(standardize_techniques), figsize=(20, 20))
    for i, impute_method in enumerate(imputation_techniques):
        for j, standardize_method in enumerate(standardize_techniques):
            X_train_imputed, X_test_imputed = fill_missing_values(X_train, impute_method, X_test)
            X_train_scaled, X_test_scaled = standardize_data(X_train_imputed, standardize_method, X_test_imputed)
            plot_pca(axs[i, j], X_train_scaled, X_test_scaled, f"PCA (Impute: {impute_method}, Scale: {standardize_method})")
    plt.savefig(f'{output_dir}impute_scale_pca.pdf')
    
    # == OUTLIER DETECTION ==
    outlier_detection_techniques = ['OneClassSVM', 'IsolationForest', 'LocalOutlierFactor']
    
    for outlier_det_method in outlier_detection_techniques:
        for pca_enabled in [True, False]:
            fig, axs = plt.subplots(len(imputation_techniques), len(standardize_techniques), figsize=(20, 20))
            for i, impute_method in enumerate(imputation_techniques):
                for j, standardize_method in enumerate(standardize_techniques):
                    technique_name = f"Impute: {impute_method}, Scale: {standardize_method}, Outliers Detection: {outlier_det_method} (PCA {'Enabled' if pca_enabled else 'Disabled'})"
                    X_train_imputed, X_test_imputed = fill_missing_values(X_train, impute_method, X_test)
                    X_train_scaled, X_test_scaled = standardize_data(X_train_imputed, standardize_method, X_test_imputed)
                    # Perform outlier detection on PCA-transformed data
                    inliers_mask, outlier_detector = outlier_detection(X_train_scaled, outlier_det_method, pca_enabled)
                    # Plot PCA with outliers marked and decision boundary
                    plot_pca(axs[i, j], X_train_scaled, X_test_scaled, f"Impute: {impute_method}\nScale: {standardize_method}", inliers_mask=inliers_mask, outlier_detector=outlier_detector)
                    # Describe and analyze the distribution of the data, with and without outliers
                    describe_and_analyze_distribution(X_train_scaled, X_test_scaled, f"Impute: {impute_method}, Scale: {standardize_method}, Outliers Detection: None")
                    describe_and_analyze_distribution(X_train_scaled[inliers_mask], X_test_scaled, technique_name)
            # Set the overall figure title
            if pca_enabled:
                fig.suptitle(f'Outlier Detection: {outlier_det_method} (PCA Enabled)')
                plt.savefig(f'{output_dir}impute_scale_pca_outliers_{outlier_det_method}_with_pca.pdf')
            else:
                fig.suptitle(f'Outlier Detection: {outlier_det_method}')
                plt.savefig(f'{output_dir}impute_scale_pca_outliers_{outlier_det_method}_without_pca.pdf')
    
    # Find the best methods for imputing and scaling the data
    find_best_methods(n=5)
    
    # One of the best methods is (Impute: most_frequent, Scale: quantile, Outliers Detection: IsolationForest)
    X_train_imputed, X_test_imputed = fill_missing_values(X_train, 'most_frequent', X_test)
    X_train_scaled, X_test_scaled = standardize_data(X_train_imputed, 'quantile', X_test_imputed)
    inliers_mask, outlier_detector = outlier_detection(X_train_scaled, 'IsolationForest', pca_enabled=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_pca(ax, X_train_scaled, X_test_scaled, "IsolationForest (Impute: most_frequent, Scale: quantile)", inliers_mask=inliers_mask, outlier_detector=outlier_detector)
    plt.savefig(f'{output_dir}best_method.pdf')
    

def standardize_data(X_train, method: str, X_test = None):
    """
    Standardize the data using RobustScaler (which is less sensitive to outliers) or StandardScaler.
    """
    if method is None:
        return X_train, X_test
    elif method == 'standard':
        scaler = preprocessing.StandardScaler()
    elif method == 'robust':
        scaler = preprocessing.RobustScaler()
    elif method == 'quantile': 
        scaler = preprocessing.QuantileTransformer(output_distribution='normal', n_quantiles=100)
    else:
        raise ValueError(f"Unknown method: '{method}'.")
    
    print(f"Standardizing data using {method} scaler.")
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    return X_train_scaled, None


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

def outlier_detection(X_train, method: str, pca_enabled: bool):
    """
    Detect outliers using the specified method.
    Methods available: 'OneClassSVM', 'IsolationForest', 'LocalOutlierFactor'.
    """
    if pca_enabled:
        pca = decomposition.PCA(n_components=2, random_state=RANDOM_STATE)
        X_train = pca.fit_transform(X_train)
        
    if method == 'OneClassSVM':
        outlier_det = svm.OneClassSVM(nu=0.05, gamma='scale')
    elif method == 'IsolationForest':
        outlier_det = ensemble.IsolationForest(random_state=RANDOM_STATE, contamination=0.05)
    elif method == 'LocalOutlierFactor':
        outlier_det = neighbors.LocalOutlierFactor(novelty=True)
    else:
        raise ValueError(f"Unknown method: '{method}'")

    print(f"Detecting outliers using {method}.")
    outlier_det.fit(X_train)
    predictions = outlier_det.predict(X_train)
    inliers_mask = predictions == 1
    print(f"Detected {np.sum(~inliers_mask)} outliers using {method}.")

    return inliers_mask, outlier_det

def plot_pca(ax, X_train, X_test, title, inliers_mask=None, outlier_detector=None):
    """
    Plot the PCA components for the training and test data on the given axes.
    If inliers_mask is provided, mark inliers and outliers differently.
    If outlier_detector is provided, plot the decision boundary.
    """
    # Dimensionality Reduction for Visualization
    pca = decomposition.PCA(n_components=2, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Plot inliers and outliers differently
    if inliers_mask is not None:
        inliers = X_train_pca[inliers_mask]
        outliers = X_train_pca[~inliers_mask]
        ax.scatter(inliers[:, 0], inliers[:, 1], alpha=0.5, label='Inliers', color="blue")
        ax.scatter(outliers[:, 0], outliers[:, 1], alpha=0.5, label='Outliers', color='red', marker='x')
    else:
        ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.5, label='Train', color="blue")
    
    # Plot test data
    ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], alpha=0.5, label='Test', color='orange')
    
    # Plot decision boundary if outlier_detector is provided and was trained on PCA-transformed data
    if outlier_detector is not None:
        # Check if the outlier detector was trained on data with 2 features
        if outlier_detector.n_features_in_ == 2:
            xx, yy = np.meshgrid(
                np.linspace(X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1, 100),
                np.linspace(X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1, 100)
            )
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = outlier_detector.decision_function(grid)
            Z = Z.reshape(xx.shape)
            # Plot decision boundary
            ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
            # Plot number of outliers as text in the plot
            n_outliers = np.sum(~inliers_mask)
            ax.text(0.95, 0.05, f"Outliers: {n_outliers}", transform=ax.transAxes, ha='right', va='bottom')
        else:
            # Skip plotting the decision boundary
            n_outliers = np.sum(~inliers_mask)
            ax.text(0.95, 0.05, f"Outliers: {n_outliers}", transform=ax.transAxes, ha='right', va='bottom')

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    
    return X_train_pca, X_test_pca
    
def plot_tsne(ax, X_train, X_test, title):
    """
    NOTE: DID NOT WORK WELL!
    Plot the t-SNE components for the training and test data on the given axes.
    """
    # Combine X_train and X_test
    X_combined = np.vstack((X_train, X_test))
    tsne = manifold.TSNE(n_components=2, random_state=RANDOM_STATE)
    X_combined_tsne = tsne.fit_transform(X_combined)
    # Split the transformed data back into train and test sets
    X_train_tsne = X_combined_tsne[:len(X_train)]
    X_test_tsne = X_combined_tsne[len(X_train):]

    ax.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], alpha=0.5, label='Train')
    ax.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], alpha=0.5, label='Test')
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    

def describe_and_analyze_distribution(X_train, X_test, method: str):
    """
    Describe and analyze the distribution of the data.
    Checks how similar the distributions of X_train and X_test are using various methods.
    
    Keeps tracks of all seen methods and their results.
    """
    # Keep track of all results
    comparison_results = {}
    
    # General statistics
    mean_train = np.mean(X_train, axis=0)
    mean_test = np.mean(X_test, axis=0)
    mean_diff = mean_train - mean_test
    
    var_train = np.var(X_train, axis=0)
    var_test = np.var(X_test, axis=0)
    var_diff = var_train - var_test
    
    std_train = np.std(X_train, axis=0)
    std_test = np.std(X_test, axis=0)
    std_diff = std_train - std_test
    
    median_train = np.median(X_train, axis=0)
    median_test = np.median(X_test, axis=0)
    median_diff = median_train - median_test
    
    skew_train = pd.DataFrame(X_train).skew(axis=0)
    skew_test = pd.DataFrame(X_test).skew(axis=0)
    skew_diff = skew_train - skew_test
    
    kurt_train = pd.DataFrame(X_train).kurtosis(axis=0)
    kurt_test = pd.DataFrame(X_test).kurtosis(axis=0)
    kurt_diff = kurt_train - kurt_test
    
    # Store results
    comparison_results['mean_difference'] = np.mean(np.abs(mean_diff))
    comparison_results['variance_difference'] = np.mean(np.abs(var_diff))
    comparison_results['std_difference'] = np.mean(np.abs(std_diff))
    comparison_results['median_difference'] = np.mean(np.abs(median_diff))
    comparison_results['skewness_difference'] = np.mean(np.abs(skew_diff))
    comparison_results['kurtosis_difference'] = np.mean(np.abs(kurt_diff))
    
    # Range and IQR
    range_train = np.ptp(X_train, axis=0)
    range_test = np.ptp(X_test, axis=0)
    range_diff = range_train - range_test
    comparison_results['range_difference'] = np.mean(np.abs(range_diff))
    
    q1_train = np.percentile(X_train, 25, axis=0)
    q3_train = np.percentile(X_train, 75, axis=0)
    q1_test = np.percentile(X_test, 25, axis=0)
    q3_test = np.percentile(X_test, 75, axis=0)
    iqr_train = q3_train - q1_train
    iqr_test = q3_test - q1_test
    iqr_diff = iqr_train - iqr_test
    comparison_results['iqr_difference'] = np.mean(np.abs(iqr_diff))
    
    # Coefficient of Variation (CV)
    epsilon = 1e-10  # A small constant to prevent division by zero
    cv_train = std_train / (mean_train + epsilon)
    cv_test = std_test / (mean_test + epsilon)
    cv_diff = cv_train - cv_test
    comparison_results['cv_difference'] = np.mean(np.abs(cv_diff))
    
    # Store the results for the method
    results[method] = comparison_results
    
    # TODO: Add more statistical tests to compare the distributions

def find_best_methods(n=3):
    """
    Find the top n methods for imputing and scaling the data.
    All values are "the lower the better", therefore we sum all values and
    choose the methods with the lowest sum.
    """
    sorted_methods = sorted(results.items(), key=lambda item: sum(item[1].values()))
    best_methods = sorted_methods[:n]
    
    for rank, (method, scores) in enumerate(best_methods, start=1):
        score = sum(scores.values())
        print(f"Rank {rank}: Method {method} with score {score}")

if __name__ == "__main__":
    main()