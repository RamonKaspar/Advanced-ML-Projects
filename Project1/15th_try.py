import numpy as np
import pandas as pd
from sklearn import ensemble, impute, linear_model, model_selection, svm, feature_selection, decomposition, preprocessing, neighbors, metrics

# from sklearnex import patch_sklearn
# patch_sklearn()

RANDOM_STATE = 69   # For reproducibility

def main():
    X_train = pd.read_csv('data/X_train.csv').drop(columns=['id'])
    y_train = pd.read_csv('data/y_train.csv').drop(columns=['id'])
    X_test = pd.read_csv('data/X_test.csv').drop(columns=['id'])
    y_train = y_train.values.ravel()    # we need a 1D array
    print("Shape Matrices (Input)")
    print("X_train:", X_train.shape, "y_train:", y_train.shape, "X_test:", X_test.shape)

    # Create train/test split (80/20) 
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE) 
    # print("Shape Matrices (Input)")
    # print("X_train:", X_train.shape, "y_train:", y_train.shape, "X_test:", X_test.shape)
        
    # Preprocess data: Remove zero columns and duplicate rows/columns
    # X_train, X_test = preprocess_data(X_train, X_test, threshold_nan=0.09, threshold_zero=0.2)
    X_train, X_test = X_train.values, X_test.values  # Convert to numpy arrays
    
    # == OUTLIER DETECTION ==
    print("\n== Outlier Detection ==")
    # Standardize data, use RobustScalar as outliers might be present
    X_train_scaled, _ = standardize_data(X_train, method='robust')
    # Impute missing values (use median as it is less sensitive to outliers)
    X_train_imputed, _ = fill_missing_values(X_train_scaled, strategy='knn')
    # Remove outliers
    inliers_mask = outlier_detection(X_train_imputed, y_train, method=["IsolationForest"], perform_pca=True, pca_variance=0.2)
    
    # Remove all indices from X_train that are not in X_train_without_outliers
    X_train_without_outliers = X_train[inliers_mask]
    y_train_without_outliers = y_train[inliers_mask]
    
    # Preprocess data again, now using StandardScaler (as outliers are removed) and impute missing values
    print("\n== Preprocessing ==")
    X_train_without_outliers, X_test = standardize_data(X_train_without_outliers, method='standard', X_test=X_test)
    X_train_final, X_test_final = fill_missing_values(X_train_without_outliers, strategy='knn', X_test=X_test)
    
    # == FEATURE SELECTION ==
    print("\n== Feature Selection ==")
    X_train_selected, X_test_selected = select_features(X_train_final, y_train_without_outliers, X_test_final, method=['MutualInfo', 'CorrelationRemover'], variance_threshold=None, k=600, alpha=None, corr_threshold=0.97, target_corr_threshold=0.09)
    
    print("Shape Matrices after Outlier Detection, Preprocessing, and Feature Selection:")
    print("X_train:", X_train_selected.shape, "y_train:", y_train_without_outliers.shape, "X_test:", X_test_selected.shape)

    # == FINAL REGRESSION MODEL ==
    print("\n== Final Regression Model ==")
    
    # Find the best SVR model using RandomizedSearchCV
    # print("Finding the best SVR model using GridSearchCV...")
    # svr = model_selection.GridSearchCV(
    #     svm.SVR(),
    #     param_grid={
    #         'C': np.logspace(-1, 3, 100),
    #         'gamma': ['auto', 'scale'],
    #         'epsilon': np.logspace(-6, 0, 7),
    #         'kernel': ['rbf'],
    #     },
    #     n_jobs=-1,
    #     scoring='r2'    # r2 might not be the best metric for CV?
    # )
    # svr.fit(X_train_selected, y_train_without_outliers)
    # print(f"\nBest SVR Model: {svr.best_params_} (R^2: {svr.best_score_:.4f})")
    # svr = svr.best_estimator_
    # y_pred_val = svr.predict(X_test_selected)
    # print(f"SVR: R^2 Score Validation Set: {metrics.r2_score(y_test, y_pred_val):.4f}")
    
    # Best SVR Model: {'C': 55.90810182512229, 'epsilon': 0.0001, 'gamma': 'auto', 'kernel': 'rbf'}
    # Found using above GridSearchCV
    svr = svm.SVR(C=55.90810182512229, epsilon=0.0001, gamma='auto', kernel='rbf')  # 0.6356
    
    # Find the best KNN model using RandomizedSearchCV
    # print("Finding the best KNN model using GridSearchCV...")
    # knn = model_selection.GridSearchCV(
    #     neighbors.KNeighborsRegressor(),
    #     param_grid={
    #         'n_neighbors': np.arange(1, 50, 1),
    #         'weights': ['distance'],
    #         'algorithm': ['auto'],
    #         'p': [1, 2, 3, 4, 5],
    #         'metric': ['manhattan'],
    #         'leaf_size': np.arange(1, 100, 1),
    #         'algorithm': ['kd_tree'],
    #         'n_jobs': [-1]
    #     },
    #     n_jobs=-1,
    #     scoring='r2',
    # )
    # knn.fit(X_train_selected, y_train_without_outliers)
    # print(f"\nBest KNN Model: {knn.best_params_} (R^2: {knn.best_score_:.4f})")
    # knn = knn.best_estimator_
    # y_pred_val = knn.predict(X_test_selected)
    # print(f"KNN: R^2 Score Validation Set: {metrics.r2_score(y_test, y_pred_val):.4f}")
    
    # Best KKN Model: {'algorithm': 'auto', 'leaf_size': 53, 'metric': 'manhattan', 'n_jobs': -1, 'n_neighbors': 3, 'p': 2, 'weights': 'distance'}
    # Found using above GridSearchCV
    knn = neighbors.KNeighborsRegressor(n_neighbors=3, weights='distance', p=2, metric='manhattan', leaf_size=53, algorithm='kd_tree') # 0.5326

    print("Fitting StackingRegressor...")
    model = ensemble.StackingRegressor(
        estimators=[
            ('svr', svr),
            ('knn', knn),
        ],
        final_estimator=svm.SVR(C=30, kernel='linear', gamma='scale', epsilon=0.1), # 0.6464
        # Use SVR as final estimator, find best C using GridSearchCV
        n_jobs=-1,
    ) 
    
    # Cross-validated R^2 score (using 5 folds)
    score = model_selection.cross_val_score(model, X_train_selected, y_train_without_outliers, cv=5, scoring='r2')
    print(f"Cross-validated R^2: {score.mean():.4f} (+/- {score.std() * 2:.4f} std)")
    
    # Calculate out-of-sample R^2 score (using Validation Set)
    # model.fit(X_train_selected, y_train_without_outliers)
    # y_pred_val = model.predict(X_test_selected)
    # out_of_sample_r2 = metrics.r2_score(y_test, y_pred_val)
    # print(f"Out-of-sample R^2 Score: {out_of_sample_r2:.4f}")

    # Create the final submission
    create_submission(model, X_train_selected, y_train_without_outliers, X_test_selected)

    
def preprocess_data(X_train, X_test, threshold_nan: float, threshold_zero: float):
    """ 
    Remove zero columns and duplicates from the training data.
    Furthermore, drop columns with more than threshold_nan NaN values.
    """    
    # Remove zero columns (where all non-NaN values are zero)
    zero_columns = [col for col in X_train.columns if (X_train[col].fillna(0) == 0).all()]
    X_train = X_train.drop(columns=zero_columns)
    X_test = X_test.drop(columns=zero_columns)
    print(f"Removed {len(zero_columns)} zero columns.")
    
    # Remove columns with more than threshold_zero zero values
    abs_threshold = len(X_train) * threshold_zero
    columns_to_drop = X_train.columns[(X_train == 0).sum() > abs_threshold]
    X_train = X_train.drop(columns=columns_to_drop)
    X_test = X_test.drop(columns=columns_to_drop)
    print(f"Removed {len(columns_to_drop)} columns with more than {threshold_zero * 100}% zero values.")
    
    # Remove columns with more than threshold_nan NaN values
    abs_threshold = len(X_train) * threshold_nan
    columns_to_drop = X_train.columns[X_train.isna().sum() > abs_threshold]
    X_train = X_train.drop(columns=columns_to_drop)
    X_test = X_test.drop(columns=columns_to_drop)
    print(f"Removed {len(columns_to_drop)} columns with more than {threshold_nan * 100}% NaN values.")
    
    # Remove duplicated columns
    duplicate_columns = X_train.columns[X_train.T.duplicated()]
    X_train = X_train.loc[:, ~X_train.T.duplicated()]   
    X_test = X_test.loc[:, ~X_test.T.duplicated()]
    print(f"Removed {len(duplicate_columns)} duplicate columns.")
    
    # Remove duplicate rows
    duplicated_rows = X_train[X_train.duplicated()]
    X_train = X_train.loc[~X_train.duplicated(), :]
    print(f"Removed {len(duplicated_rows)} duplicate rows.")
    
    return X_train, X_test

def standardize_data(X_train, method: str, X_test = None):
    """
    Standardize the data using RobustScaler (which is less sensitive to outliers) or StandardScaler.
    """
    if method == 'standard':
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

def outlier_detection(X_train, y_train, method: list[str], perform_pca: bool = False, pca_variance: float = 0.8):
    """
    We detect and remove outliers from the training data. If PCA is enabled, we first reduce the dimensionality
    such that the lower dim representation captures pca_variance of variance (default to 80%) before applying 
    the outlier detection method.
    Methods available: 'OneClassSVM', 'IsolationForest', 'LocalOutlierFactor'.
    Reference: https://scikit-learn.org/1.5/modules/outlier_detection.html and
    https://towardsdatascience.com/a-brief-overview-of-outlier-detection-techniques-1e0b2c19e561
    """
    if perform_pca:
        # Data already standardized, so we can use PCA
        pca = decomposition.PCA(n_components=pca_variance)
        X_train = pca.fit_transform(X_train)
        print(f"PCA: X_train reduced to {X_train.shape[1]} components.")
        
    # Keep track of the combined predictions from all outlier detection methods
    combined_predictions = np.ones(X_train.shape[0])
    
    if 'OneClassSVM' in method:
        outlier_det = svm.OneClassSVM(nu=0.025, gamma="scale")
        predictions = outlier_det.fit_predict(X_train, y_train)
        print(f"Removed {np.sum(predictions == -1)} outliers using OneClassSVM.")
        combined_predictions = np.logical_and(combined_predictions, predictions == 1)
    
    if 'IsolationForest' in method:
        outlier_det = ensemble.IsolationForest(random_state=RANDOM_STATE, contamination=0.05)   # Set random state for reproducibility
        predictions = outlier_det.fit_predict(X_train, y_train)
        print(f"Removed {np.sum(predictions == -1)} outliers using IsolationForest.")
        combined_predictions = np.logical_and(combined_predictions, predictions == 1)
    
    if 'LocalOutlierFactor' in method:
        outlier_det = neighbors.LocalOutlierFactor()
        predictions = outlier_det.fit_predict(X_train, y_train)
        print(f"Removed {np.sum(predictions == -1)} outliers using LocalOutlierFactor.")
        combined_predictions = np.logical_and(combined_predictions, predictions == 1)
    
    # Filter out the outliers
    inliers_mask = combined_predictions
    print(f"Removed {np.sum(~inliers_mask)} outliers using {method} (PCA {'enabled' if perform_pca else 'disabled'}).")
        
    return inliers_mask


def select_features(X_train, y_train, X_test, method: list[str], variance_threshold: float = 0.0, k: int = 200, alpha: float = 0.01, corr_threshold: float = 0.98, target_corr_threshold: float = 0.1):
    """
    Perform feature selection on the training data.
    Available methods: 'VarianceThreshold', 'SelectKBest', 'Lasso', 'CorrelationRemover', 'MutualInfo'
    Reference: https://scikit-learn.org/stable/modules/feature_selection.html
    
    Parameters:
        variance_threshold (float): Threshold below which features are removed (VarianceThreshold).
        k (int): Number of top features to select (SelectKBest).
        alpha (float): Regularization parameter (for Lasso).
        corr_threshold (float): Threshold for correlation between features.
        target_corr_threshold (float): Threshold for correlation with target variable.
    """
    num_features = X_train.shape[1]
    
    if 'VarianceThreshold' in method:
        selector = feature_selection.VarianceThreshold(threshold=variance_threshold)
        X_train = selector.fit_transform(X_train)
        X_test = selector.transform(X_test)
        print(f"Selected features after 'VarianceThreshold': {X_train.shape[1]}/{num_features}")
    
    if 'SelectKBest' in method:
        selector = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=k)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        print(f"Selected features after 'SelectKBest': {X_train.shape[1]}/{num_features}")
    
    if 'Lasso' in method:
        # lasso = linear_model.LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, random_state=RANDOM_STATE)
        lasso = linear_model.Lasso(alpha=alpha, random_state=RANDOM_STATE)
        selector = feature_selection.SelectFromModel(lasso)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        print(f"Selected features after 'Lasso': {X_train.shape[1]}/{num_features}")
        
    if 'MutualInfo' in method:
        # Calculate mutual information scores
        mi_scores = feature_selection.mutual_info_regression(X_train, y_train, n_neighbors=5, random_state=RANDOM_STATE)
        # Select top k features based on mutual information scores
        top_k_indices = np.argsort(mi_scores)[-k:]  # get indices of k highest scores
        X_train = X_train[:, top_k_indices]
        X_test = X_test[:, top_k_indices]
        print(f"Selected features after 'MutualInfo': {X_train.shape[1]}/{num_features}")
        
    if 'CorrelationRemover' in method:
        def correlation_remover(X, threshold):
            """Helper function to remove correlated features."""
            corr_matrix = X.corr().abs()
            # Select the upper triangle of the correlation matrix
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # Find features with correlation greater than the threshold
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] >= threshold)]
            # Drop the features
            reduced_dataset = X.drop(columns=to_drop)
            return reduced_dataset, set(to_drop)
        
        # Combine correlation and mutual information scores
        corr_y = []
        mi_scores = feature_selection.mutual_info_regression(X_train, y_train)
        
        for idx in range(X_train.shape[1]):
            # Calculate Pearson correlation
            corr = np.corrcoef(X_train[:, idx], y_train)[0, 1]
            corr_y.append((idx, corr))
        # Select features with correlation above target_corr_threshold
        # Because we want features that have a significant correlation with the target variable (i.e., age).
        high_corr_idx = [idx for idx, corr in corr_y if abs(corr) >= target_corr_threshold]
        print(f"Selected {len(high_corr_idx)} features with correlation to target (i.e., age) above {target_corr_threshold}.")
        X_train = X_train[:, high_corr_idx]
        X_test = X_test[:, high_corr_idx]
        # Remove correlated features among the selected features
        # Because we want to reduce redundancy and multicollinearity
        X_train_df = pd.DataFrame(X_train)
        X_train_df, removed_cols = correlation_remover(X_train_df, corr_threshold)
        X_train = X_train_df.values
        X_test_df = pd.DataFrame(X_test)
        X_test_df = X_test_df.drop(columns=removed_cols)
        X_test = X_test_df.values
        print(f"Removed {len(removed_cols)} features due to high correlation to each other (threshold={corr_threshold}).")
        num_features = X_train.shape[1]
    
    return X_train, X_test


def create_submission(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_ids = pd.read_csv('data/X_test.csv')['id'] # Use same IDs as in X_test
    Y_pred_final = pd.DataFrame({
        'id': test_ids,
        'age': y_pred
    })
    Y_pred_final.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")


if __name__ == "__main__":
    main()