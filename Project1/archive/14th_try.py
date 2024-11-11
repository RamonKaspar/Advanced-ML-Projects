import numpy as np
import pandas as pd
from sklearn import ensemble, impute, linear_model, model_selection, svm, feature_selection, decomposition, preprocessing, neighbors, metrics
import os 

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
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE) 
    print("Shape Matrices (Input)")
    print("X_train:", X_train.shape, "y_train:", y_train.shape, "X_test:", X_test.shape)
    
    X_test = pd.read_csv('data/X_test.csv').drop(columns=['id'])
        
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
    X_train_selected, X_test_selected = select_features(X_train_final, y_train_without_outliers, X_test_final, method=['CorrelationRemover'], variance_threshold=None, k=None, alpha=None) # alpha found using LassoCV
    
    print("Shape Matrices after Outlier Detection, Preprocessing, and Feature Selection:")
    print("X_train:", X_train_selected.shape, "y_train:", y_train_without_outliers.shape, "X_test:", X_test_selected.shape)

    # == FINAL REGRESSION MODEL ==
    print("\n== Training Neural Network ==")
    
    X_train_tensor = torch.from_numpy(X_train_selected).float()
    y_train_tensor = torch.from_numpy(y_train_without_outliers).float().unsqueeze(1)
    X_test_tensor = torch.from_numpy(X_test_selected).float()
    y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)
    input_size = X_train_selected.shape[1]

    num_epochs = 1000
    batch_size = 32
    patience = 200  # Number of epochs with no improvement after which training will be stopped
    best_val_r2 = float('-inf')    # Keep track of the best validation loss
    epochs_no_improve = 0
    
    num_epochs = 1000
    batch_size = 32
    patience = 200  # Number of epochs with no improvement after which training will be stopped
    best_val_r2 = float('-inf')    # Keep track of the best validation loss
    epochs_no_improve = 0
    
    # for i in range(100):
    #     print(f"== Training Neural Network {i+1} ==")
    #     try:
    #         if os.path.exists('best_model.pth'):
    #             os.remove('best_model.pth')
    #         out_of_sample_r2 = train_NN(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_size)
    #         print(f"Out-of-sample R^2 Score: {out_of_sample_r2:.4f}")
    #         if out_of_sample_r2 > 0.63:
    #             print("STOP: R^2 Score is above 0.63, namely", out_of_sample_r2)
    #             break
    #     except Exception as e:
    #         print("Error:", e)
    #         continue
        
        
    #     if (epoch + 1) % 10 == 0:
    #         print(f'Epoch [{epoch + 1}/{num_epochs}]: Training Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation R^2: {val_r2:.4f}')
            
    #     if (epoch + 1) % 10 == 0:
    #         print(f'Epoch [{epoch + 1}/{num_epochs}]: Training Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation R^2: {val_r2:.4f}')


    # # We load the best model (based on validation loss)
    # model.load_state_dict(torch.load('best_model.pth'))

    # # Evaluation on test set
    # with torch.no_grad():
    #     model.eval()
    #     y_pred_val = model(X_test_tensor)
    #     y_pred_val_np = y_pred_val.numpy()
    #     y_test_np = y_test_tensor.numpy()
    #     out_of_sample_r2 = metrics.r2_score(y_test_np, y_pred_val_np)
    #     print(f"Out-of-sample R^2 Score: {out_of_sample_r2:.4f}")

    
    # # We load the best model (based on validation loss)
    # model.load_state_dict(torch.load('best_model.pth'))

    # # Evaluation on test set
    # with torch.no_grad():
    #     model.eval()
    #     y_pred_val = model(X_test_tensor)
    #     y_pred_val_np = y_pred_val.numpy()
    #     y_test_np = y_test_tensor.numpy()
    #     out_of_sample_r2 = metrics.r2_score(y_test_np, y_pred_val_np)
    #     print(f"Out-of-sample R^2 Score: {out_of_sample_r2:.4f}")

    # Create the final submission
    model = NeuralNetwork(input_size)
    model.load_state_dict(torch.load('best_model.pth'))
    with torch.no_grad():
        model.eval()
        y_pred = model(X_test_tensor)
        test_ids = pd.read_csv('data/X_test.csv')['id'] # Use same IDs as in X_test
        Y_pred_final = pd.DataFrame({
            'id': test_ids,
            'y': y_pred.numpy().flatten()
        })
        Y_pred_final.to_csv('submission.csv', index=False)
        print("Predictions saved to submission.csv")

    
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
        outlier_det = ensemble.IsolationForest(random_state=RANDOM_STATE, contamination=0.045)   # Set random state for reproducibility
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
    Available methods: 'VarianceThreshold', 'SelectKBest', 'Lasso'
    Reference: https://scikit-learn.org/stable/modules/feature_selection.html
    
    Parameters:
        variance_threshold (float): Threshold below which features are removed (VarianceThreshold).
        k (int): Number of top features to select (SelectKBest).
        alpha (float): Regularization parameter (for Lasso).
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
        
        # Compute correlation between each feature and the target variable
        corr_y = []
        for idx in range(X_train.shape[1]):
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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.model(x)


def train_NN(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_size):
    model = NeuralNetwork(input_size)
    criterion = nn.MSELoss()
    # NOTE: Add weight decay to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # TODO: Try different learning rate
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # Overfits too fast

    num_epochs = 3000
    batch_size = 32
    patience = 400  # Number of epochs with no improvement after which training will be stopped
    best_val_r2 = float('-inf')    # Keep track of the best validation loss
    epochs_no_improve = 0

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Compute validation loss
        with torch.no_grad():
            model.eval()
            val_outputs = model(X_test_tensor)
            y_pred_val_np = val_outputs.numpy().flatten()
            y_test_np = y_test_tensor.numpy().flatten()
            val_r2 = metrics.r2_score(y_test_np, y_pred_val_np)
            val_loss = criterion(val_outputs, y_test_tensor).item()
        
        # If validation loss is lower than the current best ==> Save the model
        # Add small tolerance to avoid small fluctuations
        if val_r2 > best_val_r2 + 1e-4:
            epochs_no_improve = 0
            # Make sure that model generalizes well ==> Training and Validation MSE should be close
            if abs(val_loss - (epoch_loss / len(train_loader))) < 5.0:
                if val_r2 > 0.6: print(f"Validation R^2 improved from {best_val_r2:.4f} to {val_r2:.4f} at epoch {epoch}. Saving model.")
                best_val_r2 = val_r2
                if os.path.exists('best_model.pth'):
                    os.remove('best_model.pth')
                torch.save(model.state_dict(), 'best_model.pth')
        # If validation loss is higher, increase the counter for early stopping
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('Stopping early at epoch', epoch)
                break
            
        if abs(val_loss - (epoch_loss / len(train_loader))) > 20.0 and epoch > 1000:
            # We are overfitting, stop training
            print('Stopping early due to overfitting at epoch', epoch, ". MSE difference:", abs(val_loss - (epoch_loss / len(train_loader))))
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]: Training Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation R^2: {val_r2:.4f}')

    # We load the best model (based on validation loss)
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluation on test set
    with torch.no_grad():
        model.eval()
        y_pred_val = model(X_test_tensor)
        y_pred_val_np = y_pred_val.numpy()
        y_test_np = y_test_tensor.numpy()
        out_of_sample_r2 = metrics.r2_score(y_test_np, y_pred_val_np)
        print(f"Out-of-sample R^2 Score: {out_of_sample_r2:.4f}")
    
    return out_of_sample_r2

if __name__ == "__main__":
    main()