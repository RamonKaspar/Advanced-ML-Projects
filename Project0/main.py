import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

X = train_df.drop(columns=['id', 'target'])
y = train_df['target']
X_test = test_df.drop(columns=['id'])

# Models to try
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'k-NN': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=200, random_state=42)
}

accuracies = {}
for name, model in models.items():
    # Cross-validation with 5 folds
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    accuracy = cv_scores.mean()  # Average accuracy across the folds
    accuracies[name] = accuracy
    print(f"'{name}' Cross-Validation Accuracy: {accuracy:.6f}")

# Choose the best model based on cross-validation scores
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"\nBest Model is '{best_model_name}' with cross-validation accuracy: {accuracies[best_model_name]:.4f}")

# Train the best model on the full training set and predict on the test set
best_model.fit(X, y)
predictions = best_model.predict(X_test)

submission = pd.DataFrame({'id': test_df['id'], 'target': predictions})
submission.to_csv('submission.csv', index=False)