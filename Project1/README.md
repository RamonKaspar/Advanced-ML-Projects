# Task 1: Predict the age from a brain from MRI features

This task is primarily concerned with regression. However, we have perturbed the original MRI features in several ways.

You will have to perform the following preprocessing steps:

1. outlier detection
2. feature selection
3. imputation of missing values

You are required to document each of the three steps in the description that you will submit with your project. Besides the data processing steps, you have to provide a succinct description of the regression model you used.

## Evaluation

The evaluation metric for this task is the Coefficient of Determination $R^2$ Score which ranges from minus infinity to 1.

Given true values $y_i$, and predicted values $\hat{y_i}$, the formula reads:

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y_i})^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

where $\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i$.

## Submission File

For each signal id in the test set, submission files should contain two columns (no additional index column): id and y, where y should be the predicted brain age. The file should contain a header with columns names.

## Baseline score

For passing this project (4.0), a R2 score of 0.5 is required.

## Data

The data for this task contains the following files:

- **X_train.csv** - the training features
- **y_train.csv** - the training targets
- **X_test.csv** - the test features (you need to make predictions for these samples)
- **sample.csv** - a sample submission file in the correct format

Each row in X_train.csv is one sample indexed by an id, so the first column contains the id. In addition to the id column, each sample has 831 features:

```txt
id,x0,x1,...,x831
0,10.8,832442.8,...,103088.6
...
```

The training targets (age in years) are contained in y_train.csv:

```txt
id,y
0,71
1,73
2,66
...
```

Note that, for each prediction you need to include the id of the sample as in X_test.csv.
