# Task 31: Features and Target Selection
**Veda Technology – AI & ML Internship**

## Objective
Select suitable input features and a target variable from a machine-learning dataset. This notebook demonstrates how to separate them for a supervised learning problem.
## 1. Import the required library
Pandas is used to load and work with the CSV dataset.
import pandas as pd

print('Pandas version:', pd.__version__)
## 2. Load the dataset
Make sure `iris.csv` is in the same folder as this notebook.
df = pd.read_csv('iris.csv')

print('Dataset loaded successfully!')
print('Shape:', df.shape)
df.head()
## 3. Inspect the dataset
Check the column names, data types, and whether any values are missing.
print('Column names:')
print(df.columns.tolist())

print('\nDataset information:')
df.info()

print('\nMissing values:')
print(df.isnull().sum())
## 4. Select input features (X)
The four flower measurements are selected as input features because they provide information that can help identify a flower's species.
feature_columns = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width'
]

X = df[feature_columns]
print('Input features (X):')
display(X.head())
## 5. Select the target variable (y)
`species` is selected as the target because it is the flower category the model would learn to predict.
y = df['species']
print('Target variable (y):')
display(y.head())

print('Target classes:')
print(y.unique())
## 6. Verify the separation of X and y
print('X shape:', X.shape)
print('y shape:', y.shape)
print('\nFeature columns:', X.columns.tolist())
print('Target column:', y.name)
## 7. Reason for selecting these columns
- **Input features (X):** `sepal_length`, `sepal_width`, `petal_length`, and `petal_width` are numerical measurements of each flower. They are useful inputs for distinguishing flower types.
- **Target variable (y):** `species` contains the category to be predicted: setosa, versicolor, or virginica.
- **Avoiding data leakage:** The target column is kept separate and is not included in X. This prevents the model from receiving the answer as an input.
## Conclusion
The Iris dataset was loaded successfully. Four numerical columns were selected as input features (X), and `species` was selected as the target variable (y). This demonstrates the basic feature-target separation used in supervised machine learning.