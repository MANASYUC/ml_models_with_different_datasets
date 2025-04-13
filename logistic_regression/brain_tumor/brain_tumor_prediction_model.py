import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load and preview data
df = pd.read_csv('logistic_regression/brain_tumor/brain_tumor_dataset.csv')

# Binary Encoding for Yes/No and Binary Classes
binary_cols = ['Radiation_Treatment', 'Surgery_Performed', 'Chemotherapy',
               'Family_History', 'MRI_Result', 'Follow_Up_Required', 'Gender']

for col in binary_cols:
    df[col] = df[col].apply(lambda x: 1 if x in ['Yes', 'Male', 'Positive'] else 0)

# Target Variable Encoding
df['Tumor_Type'] = df['Tumor_Type'].apply(lambda x: 1 if x == "Malignant" else 0)

# One-hot encoding for categorical columns (drop_first to avoid multicollinearity)
df = pd.get_dummies(df, columns=['Location', 'Histology', 'Stage', 'Symptom_1', 'Symptom_2', 'Symptom_3'],
                    drop_first=True, dtype=int)

# Drop identifier column
df.drop(columns=['Patient_ID'], inplace=True)

# Split data into features and target
X = df.drop(columns='Tumor_Type')
y = df['Tumor_Type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Optional: Scale only numeric features (not strictly needed for tree models but we include it)
numeric_cols = X.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X_train_res[numeric_cols] = scaler.fit_transform(X_train_res[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(class_weight='balanced', random_state=42),
                           param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)

# Train with the resampled data
grid_search.fit(X_train_res, y_train_res)

# Best model from GridSearch
best_rf = grid_search.best_estimator_

# Predictions
predictions = best_rf.predict(X_test)

# Evaluation
print("Best Parameters from GridSearch:", grid_search.best_params_)
print("\nClassification Report:\n", classification_report(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))

# Feature Importance (Optional)
importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Important Features:\n", importances.head(10))
