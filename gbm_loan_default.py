import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load your data
data = pd.read_excel('Loan_default.xlsx')  # Adjust path accordingly

# Prepare the data
X = data.drop(['LoanID', 'Default'], axis=1)  # Exclude 'LoanID' and 'Default' from features
y = data['Default'].astype(int)  # Assuming 'Default' is already in a binary format suitable for classification

# Define categorical and numeric columns
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
                    'HasDependents', 'LoanPurpose', 'HasCoSigner']  # Actual categorical columns

numeric_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

# Preprocessing for categorical data: One-hot encoding with handle_unknown set to 'ignore'
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),  # Scale numeric columns
        ('cat', OneHotEncoder(drop='first'), categorical_cols)  # Encode categorical columns
    ])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the XGBoost pipeline
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(eval_metric='logloss'))
])

xgb_pipeline.fit(X_train, y_train)

# Define the parameter grid for GridSearchCV
param_grid = {
    'classifier__max_depth': [3, 5, 7],
    'classifier__min_child_weight': [1, 5, 10],
    'classifier__gamma': [0.5, 1, 1.5],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__n_estimators': [100, 200]
}

# Implement GridSearchCV
grid_search = GridSearchCV(xgb_pipeline, param_grid, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)
grid_search.fit(X_train, y_train)

# Best model results
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Cross-validated ROC AUC score
cv_roc_auc = cross_val_score(best_model, X, y, cv=5, scoring='roc_auc')
print(f"Mean cross-validated ROC AUC: {cv_roc_auc.mean():.2f}")

# Evaluate the best model on the test set
X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
y_pred = xgb_pipeline.predict(X_test)
y_proba = xgb_pipeline.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
kappa = cohen_kappa_score(y_test, y_pred)
print(f'Test set Accuracy: {accuracy:.2f}, ROC AUC: {roc_auc:.2f}, Kappa Score: {kappa:.2f}')

# Feature Importance
feature_names = best_model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(categorical_cols)
feature_names = np.concatenate([feature_names, numeric_cols])

importances = best_model.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Top 10 Important Features:\n", importance_df.head(10))
