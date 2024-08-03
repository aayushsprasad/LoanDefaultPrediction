import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, cohen_kappa_score, make_scorer

# Load the Excel file
data = pd.read_excel('Loan_default.xlsx')  # Adjust path accordingly

# Prepare the data
X = data.drop(['LoanID', 'Default'], axis=1)  # Exclude 'LoanID' and 'Default' from features
y = data['Default'].astype(int)

# Identifying categorical columns for one-hot encoding
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
                    'HasDependents', 'LoanPurpose', 'HasCoSigner']  # Actual categorical columns

numeric_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
# Preprocessing pipeline for applying OneHotEncoder to categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop=None), categorical_cols)  # Changed drop to None to include all categories
    ], remainder='passthrough')

# Pipeline setup with a decision tree classifier
dt_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Parameter grid for GridSearchCV
param_grid = {
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 10, 20, 40],
    'classifier__min_samples_leaf': [1, 2, 5, 10]
}

kappa_scorer = make_scorer(cohen_kappa_score)

# Setup GridSearchCV to tune hyperparameters and perform cross-validation
grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring=kappa_scorer, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
print("Best Kappa score:", grid_search.best_score_)
print("Best parameters:", grid_search.best_params_)

# Best model evaluation
best_model = grid_search.best_estimator_
y_pred_dt = best_model.predict(X_test)

# Calculate performance metrics
accuracy_dt = accuracy_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
conf_mat_dt = confusion_matrix(y_test, y_pred_dt)
kappa_score_dt = cohen_kappa_score(y_test, y_pred_dt)

# Extract feature importances from the best decision tree model
feature_importances = best_model.named_steps['classifier'].feature_importances_
encoded_features = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
numeric_features = X.select_dtypes(exclude=['object']).columns.tolist()
all_features = list(encoded_features) + numeric_features

feature_importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Output results
print("Decision Tree - Accuracy:", accuracy_dt, "Recall:", recall_dt, "Kappa Score:", kappa_score_dt)
print("Confusion Matrix:\n", conf_mat_dt)
print("Feature Importances:\n", feature_importance_df.head(10))
