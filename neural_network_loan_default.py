import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from scikeras.wrappers import KerasClassifier

# Load your data
data = pd.read_excel('Loan_default.xlsx')  # Adjust path accordingly

# Prepare the data
X = data.drop(['LoanID', 'Default'], axis=1)  # Exclude 'LoanID' and 'Default' from features
y = data['Default'].astype(int)

# Define categorical and numeric columns
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
                    'HasDependents', 'LoanPurpose', 'HasCoSigner']  # Actual categorical columns

numeric_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

# Preprocessing for categorical data: One-hot encoding and scaling for numeric data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Function to create model, required for KerasClassifier
def create_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(preprocessor.transform(X_train).shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Wrap the model so it can be used by scikit-learn
#neural_network = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10)
# Create and train the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])
pipeline.fit(X_train, y_train)

# Predicting the test set results
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Get probabilities instead of binary predictions

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
kappa = cohen_kappa_score(y_test, y_pred)

print(f'Neural Network - Accuracy: {accuracy:.2f}, ROC AUC: {roc_auc:.2f}, Kappa Score: {kappa:.2f}')
