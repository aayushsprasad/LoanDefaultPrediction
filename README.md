# Credit Risk Prediction Using Machine Learning

This project implements three machine learning models based on loan application data to predict credit risk. The models include a Decision Tree, a Neural Network, and a Gradient Boosting Machine. Each model aims to predict whether a loan will default, aiding in better risk management for financial institutions.

## Models

- **Decision Tree**: A simple yet effective model for making easily interpreted predictions.
- **Neural Network**: A more complex model that can capture nonlinear relationships in data.
- **Gradient Boosting Machine (GBM)**: An ensemble learning method that combines the predictions from multiple decision trees.

## Repository Structure

- `decision_tree_loan_default.py`: Contains the Decision Tree model and its evaluation.
- `neural_network_loan_default.py`: Contains the Neural Network model setup and predictions.
- `gbm_loan_default.py`: Script for the GBM model with cross-validation and performance metrics.

## Setup and Running

### Prerequisites

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- tensorflow (for the Neural Network model)

### Installation

Clone the repository to your local machine: https://github.com/aayushsprasad/LoanDefaultPrediction.git

### Running the Models

To run any of the model scripts, use: python <script_name.py>
For example:python decision_tree_loan_default.py

## Dataset

The data used in this project is from the "Loan Default Prediction" dataset, made available by [Kaggle](https://www.kaggle.com). You can access the dataset through the following link:

[Loan Default Prediction Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default)

We thank the contributors of the dataset for making it publicly available and encourage users to review the original source.

