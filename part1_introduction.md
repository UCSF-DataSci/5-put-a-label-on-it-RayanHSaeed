# Install necessary packages

```python
%pip install -r requirements.txt
```
# Part 1: Introduction to Classification & Evaluation

**Objective:** Load the synthetic health data, train a Logistic Regression model, and evaluate its performance.

## 1. Setup

Import necessary libraries.

```python
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
```

## 2. Data Loading

Implement the `load_data` function to read the dataset.

```python
def load_data(file_path):
    """
    Load the synthetic health data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data
    """
    # Load the CSV file using pandas
    return pd.read_csv(file_path)
```

## 3. Data Preparation

Implement `prepare_data_part1` to select features, split data, and handle missing values.

```python
def prepare_data_part1(df, test_size=0.2, random_state=42):
    """
    Prepare data for modeling: select features, split into train/test sets, handle missing values.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Drop unused columns
    df = df.drop(columns=["timestamp", "patient_id"])

    # Separate target and features
    X = df.drop(columns=["disease_outcome"])
    y = df["disease_outcome"]

    # Handle missing values (before train-test split!)
    for col in X.select_dtypes(include=["float64", "int64"]).columns:
        X[col] = X[col].fillna(X[col].mean())
    
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].fillna(X[col].mode()[0])

    # One-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
```

## 4. Model Training

Implement `train_logistic_regression`.

```python
def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained logistic regression model
    """
    # Initialize and train a LogisticRegression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model
```

## 5. Model Evaluation

Implement `calculate_evaluation_metrics` to assess the model's performance.

```python
def calculate_evaluation_metrics(model, X_test, y_test):
    """
    Calculate classification evaluation metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, auc, and confusion_matrix
    """
    # 1. Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # For AUC

    # 2. Calculate metrics: accuracy, precision, recall, f1, auc
    # 3. Create confusion matrix
    # 4. Return metrics in a dictionary
    
     metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    return metrics
```

## 6. Save Results

Save the calculated metrics to a text file.

```python
# Create results directory and save metrics
# 1. Create 'results' directory if it doesn't exist
# 2. Format metrics as strings
# 3. Write metrics to 'results/results_part1.txt'

import json
import os

def save_metrics(metrics, filepath='results/results_part1.txt'):
    os.makedirs('results', exist_ok=True)
    with open(filepath, 'w') as f:  
        json.dump(metrics, f, indent=4)
```

## 7. Main Execution

Run the complete workflow.

```python
# Main execution
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Prepare data
    X_train, X_test, y_train, y_test = prepare_data_part1(df)
    
    # 3. Train model
    model = train_logistic_regression(X_train, y_train)
    
    # 4. Evaluate model
    metrics = calculate_evaluation_metrics(model, X_test, y_test)
    
    # 5. Print metrics
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    
    # 6. Save results
    save_metrics(metrics)
    
    # 7. Interpret results
    interpretation = interpret_results(metrics)
    print("\nResults Interpretation:")
    for key, value in interpretation.items():
        print(f"{key}: {value}")
```

## 8. Interpret Results

Implement a function to analyze the model performance on imbalanced data.

```python
def interpret_results(metrics):
    """
    Analyze model performance on imbalanced data.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        
    Returns:
        Dictionary with keys:
        - 'best_metric': Name of the metric that performed best
        - 'worst_metric': Name of the metric that performed worst
        - 'imbalance_impact_score': A score from 0-1 indicating how much
          the class imbalance affected results (0=no impact, 1=severe impact)
    """
    # 1. Determine which metric performed best and worst
    # 2. Calculate an imbalance impact score based on the difference
    #    between accuracy and more imbalance-sensitive metrics like F1 or recall
    # 3. Return the results as a dictionary
    scores = {k: v for k, v in metrics.items() if k in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
    
    best_metric = max(scores, key=scores.get)
    worst_metric = min(scores, key=scores.get)
    
    imbalance_impact_score = round(abs(metrics['accuracy'] - metrics['f1']), 2)
    

    # Placeholder return - replace with your implementation
    return {
        'best_metric': best_metric,
        'worst_metric': worst_metric,
        'imbalance_impact_score': min(imbalance_impact_score, 1.0)
    }