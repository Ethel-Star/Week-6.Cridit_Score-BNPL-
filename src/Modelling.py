# src/modelling.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Function to load and clean data
def load_data(file_path):
    data = pd.read_csv(file_path)
    print(f"Data Loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    # Additional cleaning logic if necessary
    return data

# Function to split data into train and test sets
def split_data(data, target, features):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Data Split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test

# Function to train Logistic Regression model
def train_logreg(X_train, y_train):
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg.fit(X_train, y_train)
    return logreg

# Function to train Random Forest model
def train_rf(X_train, y_train):
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    return rf

# Function for hyperparameter tuning using RandomizedSearchCV
def tune_random_forest(X_train, y_train):
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }
    
    rf_search = RandomizedSearchCV(
        rf, param_distributions=param_grid, 
        n_iter=20,  # Fewer iterations for speed
        cv=5, 
        scoring='roc_auc', 
        n_jobs=-1, 
        random_state=42
    )
    rf_search.fit(X_train, y_train)
    print("Best Hyperparameters:", rf_search.best_params_)
    print(f"Best ROC-AUC: {rf_search.best_score_:.3f}")
    
    return rf_search  # Return the RandomizedSearchCV object

# Function to evaluate model performance
def evaluate_model(X_test, y_test, model):
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # For AUC

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Print metrics
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"AUC: {auc:.3f}")

    # Return metrics as a dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
def radar_chart(lr_metrics, rf_metrics):
    categories = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    lr_values = [lr_metrics['accuracy'], lr_metrics['precision'], lr_metrics['recall'], lr_metrics['f1'], lr_metrics['auc']]
    rf_values = [rf_metrics['accuracy'], rf_metrics['precision'], rf_metrics['recall'], rf_metrics['f1'], rf_metrics['auc']]
    
    # Number of variables
    N = len(categories)

    # Compute angle for each category
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Complete the loop by appending the first value
    lr_values += lr_values[:1]
    rf_values += rf_values[:1]
    angles += angles[:1]

    # Create subplots inside the function
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))

    # Radar chart on the first subplot
    axs[0].fill(angles, lr_values, color='blue', alpha=0.25, label='Logistic Regression')
    axs[0].fill(angles, rf_values, color='green', alpha=0.25, label='Random Forest')
    axs[0].plot(angles, lr_values, color='blue', linewidth=2)
    axs[0].plot(angles, rf_values, color='green', linewidth=2)
    axs[0].set_yticklabels([])
    axs[0].set_xticks(angles[:-1])
    axs[0].set_xticklabels(categories)
    axs[0].set_title('Model Performance Comparison', size=12)
    axs[0].legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    # Bar chart on the second subplot
    x = np.arange(len(categories))
    width = 0.35

    lr_values = [lr_metrics[key] for key in categories]
    rf_values = [rf_metrics[key] for key in categories]

    axs[1].bar(x - width/2, lr_values, width, label='Logistic Regression', color='blue', alpha=0.7)
    axs[1].bar(x + width/2, rf_values, width, label='Random Forest', color='green', alpha=0.7)

    axs[1].set_xticks(x)
    axs[1].set_xticklabels(categories)
    axs[1].set_title('Bar Plot of Model Metrics', size=12)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# Function to plot hyperparameter tuning results
def plot_hyperparameter_results(rf_search):
    results = pd.DataFrame(rf_search.cv_results_)
    top_results = results.sort_values('rank_test_score').head(20)  # Top 20 combinations
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1: Hyperparameter Interaction (Scatter)
    sns.scatterplot(
        x='param_n_estimators',
        y='param_max_depth',
        size='mean_test_score',
        hue='mean_test_score',
        sizes=(50, 200),
        palette='viridis',
        data=top_results,
        ax=ax1
    )
    ax1.set_title('Hyperparameter Interaction (Top 20 Combinations)', fontsize=12)
    ax1.set_xlabel('n_estimators')
    ax1.set_ylabel('max_depth')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 2: ROC-AUC Distribution Across Iterations (Sorted)
    sorted_scores = results['mean_test_score'].sort_values(ascending=False).reset_index(drop=True)
    ax2.plot(sorted_scores, 'o-', color='orange')
    ax2.set_title('Sorted ROC-AUC Scores Across Iterations', fontsize=12)
    ax2.set_xlabel('Iteration (Sorted by ROC-AUC)')
    ax2.set_ylabel('Mean ROC-AUC (5-Fold CV)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Function to plot feature importances for Random Forest
def plot_feature_importance(model, X_train):
    # Extract feature importances and features
    feature_importance = model.feature_importances_
    features = X_train.columns

    # Sort feature importances in descending order
    sorted_idx = feature_importance.argsort()

    # Generate a colormap
    cmap = cm.get_cmap('viridis', len(features))  # Use any colormap like 'viridis', 'plasma', etc.
    colors = [cmap(i) for i in range(len(features))]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(features[sorted_idx], feature_importance[sorted_idx], color=np.array(colors)[sorted_idx])
    plt.xlabel("Feature Importance", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.title("Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Saving models
def save_model(model, filename):
    import joblib
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

# Load a saved model
def load_model(filename):
    import joblib
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

