import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE

import xgboost as xgb
from xgboost import XGBClassifier

def load_and_preprocess_data(json_file):
    # Load dataset
    with open(json_file, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Check that 'crop' is in the DataFrame
    if 'crop' not in df.columns:
        raise ValueError("Dataset must include a 'crop' column representing the target.")
    
    # Label encode the target variable
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['crop'])
    
    # If month exists, perform cyclical encoding, then drop the raw column
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df.drop(columns=['month'], inplace=True)
    else:
        print("Warning: 'month' column not found. Skipping cyclical encoding.")
    
    return df, le

def plot_class_distribution(df):
    # Plot class distribution for crops
    plt.figure(figsize=(14, 6))
    sns.countplot(data=df, x='crop', order=df['crop'].value_counts().index)
    plt.xticks(rotation=45, ha='right')
    plt.title("Crop Class Distribution")
    plt.tight_layout()
    plt.show()

def prepare_features(df):
    # Define feature columns based on available data.
    base_features = ['temperature', 'humidity', 'precipitation', 'soilPh',
                     'nitrogen', 'phosphorus', 'potassium', 'moisture']
    cyclical_features = ['month_sin', 'month_cos']
    features = base_features.copy()
    for feat in cyclical_features:
        if feat in df.columns:
            features.append(feat)
    
    # Extract features and labels
    X = df[features].values
    y = df['label'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, features

def visualize_features(X, y, le):
    # Use t-SNE to visualize feature separability
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab20', s=20)
    plt.title("t-SNE Visualization of Features")
    plt.colorbar(scatter, ticks=range(len(le.classes_)), label="Crop Label")
    plt.show()

def run_xgb_grid_search(X_train, X_test, y_train, y_test, le):
    # Use XGBClassifier (scikit-learn API) so we can perform GridSearchCV.
    xgb_clf = XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y_train)), 
                            eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    
    # Define a parameter grid.
    # You can further experiment with these values.
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Initialize GridSearchCV.
    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-validation Accuracy: {:.2f}%".format(grid_search.best_score_ * 100))
    
    # Use the best estimator on the test set.
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred) * 100
    print("\nXGBoost Test Accuracy: {:.2f}%".format(test_accuracy))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.show()
    
    return best_model

if __name__ == "__main__":
    # Load and preprocess data
    df, le = load_and_preprocess_data("dataset.json")
    
    # Plot the class distribution to observe potential imbalance issues.
    plot_class_distribution(df)
    
    # Prepare features and labels
    X_scaled, y, scaler, feature_names = prepare_features(df)
    
    # Visualize feature separability using t-SNE
    visualize_features(X_scaled, y, le)
    
    # Split data (stratified) into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Run Grid Search to optimize XGBoost hyperparameters.
    print("=== Running XGBoost Hyperparameter Tuning ===")
    best_model = run_xgb_grid_search(X_train, X_test, y_train, y_test, le)
    
    # Save the tuned model and preprocessing objects for later inference.
    best_model.save_model("xgb_best_model.json")
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    
    print("Tuned model and preprocessing objects saved.")
