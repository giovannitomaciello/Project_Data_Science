import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix

# Load the dataset
data = fetch_covtype()
X, y = data.data, data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Load the sampled datasets
df_SMOTE= pd.read_parquet("dataset_SMOTE.parquet")
X_SMOTE = df_SMOTE.drop(columns=['target'])
y_SMOTE = df_SMOTE['target']

df_UnderSampler= pd.read_parquet("dataset_UnderSampler.parquet")
X_UnderSampler = df_UnderSampler.drop(columns=['target'])
y_UnderSampler = df_UnderSampler['target']

df_PCA= pd.read_parquet("dataset_PCA.parquet")
X_PCA = df_PCA.drop(columns=['target'])
y_PCA = df_PCA['target']

df_test_PCA= pd.read_parquet("datatest_PCA.parquet")
X_test_PCA = df_test_PCA.drop(columns=['target'])
y_test_PCA = df_test_PCA['target']

df_PCA_SMOTE= pd.read_parquet("dataset_PCA_SMOTE.parquet")
X_PCA_SMOTE = df_PCA_SMOTE.drop(columns=['target'])
y_PCA_SMOTE = df_PCA_SMOTE['target']

df_test_PCA_SMOTE= pd.read_parquet("datatest_PCA_SMOTE.parquet")
X_test_PCA_SMOTE = df_test_PCA_SMOTE.drop(columns=['target'])
y_test_PCA_SMOTE = df_test_PCA_SMOTE['target']

# Define the resampling strategies
strategies = {
    "Baseline (No Resampling)": (X_train, y_train),
    "Oversampling (SMOTE)": (X_SMOTE, y_SMOTE),
    "Undersampling (Random)": (X_UnderSampler, y_UnderSampler),
    "PCA": (X_PCA, y_PCA),
    "PCA+SMOTE": (X_PCA_SMOTE,y_PCA_SMOTE)
}

# Train and evaluate the Random Forest model for each resampling strategy
import time
results = {}
for method, (X_resampled, y_resampled) in strategies.items():
    print(f"\n📊 Method: {method}")
    print("Distribution:", Counter(y_resampled))

    # Train Random Forest
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42)
    model.fit(X_resampled, y_resampled)
    end_time = time.time()

    # Predictions
    if method == "PCA":
        X_test = X_test_PCA
        y_test = y_test_PCA
    
    if method == "PCA+SMOTE":
        X_test = X_test_PCA_SMOTE
        y_test = y_test_PCA_SMOTE
  
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)

    results[method] = {"Accuracy": acc, "Time (s)": end_time - start_time}
    
    # Confusion Matrix
    plt.figure()
    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat.T, square=True,annot=True,fmt="d",cbar=False,xticklabels=range(1, 8), yticklabels=range(1, 8))
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title("Confusion Matrix - Random Forest - " + method)
    plt.savefig(f"ConfusionMatrix_Random Forest_{method}.jpg")
    plt.close()

# Save and display the results
results_df = pd.DataFrame(results).T
results_df.to_csv("resultsRandomForest.csv")

print("\n📈 Final Results:")
print(results_df)

