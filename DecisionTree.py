import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix


# Load the dataset
data = fetch_covtype()
X, y = data.data, data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define the resampling strategies
SMOTE = True
UnderSampler = True
PCA = True
PCA_SMOTE = True

# Load the sampled datasets
if SMOTE==True:
    df_SMOTE= pd.read_parquet("dataset_SMOTE.parquet")
    X_SMOTE = df_SMOTE.drop(columns=['target'])
    y_SMOTE = df_SMOTE['target']

if UnderSampler == True:
    df_UnderSampler= pd.read_parquet("dataset_underSampler.parquet")
    X_UnderSampler = df_UnderSampler.drop(columns=['target'])
    y_UnderSampler = df_UnderSampler['target']

if PCA == True:
    df_PCA= pd.read_parquet("dataset_PCA.parquet")
    X_PCA = df_PCA.drop(columns=['target'])
    y_PCA = df_PCA['target']

    df_test_PCA= pd.read_parquet("datatest_PCA.parquet")
    X_test_PCA = df_test_PCA.drop(columns=['target'])
    y_test_PCA = df_test_PCA['target']

if PCA_SMOTE == True:
    df_PCA_SMOTE= pd.read_parquet("dataset_PCA_SMOTE.parquet")
    X_PCA_SMOTE = df_PCA_SMOTE.drop(columns=['target'])
    y_PCA_SMOTE = df_PCA_SMOTE['target']

    df_test_PCA_SMOTE= pd.read_parquet("datatest_PCA_SMOTE.parquet")
    X_test_PCA_SMOTE = df_test_PCA_SMOTE.drop(columns=['target'])
    y_test_PCA_SMOTE = df_test_PCA_SMOTE['target']

# Define the resampling strategies
strategies = {
    "Baseline (No Resampling)": (X_train, y_train)
}

if SMOTE==True:
    strategies["Oversampling (SMOTE)"] = (X_SMOTE, y_SMOTE)

if UnderSampler==True:
    strategies["Undersampling (Random)"] = (X_UnderSampler, y_UnderSampler)

if PCA==True:
    strategies["PCA"] = (X_PCA, y_PCA)

if PCA_SMOTE==True:
    strategies["PCA+SMOTE"] = (X_PCA_SMOTE, y_PCA_SMOTE)

# Train and evaluate the Decision Tree model for each resampling strategy
import time
results = {}
for method, (X_resampled, y_resampled) in strategies.items():
    print(f"\n📊 Method: {method}")
    print("Distribution:", Counter(y_resampled))

    # Train Decision Tree
    start_time = time.time()
    model = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    model.fit(X_resampled, y_resampled)
    end_time = time.time()

    # Test Decision Tree
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
    sns.heatmap(mat.T, square=True,annot=True,fmt="d",cbar=False)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title("Confusion Matrix - Decision Tree - " + method)
    plt.savefig(f"ConfusionMatrix_DecisionTree_{method}.jpg")
    plt.close()

# Save the results in a DataFrame
results_df = pd.DataFrame(results).T
# Save the results in a CSV file
results_df.to_csv("resultsDecisionTree.csv")
# Display the results
print("\n📈 Final Results:")
print(results_df)

