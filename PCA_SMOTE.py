from sklearn.preprocessing import StandardScaler
import time 
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

start_time = time.time()

# Load the sampled dataset
df_SMOTE= pd.read_parquet("dataset_SMOTE.parquet")
X_SMOTE = df_SMOTE.drop(columns=['target'])
y_SMOTE = df_SMOTE['target']

X_train, X_test, y_train, y_test = train_test_split(X_SMOTE, y_SMOTE, test_size=0.2, stratify=y_SMOTE, random_state=42)

print("X_train shape:", X_train.shape)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X_train)
data_test_scaled = scaler.fit_transform(X_test)
print("Data scaled shape:", data_scaled.shape)

# Apply PCA
pca = PCA()
pca.fit(data_scaled)

# Find the number of principal components to reach at least 98% of variance
explained_variance = pca.explained_variance_ratio_.cumsum()
n_comp = next(i for i, v in enumerate(explained_variance) if v>=0.98) + 1
print(f"Number of principal componets to reach at least 98% of variance: {n_comp} ")

# Apply PCA and create a DataFrame
pca = PCA(n_components=n_comp)
pca.fit(data_scaled)
principal_components = pca.components_.T
X_pca = data_scaled @ principal_components
print("X_pca shape:", X_pca.shape)
loadings = pd.DataFrame(data=X_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
loadings['target'] = y_train.reset_index(drop=True) # Aggiungi la colonna target
# Save the dataset in parquet format
loadings.to_parquet("dataset_PCA_SMOTE.parquet", index=False)

# Apply PCA to the test set
pca.fit(data_test_scaled)
# Create a DataFrame for the test set
X_test_pca = data_test_scaled @ principal_components
loadings_test = pd.DataFrame(data=X_test_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
loadings_test['target'] = y_test.reset_index(drop=True)
loadings_test.to_parquet("datatest_PCA_SMOTE.parquet", index=False)

end_time = time.time()

print(f"Time to apply PCA+SMOTE: {end_time - start_time:.2f} seconds")


# Plot explained variance
plt.figure()
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of principal components')
plt.ylabel('Explained variance')
plt.title('Selecting the number of principal components')
plt.axhline(y=0.98, color='r', linestyle='--', label=f'98% explained variance')
plt.legend(loc='best')
plt.savefig("N_Components_choice.png")
plt.close()


# Plot PCA
plt.figure()
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_train, alpha=0.5, s=10, cmap=plt.cm.get_cmap("jet",10)) # alfa = trasperency, s = size
                                                                                                            # we want 10 colours: cmap=plt.cm.get_cmap("jet",10)
plt.colorbar() # we have a label for each color
plt.savefig("PCA.png")
plt.close()