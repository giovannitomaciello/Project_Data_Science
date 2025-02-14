import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import time 
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns


start_time = time.time()

#Import dataset
data = fetch_covtype()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# RandomUnderSampler
under_sampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = under_sampler.fit_resample(X_train, y_train)

# Create a DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=data.feature_names)  # Create a Datframe with features
df_resampled['target'] = y_resampled  # Add target column

# Save in parquet format
df_resampled.to_parquet("dataset_underSampler.parquet", index=False)

end_time = time.time()

print(f"Time to apply UnderSampler: {end_time - start_time:.2f} seconds")


# Distribution of classes before and after UnderSampler

# Count the classes before UnderSampler
counter = Counter(y)

plt.figure(figsize=(8,5))
sns.barplot(x=list(counter.keys()), y=list(counter.values()), palette="viridis")
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.title("Distribution of classes in the original dataset")
plt.savefig("Distribution of classes in the original dataset.png")


# Count the classes after UnderSampler
counter = Counter(y_resampled)

plt.figure(figsize=(8,5))
sns.barplot(x=list(counter.keys()), y=list(counter.values()), palette="viridis")
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.title("Distribution of classes after UnderSampler")
plt.savefig("Distribution of classes after UnderSampler.png")
