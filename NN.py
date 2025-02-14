from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import random
import sys
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns


import tensorflow as  tf
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#Dataset resampled with SMOTE
df_SMOTE= pd.read_parquet("dataset_SMOTE.parquet")
X = df_SMOTE.drop(columns=['target'])
y = df_SMOTE['target']

y = tf.keras.utils.to_categorical(y-1, num_classes=7)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.7,random_state=SEED)

# Normalizzazion 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#First net 
net1 = tf.keras.models.Sequential([
    # Input layer
    tf.keras.Input(shape=(X_train.shape[1])),
    # Hidden layers
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    # Output layer
    tf.keras.layers.Dense(7,activation="softmax") #Softmax (per classificazione multi-classe)
])

#Second convolutional net

# Reshape the input to 2D image
X_train_2 = X_train.reshape(X_train.shape[0], 9, 6, 1)  
X_test_2 = X_test.reshape(X_test.shape[0], 9, 6, 1)

net2 = tf.keras.Sequential([
    # Convolutional layer
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu',input_shape=(9, 6, 1)),
    tf.keras.layers.Flatten(),
    # Hidden layers
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    # Output layer
    tf.keras.layers.Dense(7, activation='softmax')  # 7 class 
])

#Optimizer
optimizer = 'adam'

net = {
    "Multi Layer Perceptron": (net1,X_train,X_test),
    "Convolutional Neural Network": (net2,X_train_2,X_test_2),
}


results = {}


from tensorflow.keras.callbacks import ModelCheckpoint

# Training and evaluation
for net,(model,X_train,X_test) in net.items():
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    start_time = time.time() 
    hystory =[]
    
    model_checkpoint = ModelCheckpoint(
    f'best_model_{net}.weights.h5',          # Save the model in 'best_model.h5'
    monitor='val_loss',       # Monitor validation loss
    save_best_only=True,     # Save only the best model
    save_weights_only=True ) # Save Only weights
    
    history = model.fit(X_train,y_train,epochs=200
                    ,batch_size=2000, validation_split=0.2,callbacks=[model_checkpoint])  
    end_time = time.time()
    
    # Print model summary
    model.summary()
    
    # Number of parameters
    total_params = sum([np.prod(v.shape) for v in model.trainable_variables])
    # Dimension occupied in bytes (float32 -> 4 byte per param)
    bytes_per_param = 4  # float32 (32-bit precision)
    model_size_bytes = total_params * bytes_per_param

    # Convertion in MB (1024^2 byte = 1 MB)
    model_size_mb = model_size_bytes / (1024 ** 2)  # Megabytes
    
    val_loss = history.history['val_loss']
    train_loss = history.history['loss']
    best_epoch = np.argmin(val_loss)
    val_accuracy = history.history['val_accuracy']
    
    model.load_weights(f'best_model_{net}.weights.h5')
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    results[net] = {"Number of parameters": total_params,"Size [MB]":model_size_mb,"Validation Accuracy":  val_accuracy[best_epoch],"Test accuracy": test_accuracy ,"Training Time (s)": end_time - start_time, "Best Number of Epochs": best_epoch+1}
    
    # Plotting Loss
    epochs = range(1, len(train_loss)+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue')
    plt.yscale("log",base = 10)
    plt.plot(epochs, val_loss, label='Validation Loss', color='green')
    plt.yscale("log",base = 10)
    plt.plot([best_epoch+1, best_epoch+1], [0, val_loss[best_epoch]], 'k--', linewidth=1)  # Linea verticale
    plt.plot([0, best_epoch+1], [val_loss[best_epoch], val_loss[best_epoch]], 'k--', linewidth=1)  # Linea orizzontale

   # Best epoch point
    plt.scatter(best_epoch+1, val_loss[best_epoch], facecolors='none',edgecolors='green',marker='o',s=200, label=f'Best Epoch')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Cross-Entropy (Loss)-({net})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Cross-Entropy (Loss)-({net}).png")
    plt.close()
    
    # Plotting Confusion Matrix
    cm = []
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test_conf_matrix = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test_conf_matrix , y_pred)
    plt.figure(figsize=(8,5))
    plt.title(f'Confusion Matrix-({net}) ')
    sns.heatmap(confusion_matrix(y_test_conf_matrix , y_pred), annot=True,annot_kws={"size": 12})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f"Confusion Matrix-({net}).png")
    plt.close()
    

#Print and save results
print("\n📈 Final Results:")
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("NNs_results.csv")

        








