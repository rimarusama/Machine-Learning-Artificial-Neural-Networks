# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# Read data from a CSV file into a DataFrame
df = pd.read_csv('Churn_Modelling.csv')

# Extract features (X) and target variable (Y) from the DataFrame
X = df.iloc[:, 3:-1].values  # Features
Y = df.iloc[:, -1].values    # Target variable

# Use LabelEncoder to encode categorical data (e.g., country) to numerical values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])  # Encode the third column

# Use ColumnTransformer and OneHotEncoder to handle categorical variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train , Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Standardize feature values using StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Create a Sequential Neural Network model using TensorFlow/Keras
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  # First hidden layer with 6 neurons and ReLU activation
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  # Second hidden layer with 6 neurons and ReLU activation
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  # Output layer with 1 neuron and sigmoid activation

# Compile the model with binary cross-entropy loss and accuracy metric
ann.compile(loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
ann.fit(X_train, Y_train, batch_size=32, epochs=100)

# Make a prediction on a new data point and print the result
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 120000, 2, 1, 1, 50000]])) > 0.5)

# Make predictions on the test data and concatenate the results with the actual values
Y_pred = ann.predict(X_test)
Y_pred = (Y_pred > 0.5)
print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_pred), 1)), 1))

# Calculate and print the confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
accuracy_score(Y_test, Y_pred)
