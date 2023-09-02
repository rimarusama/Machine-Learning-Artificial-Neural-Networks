# Machine Learning Artificial Neural Networks

This project aims to predict customer churn using a neural network model. Customer churn, also known as customer attrition, refers to the phenomenon where customers stop using a service or leave a business. In this project, we leverage machine learning techniques to predict whether a customer is likely to churn based on various features and historical data.

## Prerequisites

Before you begin, ensure you have the following prerequisites in place:

- Python 3.x installed
- Required Python libraries can be installed using the following:
  ```bash
  pip install numpy pandas tensorflow scikit-learn

## Getting Started 

1. Clone this repository to your local machine:
   git clone https://github.com/sudammajhi/Machine-Learning-Artificial-Neural-Networks.git
   cd Machine-Learning-Artificial-Neural-Networks

2. Download the dataset 'Churn_Modelling.csv' and place it in the project directory.

3. Run the Python script 'customer_churn_prediction.py' to train the model and make predictions.

## Usage

1. Run the script 'Artificial Neural Networks.py' to train the neural network model and make predictions on customer churn.
   python Artificial Neural Networks.py

2. The script will output predictions and evaluation metrics such as confusion matrix and accuracy score.

## Code Explanation


Artificial Neural Networks.py:
Loads the dataset from 'Churn_Modelling.csv'.
Preprocesses the data, including encoding categorical variables and scaling features.
Builds and trains a neural network using TensorFlow/Keras.
Makes predictions and evaluates the model's performance.

## Result

The script provides the following results:

Prediction on a sample data point.
Confusion matrix: A table used to describe the performance of a classification model.
Accuracy score: A metric that measures the accuracy of the model's predictions.
