## Heartbeat Prediction Using Machine Learning

## Project Objectives
The objective of this project is to develop a machine learning model to predict heartbeat anomalies. The project focuses on analyzing heart rate data to detect irregularities or abnormalities in heartbeat patterns, which can be useful in identifying potential health issues. The dataset consists of heartbeat measurements, and the goal is to build a model that can classify whether a heartbeat is normal or anomalous based on 80 features representing each measurement.
The project aims to
1. Preprocess and analyze the heartbeat dataset.
2. Build and evaluate multiple machine learning models for classification.
3. Identify the model with the highest prediction accuracy.
3. Compare the performance of various models, including traditional neural networks and recurrent neural networks (RNNs) such as LSTM and GRU.

## About the Dataset
The dataset for this project is provided by MIT and contains 9,026 rows of heartbeat measurements. Each measurement is represented by 80 input features (T1 to T80), along with a Target column indicating the classification of the heartbeat. The target variable represents different categories of heartbeats, including normal and anomalous heartbeats.
## Key Dataset Information
Number of samples: 9,026
Number of features: 80 (T1 to T80)
Target variable: Classification of heartbeats (Normal or Anomalous)
File: heartbeat_cleaned.csv

## Methodology
The approach followed for this project involves several stages:
# Data Preprocessing
The dataset was cleaned, and any missing values were handled.
The dataset was split into training (70%) and testing (30%) sets.
The features (T1 to T80) were separated from the target variable, which was converted to a suitable format for classification.

# Model Selection
Multiple models were tested, including a simple DummyClassifier as a baseline model, and more advanced models such as:
Cross-Sectional Neural Network (1 hidden layer).
Deep Neural Network (Multiple hidden layers).
LSTM (Long Short-Term Memory) networks.
GRU (Gated Recurrent Units) networks.

# Model Training and Evaluation:
Models were trained using the training set and evaluated on the test set using accuracy as the primary metric.
Additional evaluation metrics such as loss were also considered.
The performance of each model was compared to the baseline, and insights were drawn regarding the best-performing model.
Hyperparameter Tuning:
Optimizers like Adam and Nadam were used to train the models.
Early stopping was implemented to prevent overfitting.

## Key Features
The dataset includes 80 features (T1 to T80), which represent various measurements of the heartbeats. These features were used as inputs to the machine learning models. The primary feature set includes:
T1 to T80: Numerical values representing the characteristics of each heartbeat.
Target: A categorical value indicating whether the heartbeat is normal or anomalous. The target variable is the classification to predict.

## Results
The performance of the models was evaluated based on test accuracy. The following models were tested:
# Model Name	Test Accuracy
Baseline Model	58.87%
Cross-Sectional Neural Network (1 layer)	92.46%
Deep Neural Network (Multiple layers)	93.13%
LSTM (1 layer)	91.37%
Deep LSTM (2 layers)	78.81%
GRU (1 layer)	94.14%
Deep GRU (2 layers)	85.80%

## Final Metrics
After training and evaluating several models, the GRU (1 layer) model achieved the highest test accuracy of 94.14%. This model outperformed all others and the baseline model, which had an accuracy of only 58.87%.
## Key Metrics for Best Model (GRU - 1 layer):
Test Accuracy: 94.14%
Train Accuracy: 94.62%
Loss: 0.23
Precision: 0.94
Recall: 0.94


## Insights:
The GRU model with one layer achieved the highest accuracy, demonstrating the ability of the model to capture complex patterns in the data.
The LSTM models, although promising, did not outperform the GRU models in this case. This suggests that GRU's fewer parameters may have led to faster training without compromising performance.
The deep neural network models also performed well, but they were more prone to overfitting due to their larger number of parameters.

## Future Work:
Cross-validation: Implementing cross-validation would provide a more robust evaluation of the model performance.
Hyperparameter Tuning: Further tuning of the learning rates, batch sizes, and other hyperparameters could improve model accuracy.
Feature Engineering: Exploring additional feature transformations or selecting important features could enhance model performance.
Real-World Testing: Implementing this model in a real-time health monitoring system could further validate its effectiveness in detecting anomalies in heartbeat patterns.
