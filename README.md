# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using both **supervised** and **unsupervised** learning techniques. Given the highly imbalanced nature of the dataset, various preprocessing and evaluation strategies were applied to ensure robust model performance.

## 📊 Dataset
- Source: Kaggle — Credit Card Fraud Detection
- 284,807 transactions; only 492 labeled as fraud (~0.172%)
- Features are anonymized (PCA applied), with only `Time` and `Amount` retained in original form

## 🧹 Preprocessing
- Normalized `Time` using min-max scaling
- Scaled `Amount` using `RobustScaler` to mitigate the effect of outliers
- Used stratified train-test splits to preserve class distribution
- Created a **balanced dataset** via undersampling for fair model training
- Shuffled data for unbiased training

## 🧠 Models Used

### Supervised (on Imbalanced and Balanced Data):
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Linear Support Vector Classifier (SVM)

### Unsupervised (Anomaly Detection):
- **Isolation Forest**: learned patterns of "normal" data to detect outliers
- **DBSCAN**: density-based clustering to identify anomalies

## 🧪 Evaluation
- Classification report: Precision, Recall, F1-score
- Confusion matrix to assess fraud detection performance
- Visualizations of anomalies in unsupervised models

## ✅ Key Insights
- Balancing the dataset significantly improved fraud detection for supervised models
- Ensemble methods outperformed linear models in both precision and recall
- Isolation Forest effectively detected anomalies in an unlabeled setting
- DBSCAN was sensitive to parameter settings and useful for cluster-based outlier detection


