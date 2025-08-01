# 🏨 Hotel Reservations Predictor

A machine learning system to predict whether a hotel reservation will be canceled or honored, based on historical booking data.

---

## 📌 Problem Statement

Hotel cancellations create operational and financial disruptions. This project uses historical booking data to build a predictive system that classifies whether a reservation is likely to be honored or canceled. The model enables hotels to mitigate losses by enabling smart overbooking or customer retention strategies.

---

## 🧱 End-to-End Architecture

This project follows a full MLOps pipeline with the following stages:

- **Data Ingestion** from Google Cloud Storage  
- **Preprocessing & Feature Engineering** using `pandas`, `numpy`, `scikit-learn`, and `imbalanced-learn`  
  - Includes **SMOTE** oversampling to balance the dataset  
- **Model Training** using LightGBM (`lgb.LGBMClassifier`) with hyperparameter tuning via `RandomizedSearchCV`  
- **Experiment Tracking** with MLflow (logging metrics, parameters, models)  
- **Version Control** for both data and code using Git  
- **Web Application** built with Flask for real-time predictions  
- **Containerization & Deployment** using Docker and Jenkins CI/CD to Google Cloud Run  

---

## ⚙️ Tech Stack

- **Languages**: Python  
- **ML Libraries**: pandas, numpy, scikit-learn, imbalanced-learn, lightgbm  
- **MLOps Tools**: MLflow, Git, Jenkins, Docker, Google Cloud Run  
- **Web App**: Flask  
- **Other**: Google Cloud Storage, PyYAML  

---

## 🧪 Model Performance

| Metric     | Value    |
|------------|----------|
| Accuracy   | 0.8788   |
| F1 Score   | 0.8812   |
| Precision  | 0.8640   |
| Recall     | 0.8991   |

Hyperparameters and runs are logged and tracked in **MLflow UI**.
