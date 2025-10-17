# 🎯 Customer Churn Prediction Dashboard

**A comprehensive Machine Learning application built with Streamlit that predicts customer churn in telecommunications companies.** This project demonstrates a complete Data Science workflow, providing an interactive dashboard for real-time predictions and actionable business analysis.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)](https://en.wikipedia.org/wiki/Machine_learning)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-blue?logo=github)](https://github.com/diegotalaveracampos/churn-prediction-v4)

***

## 📊 Project Overview

This application helps telecommunications companies **proactively identify customers** who are likely to churn. Leveraging a high-performing **XGBoost** model, businesses can implement data-driven retention strategies, optimize resource allocation, and ultimately **increase Customer Lifetime Value (CLV)**.

### 🎯 Key Features

* **🔮 Individual Predictions:** Real-time churn probability for single customer profiles.
* **📋 Batch Analysis:** Upload CSV files for predicting churn on multiple customers simultaneously.
* **📊 Interactive Visualizations:** Comprehensive model performance metrics and Exploratory Data Analysis (EDA).
* **🎯 Actionable Insights:** Personalized retention recommendations based on calculated risk levels.

***

## 🚀 Quick Start

### Prerequisites

* **Python 3.9+**
* **pip**

### Installation and Execution

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/diegotalaveracampos/churn-prediction-v4
    cd churn-prediction-v4
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the complete project setup** (trains the model and saves the necessary `.pkl` files):
    ```bash
    python run_project.py
    ```

***

## 💡 How to Use the Dashboard

| Risk Level | Churn Probability | Recommended Action |
| :---: | :---: | :--- |
| **🟢 Low Risk** | `< 30%` | Maintain service quality; monitor for changes. |
| **🟡 Medium Risk** | `30% - 70%` | Proactive contact; personalized offers. |
| **🔴 High Risk** | `> 70%` | **Immediate retention actions** (e.g., exclusive discounts, dedicated support). |

***

## 🤖 Machine Learning Details

### Dataset Features

The model uses **7,043** customer records and **21 features**, including: `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, and `PaymentMethod`.

### Model Performance

| Model | Accuracy | AUC Score | Cross-Val Score |
| :--- | :---: | :---: | :---: |
| **XGBoost** (Deployed) | **82.5%** | **0.870** | **0.810 $\pm$ 0.015** |
| Random Forest | 81.8% | 0.855 | 0.805 $\pm$ 0.020 |
| Logistic Regression | 80.2% | 0.840 | 0.795 $\pm$ 0.025 |

***

## 🛠️ Tech Stack & Structure

| Category | Tools |
| :--- | :--- |
| **Frontend & Dashboard** | `Streamlit`, `Plotly`, `Matplotlib`, `Seaborn` |
| **Machine Learning** | `Scikit-learn`, `XGBoost`, `Pandas`, `NumPy` |

### 📁 Project Structure

```text
Churn Prediction V2/
├── app/
│   ├── streamlit_app.py  # Main dashboard logic
├── src/
│   └── data_processing.py # Preprocessing functions
├── models/
│   ├── best_model.pkl     # Deployed XGBoost model
│   └── data_processor.pkl # Preprocessing pipeline
├── data/
│   └── telco_churn.csv    # Raw dataset
├── requirements.txt      # Project dependencies
└── README.md
