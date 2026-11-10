
# Telecom Customer Churn Analysis and Prediction

## Project Overview

Customer churn occurs when customers stop using a company's products or services. In the telecommunications industry, predicting customer churn helps companies identify customers who are likely to leave and take appropriate retention actions.

This project uses machine learning to analyze customer data and predict whether a telecom customer will churn or remain with the service.

The project was completed as part of the SaiKet Systems Machine Learning Internship tasks.

## Objectives

- Analyze customer information and identify churn patterns.
- Clean and preprocess the dataset.
- Select important features using Mutual Information.
- Compare multiple machine learning classification algorithms.
- Train a Gradient Boosting classification model.
- Evaluate model performance using classification metrics.

## Dataset

**Dataset:** IBM Telco Customer Churn

The dataset contains 7,043 customer records and 21 original columns.

It includes:
- Customer demographics
- Account information
- Internet and phone services
- Contract information
- Monthly and total charges
- Customer churn status

The `customerID` column was removed because it is an identifier rather than a predictive feature.

The final modeling data contains:
- 19 input features
- 1 target variable (`Churn`)

### Target Variable

| Value | Meaning |
|---|---|
| No | Customer stays |
| Yes | Customer leaves |

Target distribution:

- No Churn: 5,174 (73.46%)
- Churn: 1,869 (26.54%)

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Google Colab / Jupyter Notebook
- Git and GitHub

## Project Workflow

### 1. Data Preparation

- Loaded and inspected the dataset.
- Checked missing values and duplicate records.
- Cleaned whitespace from categorical columns.
- Converted `TotalCharges` into numeric format.
- Handled 11 missing `TotalCharges` values by replacing them with 0.0 because the affected customers had zero tenure.
- Removed the `customerID` column.

### 2. Data Splitting

The dataset was divided into training and testing sets using an 80/20 stratified split.

| Dataset | Records |
|---|---:|
| Training | 5,634 |
| Testing | 1,409 |
| Total | 7,043 |

### 3. Feature Selection

- Applied One-Hot Encoding to categorical variables.
- Used Mutual Information to measure feature relevance.
- Compared different numbers of selected features using five-fold cross-validation.
- Selected 30 features from the 45 encoded features.

### 4. Model Selection

Five classification algorithms were compared using five-fold Stratified Cross-Validation.

| Model | Mean ROC-AUC |
|---|---:|
| Gradient Boosting | 0.846644 |
| Logistic Regression | 0.845614 |
| Random Forest | 0.818576 |
| K-Nearest Neighbors | 0.777024 |
| Decision Tree | 0.664930 |

Gradient Boosting was selected as the final model because it achieved the highest mean cross-validation ROC-AUC.

### 5. Model Training

The final model was trained using:

- Gradient Boosting Classifier
- 100 estimators
- Learning rate: 0.1
- Maximum depth: 3
- Random state: 42

The preprocessing and feature selection steps were included in the final Scikit-learn pipeline.

The trained pipeline was saved using Joblib.

### 6. Model Evaluation

The final model was evaluated on the unseen test dataset.

| Metric | Score |
|---|---:|
| Accuracy | 80.27% |
| Precision | 65.79% |
| Recall | 53.48% |
| F1-Score | 59.00% |
| ROC-AUC | 84.19% |

### Confusion Matrix

| Actual / Predicted | No Churn | Churn |
|---|---:|---:|
| No Churn | 931 | 104 |
| Churn | 174 | 200 |

### Evaluation Insights

- The model correctly classified 80.27% of test customers.
- The ROC-AUC score of 84.19% indicates useful discrimination between churn and non-churn customers.
- Churn precision is 65.79%.
- Churn recall is 53.48%, meaning the model identified 200 of the 374 actual churners.
- The model missed 174 actual churners.

Improving churn recall is an important direction for future work.

## Project Structure

```text
telecom-customer-churn-analysis/
│
├── notebooks/
│   └── Telecom_Customer_Retention_Analysis.ipynb
│
├── data/
│   └── README.md
│
├── models/
│   └── customer_churn_model.joblib
│
├── results/
│   ├── model_selection_results.csv
│   ├── model_evaluation_results.csv
│   ├── confusion_matrix.csv
│   └── classification_report.csv
│
├── reports/
│   └── Telecom_Customer_Churn_Final_Project_Report.pdf
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/telecom-customer-churn-analysis.git
```

### 2. Navigate into the project

```bash
cd telecom-customer-churn-analysis
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Future Improvements

- Improve recall for the churn class.
- Perform hyperparameter tuning.
- Experiment with class balancing techniques.
- Optimize the prediction probability threshold using validation data.
- Apply explainable AI techniques to understand individual predictions.
- Develop a web application for interactive customer churn prediction.
- Monitor model performance on new customer data.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for telecom customer churn analysis and prediction.

The Gradient Boosting model achieved 80.27% accuracy and 84.19% ROC-AUC on the test dataset. Although the model shows useful predictive performance, improving its ability to identify customers who actually churn remains an important future objective.

## Author

**Miruk Yilikal**

Electrical and Computer Engineering Student  
Addis Ababa University

## Acknowledgements

- SaiKet Systems — Machine Learning Internship
- IBM Telco Customer Churn Dataset
- Scikit-learn Documentation
