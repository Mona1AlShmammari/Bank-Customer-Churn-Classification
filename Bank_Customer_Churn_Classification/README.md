Bank Customer Churn Classification

A Machine Learning project that predicts whether a bank customer is likely to leave the bank or continue using its services.

 Project Overview

Customer churn is an important challenge for banks. Identifying customers who may leave can help organizations better understand customer behavior.

This project uses Binary Classification to predict customer churn based on customer information and banking activities.

 ● 0: Customer stays with the bank.
 ● 1: Customer leaves the bank.

The project also provides an interactive Streamlit application where users can enter customer information, select a trained model, view predictions, and compare results across four classification models.

 Project Objectives

 ● Explore and understand the customer churn dataset.
 ● Clean and prepare the data for Machine Learning.
 ● Apply feature engineering to create additional useful features.
 ● Train and evaluate four classification models.
 ● Compare model performance using multiple evaluation metrics.
 ● Build an interactive application for customer churn prediction.

📂 Dataset

Source: Kaggle Playground Series S4E1 – Bank Customer Churn

|Dataset      |Rows   |Columns|
|-------------|------:|------:|
|Training Data|165,034|14     |
|Test Data    |110,023|13     |

The training dataset contains the target column Exited, while the test dataset does not contain target labels.

 Data Preparation

The data preparation process includes:

 ● Exploratory Data Analysis (EDA).
 ● Checking missing values and duplicate records.
 ● Analyzing data distributions and outliers.
 ● Encoding categorical features.
 ● Scaling numerical features where required.
 ● Creating new features through Feature Engineering.

Feature Engineering

Three additional features were created:

 ● Age_Group: Groups customers based on their age.
 ● Tenure_Group: Groups customers based on their bank membership duration.
 ● HasBalance: Indicates whether a customer has a positive bank balance.

The columns id, CustomerId, and Surname were excluded from model predictors.

 Machine Learning Models

Four classification algorithms were trained and evaluated:

 1. Logistic Regression
 2. Decision Tree
 3. Random Forest
 4. Extra Trees

Hyperparameter tuning was demonstrated for Logistic Regression and Decision Tree.

The dataset was divided using a Stratified 80/20 Train-Validation Split to preserve the class distribution.

Preprocessing was included inside Scikit-learn pipelines.

Model Evaluation

The models were evaluated using:

 ● Accuracy
 ● Precision
 ● Recall
 ● F1-Score
 ● ROC-AUC

Validation Results

|Model              |Accuracy|Precision|Recall    |F1-Score  |ROC-AUC   |
|-------------------|-------:|--------:|---------:|---------:|---------:|
|Random Forest      |84.74%  |63.27%   |66.42%    |**64.81%**|87.87%    |
|Extra Trees        |82.39%  |56.30%   |75.00%    |64.32%    |87.92%    |
|Decision Tree      |80.76%  |53.05%   |**78.69%**|63.38%    |**88.15%**|
|Logistic Regression|75.99%  |45.81%   |73.63%    |56.48%    |82.27%    |

Results Summary

 ● Random Forest achieved the highest F1-Score and Accuracy.
 ● Decision Tree achieved the highest Recall and ROC-AUC.
 ● Extra Trees also showed strong performance, especially in Recall.

Random Forest was selected as the best model based on the validation F1-Score.

 Streamlit Application

The project includes an interactive web application built using Streamlit.

Application Features

 ● Enter customer personal and banking information.
 ● Select one of the four trained classification models.
 ● Predict whether a customer may leave or stay.
 ● Display the estimated churn probability.
 ● Compare predictions from all available models.

The application provides an easy way to interact with the trained models without running the training notebooks again.

 Project Structure

Bank_Customer_Churn_Classification/

│
├── Data/
│   ├── Original/
       (sample_submission.csv)  
       (test.csv)    
       (train.csv) 
│   └── Processed/
      (submission_predictions.csv)
      (train_eda_prepared.csv)
      (train_feature_engineered.csv)
│
├── Notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   └── 02_Modeling_and_Evaluation.ipynb

├── Models/│   
    ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── extra_trees.joblib
│   ├── model_comparison.csv
│   ├── project_config.joblib
│   └── training_summary.txt

│
├── Images/
│   ├── model_comparison.png
    └── confusion_matrix_*.png
    ├── confusion_matrix_logistic_regression.png
    └── confusion_matrix_decision_tree.png
    └── confusion_matrix_random_forest.png
├── Presentation/
     └──  Presentation/Bank_Customer.pptx
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

 Installation and Usage

1. Clone the Repository

git clone YOUR_GITHUB_REPOSITORY_LINK

2. Open the Project Folder

cd Bank_Customer_Churn_Classification

3. Install Dependencies

pip install -r requirements.txt

4. Run the Notebooks

jupyter notebook

Run the notebooks in the following order:

 1. 01_EDA_and_Preprocessing.ipynb
 2. 02_Modeling_and_Evaluation.ipynb

The second notebook trains the models and saves them in the Models/ folder.

5. Launch the Streamlit Application

streamlit run main.py

The application will open in your web browser.

 Important Notes

 ● The test dataset is unlabeled and was not used to calculate validation metrics.
 ● Model evaluation was performed using a separate validation set from the training data.
 ● Customer identification columns were excluded from model predictors.
 ● Saved models are used by the Streamlit application.
 ● Predictions are estimates generated by Machine Learning models.

Disclaimer

This project was developed for educational purposes as part of a Machine Learning classification project. It is not intended for real-world banking decisions or financial recommendations.