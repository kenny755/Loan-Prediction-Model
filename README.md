# Loan Prediction Project

This repository contains the code and data for a project that aims to predict loan eligibility based on various applicant features. The project involves data cleaning, preprocessing, feature engineering, and the application of a classification model (Gaussian Naive Bayes) to predict whether a loan will be approved or not.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Repository Contents](#repository-contents)
3.  [Data Description](#data-description)
4.  [Project Implementation](#project-implementation)
    * [4.1. Data Loading and Exploration](#41-data-loading-and-exploration)
    * [4.2. Data Preprocessing](#42-data-preprocessing)
        * [4.2.1. Handling Missing Values](#421-handling-missing-values)
        * [4.2.2. Encoding Categorical Features](#422-encoding-categorical-features)
        * [4.2.3. Converting Income Columns to Numeric and Handling Zero-like Strings](#423-converting-income-columns-to-numeric-and-handling-zero-like-strings)
        * [4.2.4. Splitting Data](#424-splitting-data)
    * [4.3. Model Building and Training](#43-model-building-and-training)
        * [4.3.1. Gaussian Naive Bayes](#431-gaussian-naive-bayes)
    * [4.4. Model Evaluation](#44-model-evaluation)
5.  [Usage](#usage)
6.  [Contributing](#contributing)
7.  [License](#license)

## Project Overview

The goal of this project is to develop a classification model that can predict whether a loan applicant is likely to be approved or not. This is a common problem in the financial industry, and an accurate prediction model can help streamline the loan approval process and reduce risk. We utilize the Gaussian Naive Bayes algorithm for this classification task.

## Repository Contents

* **`train_CVw08PX.csv`:** The training dataset for the loan prediction task.
* **`test_lAUu6aw.csv`:** The test dataset for evaluating the model.
* **`IBM project.ipynb`:** A Jupyter Notebook containing the Python code for data loading, preprocessing, model building, and evaluation.

## Data Description

The datasets contain various features related to loan applicants, including:

* **Loan_ID:** Unique Loan ID.
* **Gender:** Applicant gender (Male/Female).
* **Married:** Applicant married (Yes/No).
* **Dependents:** Number of dependents.
* **Education:** Applicant Education (Graduate/Not Graduate).
* **Self_Employed:** Self-employed (Yes/No).
* **ApplicantIncome:** Applicant income.
* **CoapplicantIncome:** Co-applicant income.
* **LoanAmount:** Loan amount in thousands.
* **Loan_Amount_Term:** Term of loan in months.
* **Credit_History:** Credit history meets guidelines.
* **Property_Area:** Applicant property area (Urban/Semi Urban/Rural).
* **Loan_Status:** Loan approved (Y/N) - Target variable.

## Project Implementation

The `IBM project.ipynb`[https://github.com/kenny755/Loan-Prediction-Model/blob/main/Loan%20Prediction.ipynb] Jupyter Notebook details the steps taken in this project:

### 4.1. Data Loading and Exploration

* Loading the training and testing datasets using pandas.
* Performing initial exploration of the data, including checking the shape, data types, and looking at the first few rows.

### 4.2. Data Preprocessing

* **4.2.1. Handling Missing Values:** Identifying and handling missing values in various columns using appropriate strategies (e.g., imputation with mean/median for numerical features, mode for categorical features).
* **4.2.2. Encoding Categorical Features:** Converting categorical features into numerical representations suitable for the Gaussian Naive Bayes model. This involves using techniques like Label Encoding for binary or ordinal features and One-Hot Encoding (using `pd.get_dummies()`) for nominal features.
* **4.2.3. Converting Income Columns to Numeric and Handling Zero-like Strings:** Ensuring the 'ApplicantIncome' and 'CoapplicantIncome' columns are in a numerical format by converting them using `pd.to_numeric()` and handling any string "0" values appropriately.
* **4.2.4. Splitting Data:** Splitting the training data into training and validation sets to evaluate the model's performance before applying it to the test data.

### 4.3. Model Building and Training

* **4.3.1. Gaussian Naive Bayes:** Initializing and training a Gaussian Naive Bayes classification model using the preprocessed training data.

### 4.4. Model Evaluation

* Evaluating the performance of the trained model on the validation set using appropriate classification metrics (e.g., accuracy, precision, recall, F1-score).

## Usage

To run the project:

1.  Clone this repository to your local machine:
    ```bash
    git clone [repository URL]
    ```
2.  Ensure you have the necessary Python libraries installed. You can install them using pip:
    ```bash
    pip install pandas scikit-learn
    ```
3.  Navigate to the project directory:
    ```bash
    cd loan-prediction-project
    ```
4.  Open and run the `IBM project.ipynb` Jupyter Notebook to execute the code step by step.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please feel free to submit a pull request or open an issue.

## License

[Add your license information here, e.g., MIT License]
