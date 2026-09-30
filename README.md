# HeartDiseaseDataset
A PySpark MLlib project for analyzing and predicting heart disease using classical Machine Learning classification models.
The project follows the requirements of the Machine Learning assignment and implements a complete Machine Learning workflow, including data exploration, preprocessing, feature engineering, model development, evaluation, and prediction.
Deep Learning models are not used in this project.

**Dataset**
The dataset used for this project is the Heart Disease Dataset provided through Kaggle.
Dataset: Heart Disease Dataset. The dataset contains patient-related attributes that can be used to determine whether a patient has heart disease.
**Problem Type**
This project is treated as a binary classification problem.
**Target Variable**
The target variable represents whether heart disease is present or absent.

**Objectives**
-The main objectives of this project are:
-Apply appropriate Machine Learning models to the Heart Disease dataset.
-Use PySpark DataFrames and MLlib for Machine Learning.
-Explore and visualize the dataset.
-Identify and handle missing values and other data-quality issues.
-Perform preprocessing and feature transformation.
-Perform correlation analysis.
-Prepare features and target variables.
-Split the dataset into training and testing sets.
-Train at least three classical Machine Learning models.
-Evaluate the trained models.
-Generate predictions on test data.
-Display and interpret the results.


# Heart Disease Prediction Using PySpark MLlib

## Project Overview

This project focuses on applying **Machine Learning using PySpark, Spark MLlib, and DataFrames** to a Heart Disease dataset.

The main objective is to analyze patient health-related data and build Machine Learning classification models that can predict whether a patient has heart disease.

The project follows the complete Machine Learning workflow:

1. Dataset import
2. Data exploration
3. Data visualization
4. Data preprocessing and cleaning
5. Feature transformation
6. Correlation analysis
7. Feature selection
8. Train-test splitting
9. Machine Learning model development
10. Model evaluation
11. Test-data prediction
12. Result interpretation

**Deep Learning models are not used in this project.**

---

## Dataset

**Dataset:** Heart Disease Dataset

**Source:** Kaggle

**Dataset Link:**
https://www.kaggle.com/datasets/hosammhmdali/heart-disease-dataset

The dataset contains patient-related features that can be used to analyze and predict the presence of heart disease.

### Problem Type

This project is a **binary classification** problem.

### Target

The target variable represents whether heart disease is present or absent.

---

## Objectives

The objectives of this project are:

* Apply appropriate Machine Learning models to the Heart Disease dataset.
* Use PySpark DataFrames for data processing and analysis.
* Use Spark MLlib for Machine Learning model development.
* Explore and visualize the dataset.
* Identify missing values and data-quality issues.
* Analyze possible outliers and feature distributions.
* Perform feature transformation.
* Perform correlation analysis.
* Select appropriate input features.
* Split the data into training and testing sets.
* Build at least three Machine Learning classification models.
* Evaluate the models using appropriate classification metrics.
* Generate predictions on the test dataset.
* Analyze and interpret the prediction results.

---

## Technologies Used

* Python
* PySpark
* Apache Spark
* Spark MLlib
* Spark DataFrames
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Machine Learning Models

The project uses classical Machine Learning models available through **PySpark MLlib**.

The models implemented in the notebook are:

### 1. Logistic Regression

A linear classification algorithm used to predict the probability of the target class.

### 2. Decision Tree Classifier

A tree-based classification algorithm that makes predictions through a sequence of feature-based decisions.

### 3. Random Forest Classifier

An ensemble classification algorithm that combines multiple decision trees to produce predictions.

> Deep Learning models are not used, as required by the assignment.

---

# Project Workflow

## 1. Import Libraries and Dataset

The project begins by:

* Importing the required Python libraries.
* Creating a Spark session.
* Loading the Heart Disease dataset.
* Inspecting the dataset structure.
* Checking the available features and target variable.

---

## 2. Data Visualization and Exploration

Exploratory Data Analysis is performed to understand the dataset.

The notebook includes:

* Display of the first five rows.
* Dataset shape.
* Dataset schema.
* Descriptive statistics.
* Feature inspection.
* Target-class distribution.
* Feature distribution visualizations.
* Relationships between important features and the target.

These steps help identify patterns and understand the characteristics of the dataset before Machine Learning is performed.

---

## 3. Data Pre-processing and Cleaning

The dataset is checked and prepared before model training.

The preprocessing stage includes:

* Checking for missing values.
* Checking for duplicate records.
* Checking data types.
* Examining feature distributions.
* Identifying potential outliers.
* Handling unsuitable or missing values where required.
* Applying appropriate feature transformations.
* Preparing the data for PySpark MLlib.

---

## 4. Correlation Analysis

Correlation analysis is performed to understand the relationships between numerical features.

A correlation matrix and visualization are used to identify:

* Positive relationships.
* Negative relationships.
* Weak relationships.
* Potentially important relationships between features.

This analysis helps understand which features may have useful relationships with the target variable.

---

## 5. Data Preparation

The dataset is divided into:

### Features — X

The input variables used by the Machine Learning models.

### Target — Y

The class label representing the presence or absence of heart disease.

PySpark's `VectorAssembler` is used to combine the selected features into a feature vector.

The dataset is then split into:

* **Training data**
* **Testing data**

The training data is used to develop the models, while the test data is used to evaluate their performance on unseen observations.

---

# 6. Model Building

Three Machine Learning classification models are developed separately using **Spark MLlib**:

```text
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
```

Each model is trained using the training dataset.

The notebook displays the relevant model performance and evaluation results.

---

# 7. Performance Evaluation

The trained models are evaluated using the test dataset.

The evaluation includes appropriate classification metrics such as:

* Accuracy
* Precision
* Recall
* F1 Score

A **confusion matrix** is also generated to analyze the classification results.

The confusion matrix shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

The results are interpreted to understand how effectively the models classify patients into the respective target classes.

---

# 8. Test Data Prediction

After training, the models are used to generate predictions on previously unseen test data.

The prediction results are displayed in the notebook along with the relevant actual and predicted classes.

The results are analyzed to understand the behavior of the Machine Learning models and their ability to classify new observations.

---

# Project Structure

```text
HeartDiseaseDataset/
│
├── HeartDiseaseDataset.ipynb
├── HeartDiseaseDataset.html
├── README.md
├── requirements.txt
└── .gitignore
```

# Results

The notebook provides the results of all three Machine Learning models.

The models are evaluated using the test dataset, and their classification performance is analyzed using appropriate evaluation metrics and confusion matrices.

The final section of the notebook provides predictions on the test data and an interpretation of the results.



# References

* Kaggle — Heart Disease Dataset
  https://www.kaggle.com/datasets/hosammhmdali/heart-disease-dataset

* Databricks — DataFrames with Python
  https://docs.databricks.com/getting-started/dataframes-python.html

* Kaggle — End-to-End PySpark Project
  https://www.kaggle.com/code/towhidultonmoy/end-to-end-pyspark-project

* Kaggle — Advanced PySpark for Exploratory Data Analysis
  https://www.kaggle.com/code/tientd95/advanced-pyspark-for-exploratory-data-analysis
