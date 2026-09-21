# Heart_Disease_Prediction_MachineLearning_LogisticRegression

This project uses Machine Learning with Python to predict whether a person is likely to have heart disease based on various medical and demographic features.

The project demonstrates a complete machine learning workflow, including data preprocessing, exploratory data analysis, model training, evaluation, and prediction.

## Project Overview

Heart disease is one of the major health concerns worldwide. Machine learning can be used to analyze patient information and identify patterns associated with heart disease.

In this project, a classification model is trained using a heart disease dataset to predict the presence or absence of heart disease.

### Objective

The main objective is to build a machine learning model that can:

* Load and understand the dataset
* Perform exploratory data analysis
* Preprocess the data
* Split the dataset into training and testing sets
* Train a machine learning classification model
* Evaluate the model's performance
* Predict heart disease for new patient data

## Technologies Used

* Python
* Jupyter Notebook
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## Project Structure

```text
Heart-Disease-Prediction/
│
├── heart_disease_prediction.ipynb
├── heart_disease_data.csv
├── README.md
└── requirements.txt
```

## Dataset

The dataset contains medical attributes that can be used to predict the presence of heart disease.

Typical features include:

| Feature    | Description                           |
| ---------- | ------------------------------------- |
| `age`      | Age of the patient                    |
| `sex`      | Sex of the patient                    |
| `cp`       | Chest pain type                       |
| `trestbps` | Resting blood pressure                |
| `chol`     | Serum cholesterol                     |
| `fbs`      | Fasting blood sugar                   |
| `restecg`  | Resting electrocardiographic results  |
| `thalach`  | Maximum heart rate achieved           |
| `exang`    | Exercise-induced angina               |
| `oldpeak`  | ST depression induced by exercise     |
| `slope`    | Slope of the peak exercise ST segment |
| `ca`       | Number of major vessels               |
| `thal`     | Thalassemia                           |
| `target`   | Heart disease outcome                 |

> Note: The exact features and target encoding depend on the dataset used in the notebook.

## Machine Learning Workflow

### 1. Import Libraries

The required Python libraries are imported for data manipulation, visualization, and machine learning.

### 2. Load the Dataset

The heart disease dataset is loaded using Pandas.

```python
import pandas as pd

data = pd.read_csv("heart_disease_data.csv")
```

### 3. Exploratory Data Analysis

The dataset is examined to understand:

* Number of rows and columns
* Data types
* Missing values
* Statistical characteristics
* Distribution of the target variable
* Relationships between different features

Matplotlib and Seaborn can be used to visualize the data and identify useful patterns.

### 4. Data Preprocessing

The data is prepared for machine learning by:

* Checking for missing values
* Separating features and target
* Preparing the input variables
* Splitting the data into training and testing sets

### 5. Train the Model

A classification algorithm is trained using the training dataset.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = data.drop("target", axis=1)
Y = data["target"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)
```

### 6. Model Evaluation

The trained model is evaluated using the test dataset.

```python
from sklearn.metrics import accuracy_score

X_test_prediction = model.predict(X_test)

test_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Test Accuracy:", test_accuracy)
```

Accuracy is one metric used to evaluate the classification model. Other metrics such as precision, recall, F1-score, and a confusion matrix can also provide additional information about model performance.

## Making a Prediction

After training the model, it can be used to make predictions for new patient data.

```python
prediction = model.predict(input_data)

if prediction[0] == 0:
    print("The person does not have heart disease")
else:
    print("The person has heart disease")
```

## Results

The trained model is evaluated on unseen test data to determine how well it generalizes to patients that were not included during training.

The notebook contains the detailed analysis, model training process, and evaluation results.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Heart-Disease-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd Heart-Disease-Prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open the `.ipynb` file and run the cells sequentially.

## Requirements

Create a `requirements.txt` file containing:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

## Disclaimer

This project is intended for educational and machine-learning practice purposes only. It is not a medical diagnostic system and should not be used to make real-world medical decisions.

## Tutorial Reference

This project was developed with reference to the following YouTube tutorial:

**Project 9: Heart Disease Prediction using Machine Learning with Python | Machine Learning Projects**

[Watch the tutorial on YouTube](https://youtu.be/qmqCYC-MBQo)

## Learning Outcomes

By completing this project, we can learn how to:

* Work with real-world datasets
* Perform exploratory data analysis
* Prepare data for machine learning
* Build classification models using Scikit-learn
* Evaluate machine learning models
* Make predictions using trained models
* Organize a machine learning project for GitHub

## Author 

**Anushka Verma**
