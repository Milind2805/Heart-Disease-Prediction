# Heart-Disease-Prediction
Heart disease prediction using Logistic Regression, SVM, KNN, Decision Tree and Random Forest. Best accuracy: 90.74% with Logistic Regression.
# Heart Disease Prediction 

A machine learning project that predicts whether a patient has heart disease
based on clinical parameters like age, blood pressure, cholesterol, and more.

##  About
This is my second machine learning project, built while learning ML during
my first year of B.Tech Civil Engineering at MANIT Bhopal.
Unlike my previous Iris project which had a perfect dataset, this project
involved real medical data and comparing multiple algorithms to find the
best performing model.

## 🛠️ Tech Stack
- Python
- scikit-learn
- pandas
- seaborn
- matplotlib

## 📊 Dataset
- 14 clinical features including Age, Sex, Chest Pain Type, Blood Pressure,
  Cholesterol, Max Heart Rate, and more
- Target column: Heart Disease (0 = No, 1 = Yes)
- No missing or duplicate values

## 🔍 What This Project Does
- Explores and visualises the dataset
- Trains and compares 5 ML models
- Evaluates using Cross Validation for reliable accuracy
- Plots Confusion Matrix for the best model
- Predicts heart disease for new patient input

## 📈 Model Comparison Results

| Model | Train Accuracy | Test Accuracy 
| Logistic Regression | 83.79% | 90.74%  
| SVM | 90.74% | 88.88% |
| Random Forest | 100% | 85.18% |
| KNN | 86.57% | 81.48% |
| Decision Tree | 100% | 75.92% |

## 🏆 Key Finding
Logistic Regression outperformed all other models with 90.74% test accuracy
and no overfitting — its test accuracy was actually higher than its train
accuracy, showing excellent generalisation.

Decision Tree showed clear overfitting (100% train, 75.92% test), which is
a common issue with tree-based models on small datasets.

## 🚀 How to Run
1. Clone this repository
2. Install dependencies:
   pip install pandas scikit-learn seaborn matplotlib
3. Download the dataset from Kaggle (Heart Disease Dataset by John Smith)
4. Run `heart disease prediction model.py`

## 👤 Author
Milind Kumar Singh
B.Tech Civil Engineering, MANIT Bhopal (2025-29)
