# **CSCI 632 Project: Leaf Classification**

**Student Name(s): Your Full Name(s)**  
**Kaggle Competition:** [Leaf Classification](https://www.kaggle.com/c/leaf-classification)

---

## **Introduction**

This project involves classifying plant species based on numeric features describing leaf characteristics. The dataset contains shape descriptors and other numerical values representing leaves. The goal was to achieve an accuracy higher than **97.1%**, as set by the Kaggle benchmark.

Three approaches were explored in this project:

1. **Random Forest**  
2. **Support Vector Machine (SVM)** with hyperparameter tuning  
3. **Logistic Regression** with optimized parameters

The dataset was preprocessed, analyzed, and visualized to extract meaningful insights before training these models. The results were validated and compared, with predictions formatted for Kaggle submission.

---

## **Dataset**

* **Features (X):** Numeric attributes describing the leaves.  
* **Target (y):** Species labels.  
* **Test Dataset:** Contains leaf features for which species predictions are required.

The dataset was split into training (80%) and testing (20%) sets for model evaluation. A separate Kaggle test dataset was used for submission.

---

## **Workflow**

### **1\. Data Exploration**

* **Data Examination:** Feature distributions and relationships were visualized using histograms, pair plots, and correlation heatmaps.  
* **Objective:** To identify patterns and redundancies in the data for preprocessing and model optimization.

### **2\. Preprocessing**

* **Feature Scaling:** Standardized using `StandardScaler` to ensure uniform feature ranges, especially important for SVM and neural networks.  
* **Label Encoding:** Species labels were converted into numeric values using `LabelEncoder`.

### **3\. Models**

#### **Random Forest**

* A robust ensemble-based model that trains multiple decision trees and aggregates their results.  
* **Accuracy:** **96.46%**

#### **Support Vector Machine (SVM)**

* An effective classifier that maximizes the margin between classes.  
* **Hyperparameter Tuning:** A grid search explored combinations of `C` and `gamma` values.  
* **Accuracy:** **97.98%**

#### **Logistic Regression**

* A linear model that predicts probabilities for each class.  
* **Multi-class Classification:** Implemented using 'ovr' (one-vs-rest) strategy.  
* **Accuracy:** **98.48%**

### **4\. Kaggle Submission**

* The best model (**Logistic Regression**) was used to generate predictions for the test dataset.  
* **Submission File:**  
  * `id`: Test sample IDs.  
  * Species Columns: Probabilities for each species, normalized to sum to 1 per row.

---

## **Results**

| Model | Accuracy (%) |
| ----- | ----- |
| Random Forest | 96.46 |
| SVM | 97.98 |
| Logistic Regression | **98.48** |

* The **Logistic Regression** model achieved the highest accuracy of **98.48%**, outperforming the Kaggle benchmark of **97.1%**.
* Both SVM and Logistic Regression models exceeded the benchmark accuracy.
* The Random Forest model showed competitive performance but fell slightly short of the benchmark.

---

## **Key Code Components**

### **Data Preprocessing**

* **Feature Scaling:** Ensures all attributes are on the same scale.

python  
Copy code  
`scaler = StandardScaler()`  
`X_scaled = scaler.fit_transform(X)`

* **Label Encoding:** Converts species labels into numeric values.

python  
Copy code  
`label_encoder = LabelEncoder()`  
`y = label_encoder.fit_transform(data["species"])`

### **SVM Hyperparameter Tuning**

* Explored combinations of `C` and `gamma` to optimize SVM performance.

python  
Copy code  
`param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}`  
`grid_search = GridSearchCV(SVC(), param_grid, refit=True)`  
`grid_search.fit(X_train, y_train)`  
`best_svm = grid_search.best_estimator_`

### **Logistic Regression**

* A simple linear model with optimized parameters.

python  
Copy code  
`from sklearn.linear_model import LogisticRegression`

`logreg = LogisticRegression(max_iter=1000)`  
`logreg.fit(X_train, y_train)`

### **Kaggle Submission**

* Predictions from the best model were normalized and saved to a CSV file.

python  
Copy code  
`from sklearn.preprocessing import normalize`

`test_features_scaled = scaler.transform(test_features)`  
`logreg_probabilities = normalize(logreg.predict_proba(test_features_scaled), norm='l1', axis=1)`

`submission = pd.DataFrame(logreg_probabilities, columns=label_encoder.classes_)`  
`submission.insert(0, 'id', test_ids)`  
`submission.to_csv("submission.csv", index=False)`

---

## **Conclusion**

* The project successfully classified leaf species with high accuracy.  
* The **Logistic Regression** model exceeded expectations with an accuracy of **98.48%**.  
* The results were formatted for Kaggle submission, ensuring compliance with the competition's requirements.
