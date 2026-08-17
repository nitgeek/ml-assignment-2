# ML Assignment 2 - Wine Quality Classification

## a. Problem Statement

Predict whether a red wine sample is of **good quality** (quality score ≥ 7) or **not good quality** (quality score < 7) based on its physicochemical properties. This is a **binary classification** problem where multiple machine learning models are trained and compared on their performance.

## b. Dataset Description

- **Dataset**: Wine Quality - Red Wine
- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/wine+quality
- **Instances**: 1599
- **Features**: 11 (physicochemical input variables)
- **Target**: Binary (1 = Good Quality, 0 = Not Good Quality)

### Features:
1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol

### Target Variable:
- Original quality scores (3-8) converted to binary: **Quality ≥ 7 → Good (1)**, else **Not Good (0)**

## c. GitHub Repository Link

[GitHub Repository](https://github.com/YOUR_USERNAME/ml-assignment-2)

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8938 | 0.8804 | 0.6957 | 0.3721 | 0.4848 | 0.4580 |
| Decision Tree | 0.9062 | 0.8182 | 0.6383 | 0.6977 | 0.6667 | 0.6131 |
| KNN | 0.8938 | 0.8237 | 0.6667 | 0.4186 | 0.5143 | 0.4738 |
| Naive Bayes | 0.8594 | 0.8517 | 0.4844 | 0.7209 | 0.5794 | 0.5131 |
| Random Forest (Ensemble) | 0.9375 | 0.9546 | 0.9259 | 0.5814 | 0.7143 | 0.7045 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieves high accuracy (89.38%) but struggles with recall (37.21%) indicating it misses many good wines. The linear decision boundary is insufficient for the complex non-linear patterns in the data. Best suited as a baseline model. |
| Decision Tree | Provides the most balanced precision-recall trade-off among simple models (69.77% recall, 63.83% precision). However, its AUC (0.8182) is the lowest, suggesting potential overfitting to training data. |
| KNN | Similar accuracy to Logistic Regression but slightly better precision (66.67%). The low recall (41.86%) suggests the distance-based approach is affected by the class imbalance and high-dimensional feature space. |
| Naive Bayes | Has the highest recall (72.09%) among all models, meaning it identifies most good wines. However, it has the lowest precision (48.44%), leading to more false positives. The independence assumption may not hold for correlated wine features. |
| Random Forest (Ensemble) | **Best overall performer** with highest accuracy (93.75%), AUC (0.9546), precision (92.59%), F1 (0.7143), and MCC (0.7045). The ensemble approach effectively handles feature interactions and class imbalance. |

### Overall Winner

**🏆 Random Forest (Ensemble)** is the best model for this dataset with:
- Highest F1 Score: 0.7143
- Highest AUC: 0.9546
- Highest MCC: 0.7045
- Best precision (92.59%) with competitive recall

The ensemble method's ability to combine multiple decision trees reduces overfitting and captures complex non-linear relationships in the wine quality data.

## Streamlit App

**Live App**: [Streamlit App Link](https://YOUR_APP_URL.streamlit.app)

### Features:
- CSV dataset upload for test data
- Model selection dropdown (5 models)
- Display of all evaluation metrics
- Confusion matrix visualization
- Classification report
- All models comparison chart

## Project Structure

```
project-folder/
│── app.py
│── requirements.txt
│── README.md
│── test_data.csv
│── model/
    │── train_models.py
    │── logistic_regression.pkl
    │── decision_tree.pkl
    │── knn.pkl
    │── naive_bayes.pkl
    │── random_forest.pkl
    │── scaler.pkl
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```
