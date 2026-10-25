# bank-customer-churn-prediction
End-to-end machine learning project for bank customer churn prediction. Includes exploratory analysis, feature engineering, model selection, hyperparameter tuning, SHAP-based interpretation, and a saved scoring pipeline built around a threshold-tuned LightGBM classifier.


### notebook structure

| notebook | content |
|---|---|
| `01_data_understanding_and_eda` | Data quality checks and exploratory analysis |
| `02_data_preprocessing` | Feature engineering, encoding, scaling, stratified train/test split, SMOTE on the training set |
| `03_model_building_and_evaluation` | Logistic Regression, Random Forest, XGBoost and LightGBM compared; LightGBM selected |
| `04_hyperparameter_tuning` | Optuna tuning with cross-validation that applies SMOTE inside each fold to avoid leakage |
| `05_model_interpretation` | SHAP analysis, error analysis and F2-optimised threshold selection |
| `06_deployment` | Single pipeline from raw input to prediction, saved together with its threshold, plus a SHAP explainer |


### final model

Tuned LightGBM inside an `imblearn` pipeline, with a recall-oriented decision threshold of 0.341 (chosen because missing a churner costs more than a false alarm). Test set: 2,000 customers, 407 of them churners.

| metric | default threshold (0.5) | chosen threshold (0.341) |
|---|---|---|
| Accuracy | 85.00% | 78% |
| Precision | 62.30% | 48% |
| Recall | 66.58% | 80% |
| F1 | 64.37% | 60% |
| ROC-AUC | 86.52% | 86.52% |


### limitations

- A sizeable share of missed churners do not carry the `ProductsRisk` flag, and some older customers churn despite not fitting the typical churner profile, so additional features may be needed.
- False positives resemble real churners on the main features, so some false alarms are likely to remain at the chosen threshold.
- The threshold was chosen on the same test set used to report its performance, so the reported precision and recall may be slightly optimistic.
- `Gender_Male` is among the top five drivers, with female customers pushed towards churn. Using it directly to decide who receives retention offers could be unfair and should be reviewed before real use.


### reproducing this

Dataset: [Churn Modelling (Kaggle)](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling).
Notebooks were run in Google Colab (Python 3.13). Install the pinned libraries with `pip install -r requirements.txt` and run the notebooks in order. Keep `churn_utils.py` in the same folder as the saved pipeline, since it is needed to load it. The saved model files may not load under different scikit-learn, LightGBM or imbalanced-learn versions.
