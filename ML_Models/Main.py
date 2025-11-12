import pandas as pd
from DATA_importation import data_load
from Data_preprocessing import data_pre
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

# ⬇️ Importer tes fonctions ML
from Randomforest import rf_ml
from gbm import gbm_ml
from xgb import xgb_ml
from regressionlog import reg_log_ml

churn_data = data_load()
churn_numeric = data_pre(churn_data)


models = [
    ("Random Forest", rf_ml),
    ("Gradient Boosting", gbm_ml),
    ("XGBoost", xgb_ml),
    ("Logistic Regression", reg_log_ml)
]

# Stocker les résultats
results = []

for name, model_func in models:
    scores = model_func(churn_numeric)
    scores["Modèle"] = name
    results.append(scores)

# Convertir en DataFrame
result_df = pd.DataFrame(results)

# Réorganiser les colonnes
cols = ["Modèle", "Accuracy", "Recall", "F1 Score", "Precision", "roc_auc", "log_loss"]
result_df = result_df[cols]

# Afficher les résultats
print(result_df)

