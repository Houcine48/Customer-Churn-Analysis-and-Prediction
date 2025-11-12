from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from DATA_importation import data_load
from Data_preprocessing import data_pre
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, log_loss)


churn_data=data_load()
churn_numeric = data_pre(churn_data)

# Fonction ML avec régression logistique
def reg_log_ml(churn_numeric):
    X = churn_numeric.drop("Churn Label", axis=1)
    y = churn_numeric["Churn Label"]

    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Grille d'hyperparamètres
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear'],
        'max_iter': [100, 200],
        'class_weight': [None, 'balanced']
    }

    # Entraînement avec GridSearchCV
    reglog_model = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    reglog_model.fit(X_train_scaled, y_train)

    best_rl_mod = reglog_model.best_estimator_
    y_pred_rl = best_rl_mod.predict(X_test_scaled)


    return {
        "Accuracy": accuracy_score(y_test, y_pred_rl),
        "Recall": recall_score(y_test, y_pred_rl),
        "F1 Score": f1_score(y_test, y_pred_rl),
        "Precision": precision_score(y_test, y_pred_rl),
        "roc_auc": roc_auc_score(y_test, y_pred_rl),
        "log_loss": log_loss(y_test, y_pred_rl),
    }