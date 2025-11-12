from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from DATA_importation import data_load
from Data_preprocessing import data_pre
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, log_loss)

churn_data=data_load()
churn_numeric=data_pre(churn_data)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

def rf_ml(churn_numeric):
    # Séparation des features et de la cible
    x = churn_numeric.drop("Churn Label", axis=1)
    y = churn_numeric["Churn Label"]

    # Split train/test avec stratification
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    # Grille d’hyperparamètres
    param_grid = {
        'n_estimators': [100, 200],               # Nombre d’arbres
        'max_depth': [None, 10],                  # Profondeur max
        'min_samples_split': [2, 5],              # Échantillons min pour split
        'min_samples_leaf': [1, 2]                # Échantillons min dans une feuille
    }

    # Modèle avec GridSearch
    rf_model = GridSearchCV(
        estimator=RandomForestClassifier(
            criterion="gini",
            max_features="sqrt",
            bootstrap=True,
            random_state=42
        ),
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1
    )

    # Entraînement
    rf_model.fit(x_train, y_train)

    # Meilleur modèle
    best_rf_par = rf_model.best_estimator_

    # Prédiction
    y_pred_rf = best_rf_par.predict(x_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Recall": recall_score(y_test, y_pred_rf),
        "F1 Score": f1_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf),
        "roc_auc": roc_auc_score(y_test, y_pred_rf),
        "log_loss": log_loss(y_test, y_pred_rf),
    }






