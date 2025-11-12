from sklearn.ensemble import  GradientBoostingClassifier
from  sklearn.model_selection import train_test_split
from DATA_importation import data_load
from Data_preprocessing import data_pre
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, log_loss)
churn_data=data_load()

churn_numeric =data_pre(churn_data)

def gbm_ml(churn_numeric):
    x = churn_numeric.drop("Churn Label", axis=1)
    y = churn_numeric["Churn Label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=42
    )

    # Grille d’hyperparamètres à tester
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.05],
        'max_depth': [3, 5],
        'subsample': [1.0, 0.8],
        'min_samples_leaf': [1, 3]
    }

    # Grid Search avec validation croisée
    gbm_model_gs = GridSearchCV(
        estimator=GradientBoostingClassifier(),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    gbm_model_gs.fit(x_train, y_train)

    best_model_gbm = gbm_model_gs.best_estimator_

    y_pred_gbm = best_model_gbm.predict(x_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred_gbm),
        "Recall": recall_score(y_test, y_pred_gbm),
        "F1 Score": f1_score(y_test, y_pred_gbm),
        "Precision": precision_score(y_test, y_pred_gbm),
        "roc_auc": roc_auc_score(y_test, y_pred_gbm),
        "log_loss": log_loss(y_test, y_pred_gbm),
    }

print(y_pred_gbm.head)