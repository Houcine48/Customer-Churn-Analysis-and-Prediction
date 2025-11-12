from Data_preprocessing import data_pre
from DATA_importation import data_load
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, log_loss)


churn_data=data_load()
churn_numeric =data_pre(churn_data)
def xgb_ml(churn_numeric):
    x=churn_numeric.drop("Churn Label",axis=1)
    y=churn_numeric["Churn Label"]
    print(x.shape)
    print(y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    scoring="f1"
    param_xgb={ 'n_estimators': [100, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, 3, 5] }


    xgb_mod=GridSearchCV(estimator=XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    random_state=42),
    param_grid=param_xgb,
    cv=5 ,
    scoring=scoring,
    verbose=1,
    n_jobs=-1 )


    xgb_mod=xgb_mod.fit(x_train,y_train)
    best_xgb_model = xgb_mod.best_estimator_

    y_xgb_pred = best_xgb_model.predict(x_test)

    return {
        "Accuracy": accuracy_score(y_test, y_xgb_pred),
        "Recall": recall_score(y_test, y_xgb_pred),
        "F1 Score": f1_score(y_test, y_xgb_pred),
        "Precision": precision_score(y_test, y_xgb_pred),
        "roc_auc": roc_auc_score(y_test, y_xgb_pred),
        "log_loss": log_loss(y_test, y_xgb_pred),
    }
print(y_xgb_pred.head)