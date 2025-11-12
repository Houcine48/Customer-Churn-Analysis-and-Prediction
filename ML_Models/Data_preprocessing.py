import numpy as np
from DATA_importation import data_load
churn_data=data_load()

def data_pre(churn_data):
#transformation des variables au types numerique
    churn_data["Churn Label"]=churn_data["Churn Label"].map({"Yes":1,"No":0})
    churn_data["Intl Active"] = churn_data["Intl Active"].map({"Yes": 1, "No": 0})
    churn_data["Intl Plan"] = churn_data["Intl Plan"].map({"yes": 1, "no": 0})
    churn_data["Unlimited Data Plan"] = churn_data["Unlimited Data Plan"].map({"Yes": 1, "No": 0})
    churn_data["Gender"] = churn_data["Gender"].map({"Male": 1, "Female": 0,"Prefer not to say":2})
    churn_data["Device Protection & Online Backup"] = churn_data["Device Protection & Online Backup"].map({"Yes": 1, "No": 0})
    churn_data["Contract Type"] = churn_data["Contract Type"].map({"Month-to-Month": 0, "One Year": 1, "Two Year": 2})
    churn_data["Payment Method"] = churn_data["Payment Method"].map({"Paper Check": 0, "Credit Card": 1, "Direct Debit": 2})
    churn_data["Group"] = churn_data["Group"].map({"Yes": 1, "No": 0})

#cree un variable resumer et optimiser les deux colones
    conditions = [
    churn_data["Senior"] == "Yes",
    churn_data["Under 30"] == "Yes"
]

    choices = ["Senior", "Under 30"]

    churn_data["Demographics"] = np.select(conditions, choices, default="Other")
#transforme au numerique
    churn_data["Demographics"]=churn_data["Demographics"].map({'Other':1, 'Under 30':0 ,'Senior':2})

#remplis les nan en churn category
# Remplacer les NaN par "not detected" si Churn Label == 1
    churn_data.loc[
    churn_data["Churn Category"].isna() & (churn_data["Churn Label"] == 1),
    "Churn Category"
] = "not detected"

# Remplacer les NaN par "not churned" si Churn Label == 0
    churn_data.loc[
    churn_data["Churn Category"].isna() & (churn_data["Churn Label"] == 0),
    "Churn Category"
] = "not churned"

# Remplacer les NaN dans 'Churn Reason' par "not detected" si 'Churn Label' == 1
    churn_data.loc[
    churn_data["Churn Reason"].isna() & (churn_data["Churn Label"] == 1),
    "Churn Reason"
] = "not detected"

# Remplacer les NaN dans 'Churn Reason' par "not churned" si 'Churn Label' == 0
    churn_data.loc[
    churn_data["Churn Reason"].isna() & (churn_data["Churn Label"] == 0),
    "Churn Reason"
] = "not churned"

    churn_numeric = churn_data.select_dtypes(include=['int64', 'float64'])


    return churn_numeric




