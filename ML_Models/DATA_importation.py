import pandas as pd

#importation des donnees
def data_load():
    churn_data=pd.read_csv("C:/Users/user/OneDrive - ESPRIT/Bureau/DATA CAMP/power bi/case-study-analyzing-customer-churn/Datasets/Databel - Data.csv",
                       sep=",",decimal=".")
    return churn_data





