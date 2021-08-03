import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data_base = pd.read_csv('Churn_Modelling.csv')

vari_indp = data_base.iloc[:,3:13].values

vari_dep = data_base.iloc[:,13].values


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import make_column_transformer

label_enc = LabelEncoder()

vari_indp [:,1]= LabelEncoder.fit_transform(vari_indp[:,1])

label_enc2 = LabelEncoder()

vari_indp [:,2] = label_enc2.fit_transform(vari_indp[:,2])
