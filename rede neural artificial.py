import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

        # tratamento dos dados para o treino

data_base = pd.read_csv('Churn_Modelling.csv')

vari_indp = data_base.iloc[:,3:13].values

vari_dep = data_base.iloc[:,13].values


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import make_column_transformer

label_enc = LabelEncoder()

vari_indp [:,1]= label_enc.fit_transform(vari_indp[:,1])

label_enc2 = LabelEncoder()

vari_indp [:,2]= label_enc2.fit_transform(vari_indp[:,2])

onehot=make_column_transformer((OneHotEncoder(categories='auto', sparse = False), [1]), remainder="passthrough")

vari_indp = onehot.fit_transform(vari_indp)

vari_indp = vari_indp[:,1:]



            # treino
from sklearn.model_selection import train_test_split


x_treino, x_teste, y_treino,y_teste = train_test_split(vari_indp,
 vari_dep,test_size=0.25,random_state= 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_treino = sc.fit_transform(x_treino)
x_teste = sc.fit_transform(x_teste)


            # construção da rede neural
            
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 6, kernel_initializer= 'uniform', activation='relu',input_dim=11))
               
classifier.add(Dense(units = 6, kernel_initializer= 'uniform', activation='relu'))

classifier.add(Dense(units = 1, kernel_initializer= 'uniform', activation='sigmoid'))



        #treino da rede neural
        
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'] )

classifier.fit(x_treino,y_treino ,batch_size=10, epochs=100)


pred = classifier.predict(x_teste)

pred =(pred > 0.5)


from sklearn.metrics import confusion_matrix

matriz_conf = confusion_matrix(y_teste,pred)
