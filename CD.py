#importar las librerias
import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

train = r'c:\Users\Pablish\OneDrive\Desktop\New folder\train.csv'
test = r'c:\Users\Pablish\OneDrive\Desktop\New folder\test.csv'

#importamos los datos de los archivos 
df_train = pd.read_csv(train)
df_test = pd.read_csv(test)


#verificamos de manera rapida los datos de los archivos
print(df_test.head())
print(df_train.head())

#verificamos la cantidad de datos que hay en la database
print ("\n\ncantidad de datos: ")
print (df_train.shape)
print (df_test.shape)

#verificamos los tipos de datos contenido en ambos database
print ("\n\ntipos de datos: ")
print (df_train.info())
print (df_test.info())

#verificar los datos faltantes en las dabases
print ("\n\ndatos faltantes: ")
print (pd.isnull(df_train).sum())
print (pd.isnull(df_test).sum())

#verificamos las estadisticas de cada database
print ("\n\nestadisticas del database: ")
print (df_train.describe())
print (df_test.describe())

#____________PRE-PROCESAMIENTO DE DATOS____________

#cambiar los datos de sexo en numeros
df_train['Sex'].replace(['female','male'],[0,1],inplace=True)
df_test['Sex'].replace(['female','male'],[0,1],inplace=True) 

#cambiar los datos de embarque a numeros
df_test['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)
df_train['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)

#remplazamos los datos faltantes en la edad por la media de estas columnas
print ("\n\n",df_train['Age'].mean())
print (df_test['Age'].mean())
promedio = 30
df_train['Age'] = df_train['Age'].replace(np.nan, promedio)
df_test['Age'] = df_test['Age'].replace(np.nan, promedio)

#crear varios grupos de acuerdo a bandas de las edades
#bandas: 0-8, 9-15, 16-18, 19-25, 26-40, 41-60, 61-100
bins = [0, 8, 15, 18, 25, 40, 60, 100]
names = ['1', '2', '3', '4', '5', '6', '7']
df_train['Age'] = pd.cut(df_train['Age'], bins, labels= names)
df_test['Age'] = pd.cut(df_test['Age'], bins, labels= names)

#eliminamos la columna "cabin" ya que tiene muchos datos perdidos y puede afectar el analisis 
df_train.drop(['Cabin'],axis = 1, inplace=True)
df_test.drop(['Cabin'],axis = 1, inplace=True)

#eliminamos las columnas que se consideran inecesarias para el analisis
df_train = df_train.drop(['PassengerId','Name','Ticket'], axis=1)
df_test = df_test.drop(['Name','Ticket'], axis=1)

#se eliminan las filas con los datos perdidos
df_train.dropna(axis=0, how='any',inplace=True)
df_test.dropna(axis=0, how='any',inplace=True)

#verificar los datos
print("\n\nverificar datos")
print (pd.isnull(df_train).sum())
print (pd.isnull(df_test).sum())

print (df_train.shape)
print (df_test.shape)

print(df_test.head())
print(df_train.head())

#____________aplicamos el algoritmo de machine learning____________

#separamos las columnas con la informacion de los sobrevivientes
X = np.array(df_train.drop(['Survived'], axis=1))
Y = np.array(df_train['Survived'])

#separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#regresion logistica
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print ("\n\nPrecision regresion logistica: ")
print (logreg.score(X_train, Y_train))

#support vector machine 
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
print("\n\nprecision soporte de vectores: ")
print(svc.score(X_train, Y_train))

#K neighbors
Knn = KNeighborsClassifier(n_neighbors = 3)
Knn.fit(X_train, Y_train)
print("\n\nprecision vectores mas cercanos")
print(Knn.score(X_train, Y_train))

#________________ prediccion utilizando los modelos_______________
ids = df_test['PassengerId']

#regresion logistica
prediccion_logreg = logreg.predict(df_test.drop('PassengerId', axis=1).values)
out_logreg = pd.DataFrame({'PassengerId' : ids, 'survived' : prediccion_logreg})

print ("\nprediccion regresion logistica: ")
print(out_logreg.head())

#support vector machines 
prediccion_SVC = svc.predict(df_test.drop('PassengerId', axis=1))
out_svc = pd.DataFrame({'PasseengerId' : ids, 'Survived' : prediccion_SVC})
print ("\nprediccion soporte de vectores: ")
print(out_svc.head())

#K neighbors
prediccion_Knn = Knn.predict(df_test.drop('PassengerId', axis=1))
out_Knn = pd.DataFrame({'PasseengerId' : ids, 'Survived' : prediccion_Knn})
print ("\nprediccion vecinos mas cercanos: ")
print(out_Knn.head())