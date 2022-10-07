import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error as mse, r2_score as r2s

archivo = pd.read_excel('BI_Alumnos07.xlsx', sheet_name='Hoja1')
datax = archivo[['Altura']]
x = np.array(datax)
y = archivo['Peso'].values

regL = lm.LinearRegression()
regL.fit(x, y)
y_pred = regL.predict(x)

print('Analisis de datos de BI_Alumnos07.xlsx')
print('Coeficiente de R: ', regL.coef_)
print('Termino independiente: ', regL.intercept_)
print('Error cuadrado medio: %.2f' % mse(y, y_pred))
print('Puntaje de varianza: %.2f' % r2s(y, y_pred))

predPeso = regL.predict([[180]])
print('Prediccion de peso de alumno de 180cm: ', predPeso)
