#!/usr/bin/env python
# coding: utf-8

# ¡Hola!
# 
# Mi nombre es Tonatiuh Cruz. Me complace revisar tu proyecto hoy.
# 
# Al identificar cualquier error inicialmente, simplemente los destacaré. Te animo a localizar y abordar los problemas de forma independiente como parte de tu preparación para un rol como data-scientist. En un entorno profesional, tu líder de equipo seguiría un enfoque similar. Si encuentras la tarea desafiante, proporcionaré una pista más específica en la próxima iteración.
# 
# Encontrarás mis comentarios a continuación - **por favor no los muevas, modifiques o elimines**.
# 
# Puedes encontrar mis comentarios en cajas verdes, amarillas o rojas como esta:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Éxito. Todo está hecho correctamente.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Observaciones. Algunas recomendaciones.
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Necesita corrección. El bloque requiere algunas correcciones. El trabajo no puede ser aceptado con comentarios en rojo.
# </div>
# 
# Puedes responderme utilizando esto:
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante.</b> <a class="tocSkip"></a>
# </div>

# <div class="alert alert-block alert-info">
# <b>Respuesta estudiante.</b> <a class="tocSkip"></a>
#     Hola, muchas gracias por revisar mi proyecto. En algunas secciones tuve dudas que no supe como resolver, me gustaria que me dieras feedback para saber como puedo solucionarlo ya que intente algunas formas que no resultaron. Nuevamente muchas gracias y estoy al pendiente de tu feedback. Bonito dia.
# </div>

# <div class="alert alert-block alert-warning">
# <b>Resumen de la revisión 1</b> <a class="tocSkip"></a>
# 
# Hola Ilse! Te hice algunas observaciones sobre la preparación de datos que son muy importantes para evitar los errores que me comentaste en la sección de entrenamiento del modelo. Los dos motivos principales que están causando estos problemas es la separación que hiciste de las fechas, debido a que aumentan mucho la memoria usada durante el entrenamiento y además estás usando OHE con la variable model, la cual causa una explosión en las dimensiones del DF. Te dejé comentarios al final de la sección de preparación de los datos para que puedas corregir estos detalles. No dudes en dejarme tus preguntas y comentarios en la siguiente iteración.
# </div>
# 

# <div class="alert alert-block alert-warning">
# <b>Resumen de la revisión 2</b> <a class="tocSkip"></a>
# 
# Te dejé un comentario para mejorar la forma en la que estás codificando los datos, lo cual puede ser el motivo de tus complicaciones durante el entramiento de los modelos. Si los problemas persisten házmelo saber y lo atiendo en la siguiente iteración.
# </div>

# <div class="alert alert-block alert-warning">
# <b>Resumen de la revisión 3</b> <a class="tocSkip"></a>
# 
# Excelente trabajo! Considera escribir una sugerencia final sobre qué modelo usar tomando en cuenta el tiempo de entrenamiento, el RMSE y algún otro criterio que consideres pertinente.
# </div>

# # Sprint 12 - Ilse Salinas
# # Descripción

# El servicio de venta de autos usados Rusty Bargain está desarrollando una aplicación para atraer nuevos clientes. Gracias a esa app, puedes averiguar rápidamente el valor de mercado de tu coche. Tienes acceso al historial: especificaciones técnicas, versiones de equipamiento y precios. Tienes que crear un modelo que determine el valor de mercado.
# A Rusty Bargain le interesa:
# - la calidad de la predicción;
# - la velocidad de la predicción;
# - el tiempo requerido para el entrenamiento

# ## Preparación de datos

# In[1]:


# importar liberias
import numpy as np
import pandas as pd
import re

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import time


# In[2]:


# leer dataframe e imprimir su informacion
df= pd.read_csv('/datasets/car_data.csv')
print('Informacion del dataframe: \n')
df.info()


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Cargaste las librerías y la información adecuadamente! Desde la tabla resultante de df.ifno() puedes observar que existen valores nulo y que las variables categóricas son de clase object.
# </div>
# 

# In[3]:


# imprimir muestra de dataframe
print(df.head())


# In[4]:


# definir valores nulos en dataframe
def rellenar_nulos(columna):
    if columna.dtype == 'object':
        return columna.fillna('indeterminado')
    elif columna.dtype in ['int64']:
        return columna.fillna(0)
    return columna

df =  df.apply(rellenar_nulos)
print(df.head())


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Excelente! Rellenar los valores nulos con algún valor que exprese explícitamente que no se cuenta con la información es la mejor manera de proceder en este caso. De esa forma evitas supuestos sobre la distribución de la información. La desventaja sería que le agregas una categoría más a las variables, pero es algo que se puede manejar bien en los siguientes incisos.
# </div>
# 

# In[5]:


# manejar y eliminar filas duplicadas del dataframe
def manejar_duplicados(df):
    # contar filas duplicadas
    duplicados_totales = df.duplicated().sum()
    print(f'Total de filas duplicadas: {duplicados_totales}')
    
    muestra_duplicados =  df[df.duplicated(keep = False)]
    print(f'Muestra de filas duplicadas: {muestra_duplicados.sample(3)}')
    print('\n')
    
    # elminar filas duplicadas
    sin_duplicados = df.drop_duplicates()
    print(f'Dataframe sin duplicados')
    
    return sin_duplicados

df = manejar_duplicados(df)


# In[6]:


# cambiar el nombre de las columnas para que sean legibles e imprimir dataframe final a ocupar
def insertar_guion_bajo(nombre):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', nombre).lower()
df.columns = [insertar_guion_bajo(col) for col in df.columns]


# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Noté que este código es algo tardado, recomiendo que antes de procesar de esta forma las fechas revises si esas columnas realmente aportan algo al modelo. Uno de los objetivos durante la preparación de los datos es dejar únicamente las columnas que tengan poder explicativo sobre la variable target, el precio. La variable de last seen, por ejemplo, no aportaría al modelo. Es necesario que revises qué otras variables no aportarían y que las elimines.
# 

# <div class="alert alert-block alert-info">
# <b>Respuesta estudiante.</b> <a class="tocSkip"></a>
#     Perfecto, muchas gracias por la observacion. En este caso estuve analizando las columnas que aportan y no a la variable objetivo del modelo y en este caso estaría descartando las siguientes: 'date_crawled','date_created','last_seen', 'number_of_pictures', 'postal_code'.
# </div>

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Excelente! Ninguna de las variables que eliminaste aportan a la predicción del precio de los autos.
# 
# </div>

# In[7]:


# revisar valores invalidos en las columnas
df = df[df['power'] > 0]
df = df[(df['registration_year'] >= 1900) & (df['registration_year'] <= 2024)]
df = df[(df['registration_month'] >= 1) & (df['registration_month'] <= 12)]

# eliminar las columnas que no aportan a la variable objetivo 'price' para el modelo de ml
features_eliminar = ['date_crawled', 'date_created', 'last_seen', 'number_of_pictures', 'postal_code']
df_limpio = df.drop(columns=features_eliminar)

# aplicar el one-hot encoding en columnas categoricas, excepto 'model'
columnas_categoricas_sin_model = ['vehicle_type', 'gearbox', 'fuel_type', 'brand', 'not_repaired']
df_sin_model = df_limpio.drop(columns=['model'])  # Usar df_limpio en lugar de df
df_ohe = pd.get_dummies(df_sin_model, columns=columnas_categoricas_sin_model, drop_first=True)

# aplicar ordinal encoding a la variable 'model'
columnas_categoricas = ['model', 'vehicle_type', 'gearbox', 'fuel_type', 'brand', 'not_repaired']
df_ordinal_encoding = df_limpio
df_ordinal_encoding[columnas_categoricas] = OrdinalEncoder().fit_transform(df_ordinal_encoding[columnas_categoricas])


# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Es correcto aplicar OHE encoding a tus variables categóricas ya que la mayoría de modelos de ML no reconocer los valores categóricos en texto; sin embargo, mi recomendación es que para el OHE excluyas la variable model, ya que esta al tener muchas categorías vuelve muy lento el entrenamiento de los modelos. Pero debido a que es una variable muy importante, sugiero que la utilices con Ordinal Encoding. Entonces los pasos serían:
#     
#     1.- Crear una tabla nueva a partir del df original que no contenga model y sobre esa aplicar OHE.
#     2.- Crear otra tabla a partir del df original que contenga model y sobre ella aplicar Ordinal Encoding.
#     3.- Eliminar los features o variables que no aportarían a la predicción del precio, como last seen, date created, pictures y las que tú consideres.
#     4.- Revisar valores inválidos que puedan tener algunas variables, power por ejemplo tiene valores 0, lo cual es imposible. El año de registro también podría tener valores inválidos. Explora los features con los que te quedes para asegurar que todo esté en orden.
#     5.- La única variable de fecha clave en la predicción del precio es RegistrationYear y RegistrationMonth, quédate con ambas variables. 
# 
# (comentario antiguo)

# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Me parece que hubo una pequeña confusión con respecto a las codificaciones. En tu bloque de código de arriba estás aplicando correctamente el OHE excluyendo la columna model para evitar dimensiones muy grandes de tu tabla. Esta tabla con OHE ya la puedes usar para el entrenamiento del modelo. Ordinal Encoding es una codificación que se debe hacer a todas las variables categóricas y debe ser una tabla diferente a la de OHE. Te dejo un ejemplo de cómo reescribir tu bloque de código:
#     
#     # Es preferible eliminar los valores inválidos antes de codificar, de esta forma solo es necesario hacerlo una vez
#     
#     # revisar valores invalidos en las columnas
#     df = df[df['power'] > 0]
#     df = df[(df['registration_year'] >= 1900) & (df['registration_year'] <= 2024)]
#     df = df[(df['registration_month'] >= 1) & (df['registration_month'] <= 12)]
# 
#     # eliminar las columnas que no aportan a la variable objetivo 'price' para el modelo de ml
#     features_eliminar = ['date_crawled', 'date_created', 'last_seen', 'number_of_pictures', 'postal_code']
#     df_limpio = df.drop(columns=features_eliminar)
# 
#     # Aplicar el one-hot encoding en columnas categoricas, excepto 'model'
#     columnas_categoricas_sin_model = ['vehicle_type', 'gearbox', 'fuel_type', 'brand', 'not_repaired']
#     df_sin_model = df_limpio.drop(columns=['model'])  # Usar df_limpio en lugar de df
#     df_ohe = pd.get_dummies(df_sin_model, columns=columnas_categoricas_sin_model, drop_first=True)
#                                                                               
#     # Aplicar Ordinal Encoding
#     columnas_categoricas = ['model', 'vehicle_type', 'gearbox', 'fuel_type', 'brand', 'not_repaired']
#     df_ordinal_encoding = df_limpio
#     df_ordinal_encoding[columnas_categoricas] = OrdinalEncoder().fit_transform(df_ordinal_encoding[columnas_categoricas])
# 
#                                                                               
# Algunas consideraciones que hay que tomar en cuenta con estas codificaciones es que usar una de las dos o ambas depende del modelo. Por ejemplo para la regresión lineal o regresión logística no se puede utilizar ordinal encoding, ya que este método convierte los textos de las categorías en un número y estos modelos asignan una relación lineal entre ellos, lo cual no es correcto y por lo tanto en estos modelos es obligatorio usar OHE a menos que se cambie la manera en la que se asignan valores a las categorías. Por otro lado, para los modelos como Random Forest o Árbol de decisión es preferible usar Ordinal Encoding ya que estos modelos pueden manejar adecuadamente las variables categóricas. Por último hay modelos como CatBoost para los que no es necesario codificar (a menos que se desee mejorar el rendimiento, como es nuestro caso) ya que puede interpretar las categorías de texto adecuadamente.

# <div class="alert alert-block alert-info">
# <b>Respuesta estudiante.</b> <a class="tocSkip"></a>
#     Vale ya me quedo mas claro, muchas gracias por la observacion y la correccion. Anotare lo que me mencionaste en mis notas sobre el tipo de codificacion segun el modelo para no olvidarlo.
# </div>

# ## Entrenamiento del modelo 

# In[8]:


# dividir el dataframe en caracteristicas y objetivo
# se ocuparan estas caracteristicas y objetivo para la regresion lineal 
features_1 = df_ohe.drop('price', axis = 1)
target_1 = df_ohe['price'] 

# se ocuparan estas caracteristicas y objetivo para el arbol de dessicion y bosque aleatorio
features_2 = df_ordinal_encoding.drop('price', axis = 1)
target_2 = df_ordinal_encoding['price'] 


# In[9]:


# dividir el dataframe en conjunto de entrenamiento y prueba - regresion lineal
features_train_1, features_test_1, target_train_1, target_test_1 = train_test_split(features_1, target_1, test_size= 0.2, random_state = 12345)

print("Tamaño de features_train:", features_train_1.shape)
print("Tamaño de features_test:", features_test_1.shape)
print("Tamaño de target_train:", target_train_1.shape)
print("Tamaño de target_test:", target_test_1.shape)
print("\n")

# dividir el dataframe en conjunto de entrenamiento y prueba - arbol y bosque
features_train_2, features_test_2, target_train_2, target_test_2 = train_test_split(features_2, target_2, test_size= 0.2, random_state = 12345)

print("Tamaño de features_train:", features_train_2.shape)
print("Tamaño de features_test:", features_test_2.shape)
print("Tamaño de target_train:", target_train_2.shape)
print("Tamaño de target_test:", target_test_2.shape)


# In[10]:


# funcion para evaluar los modelos donde se analizara la calidad, velocidad y tiempo
def evaluate_model(model, features_train, target_train, features_test, target_test):
    start_time = time.time()
    model.fit(features_train, target_train)
    train_time = time.time() - start_time
    
    target_pred_train = model.predict(features_train)
    target_pred_test = model.predict(features_test)
    
    rmse_train = np.sqrt(mean_squared_error(target_train, target_pred_train))
    rmse_test = np.sqrt(mean_squared_error(target_test, target_pred_test))
    
    return rmse_train, rmse_test, train_time


# ### Modelo de arbol de desicion

# In[11]:


tree_model = DecisionTreeRegressor(random_state = 42)
rmse_train_tree, rmse_test_tree, train_time_tree = evaluate_model(tree_model, features_train_2, target_train_2, features_test_2, target_test_2)


# In[12]:


# imprimir los resultados
print("RMSE en entrenamiento:", rmse_train_tree)
print("RMSE en prueba:", rmse_test_tree)
print("Tiempo de entrenamiento:", train_time_tree)


# ### Modelo de bosque aleatorio

# <div class="alert alert-block alert-info">
# <b>Respuesta estudiante.</b> <a class="tocSkip"></a>
#     Hola necesito ayuda en esta seccion. Intente implementar tanto GridSearchCV como RandomizedSearchCV para elegir los mejores hiperparametros para el modelo de bosque aleatorio pero se tarda bastante tiempo en definirlos. Necesito alguna recomendacion para que seea mas rapido ya que quiero comparar los tiempos con los otros modelos. Definitivamenete la ejecucion de este es la mas lenta pero tal vez la mas precisa en cuanto a las metricas de precision y asi. Estuve investigando que para mejorar un modelo de bosque aleatorio, puedo combinarlo con técnicas de boosting en un enfoque de Stacking pero me gustaria saber si es la mejor solucion.
# </div>

# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Recomendaría limitar el rango de los hiperparámetros para explorar menos combinaciones y mejorar tus tiempos:
#     
#     param_distributions = {
#     'n_estimators': [100, 200],
#     'max_depth': [10, 20],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2],
#     'bootstrap': [True]
#     }
# 
# En caso de que siga siendo muy tardado también puedes probar disminuyendo el número de iteraciones del RandomizedSearchCV a 20.
# </div>

# <div class="alert alert-block alert-info">
# <b>Respuesta estudiante.</b> <a class="tocSkip"></a>
#     Al final lo deje en 10 en vez de 20 iteraciones porque se tardaba mucho en elegir los mejores hiperparametros. Lo bueno es que estuve checando mas o menos el tiempo empleado en cada iteracion dependiendo de los hiperparameetros escogidos, es por eso que lo modifique a ssolo 10. 
# </div>

# In[13]:


# definir hiperparametros para el modelo - de estos se escogera el mas optimo
param_distributions = {
'n_estimators': [100, 200],
'max_depth': [10, 20],
'min_samples_split': [2, 5],
'min_samples_leaf': [1, 2],
'bootstrap': [True]
}

forest_model = RandomForestRegressor(random_state=42)

# crear el RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=forest_model, param_distributions=param_distributions,
                                    n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# ejecutar la busqueda
start_time = time.time()
random_search.fit(features_train_2, target_train_2)
search_time = time.time() - start_time

# obtener los mejores parametros
best_params = random_search.best_params_
print(f"Mejores parámetros: {best_params}")

# evaluar el mejor modelo encontrado
best_random_forest = random_search.best_estimator_


# In[14]:


rmse_train_forest, rmse_test_forest, train_time_forest = evaluate_model(best_random_forest, features_train_2, target_train_2, features_test_2, target_test_2)


# In[15]:


# imprimir los resultados
print("RMSE en entrenamiento:", rmse_train_forest)
print("RMSE en prueba:", rmse_test_forest)
print("Tiempo de entrenamiento:", train_time_forest)


# ### Modelo de regresion lineal

# In[16]:


linear_model = LinearRegression()
rmse_train_linear, rmse_test_linear, train_time_linear = evaluate_model(linear_model, features_train_1, target_train_1, features_test_1, target_test_1)


# In[17]:


# imprimir los resultados
print("RMSE en entrenamiento:", rmse_train_linear)
print("RMSE en prueba:", rmse_test_linear)
print("Tiempo de entrenamiento:", train_time_linear)


# #### Prueba de cordura

# In[18]:


# sirve para asegurarme que la potenciacion esta funcionando correctamente
boosting_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
rmse_train_boosting, rmse_test_boosting, train_time_boosting = evaluate_model(boosting_model, features_train_1, target_train_1, features_test_1, target_test_1)


# In[19]:


print("Boosting - RMSE en entrenamiento:", rmse_train_boosting)
print("Boosting - RMSE en prueba:", rmse_test_boosting)
print("Boosting - Tiempo de entrenamiento:", train_time_boosting)


# Al comparar ambos resultados, es decir, tanto el del modelo lineal como el modelo de potenciacion, siendo éste parte de la prueba de cordura para verificar que si esta funcionando correctamente, los valores de RMSE son mas bajos que los originales pero el tiempo de entrenamiento es significantemente mayor. Se sacrifica el tiempo por mayor precisión y felxibilidad en las métricas.

# #### Descenso de gradiente estocástico

# <div class="alert alert-block alert-info">
# <b>Respuesta estudiante.</b> <a class="tocSkip"></a>
#    Aqui creo que funciona peor la aplicacion del gradiente en la regresion lineal que la misma regresion. No estoy muy segura de que podria hacer para que fueran mejor los tiempos y resultados que la regresion lineal original ya que hice la prueba de cordura y salio bien pero no se si estoy pasando por alto algun detalle.   
# </div>

# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# El modelo puede estar siendo afectado debido a la codificación que usaste en tu iteración anterior. Prueba a implementar los cambios que te sugerí y en caso de que persista me lo puedes volver a comentar.
# </div>

# In[20]:


# escalar las caracteristicas
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train_1)
features_test_scaled = scaler.transform(features_test_1)


# In[31]:


# modelo de regresion lineal con aplicacion del gradiente estocastico - ayuda a encontrar los valores optimos de los coeficientes que minimicen el error entre las predicciones 
# calcular el gradiente usando pequeñas partes del conjunto de entrenamiento (minilotes/lotes)
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='optimal')
rmse_train_sgd, rmse_test_sgd, train_time_sgd = evaluate_model(sgd_regressor, features_train_scaled, target_train_1, features_test_scaled, target_test_1)


# In[32]:


# imprimir los resultados
print("RMSE en entrenamiento:", rmse_train_sgd)
print("RMSE en prueba:", rmse_test_sgd)
print("Tiempo de entrenamiento:", train_time_sgd)


# ### Tecnicas de Potenciacion del gradiente

# <div class="alert alert-block alert-info">
# <b>Respuesta estudiante.</b> <a class="tocSkip"></a>
#    Aqui no entiendo muy bien porque Catboost y XGBoost se tardan bastante en procesar. Creo que resolviendo el problema del descenso del gradiente de la regresion lineal estos funcionaran correctamente.   
# </div>

# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Lo ideal para el entrenamiento de estos modelos es Ordinal Encoding debido a que OHE aumenta considerablemente el tiempo de entrenamiento. Prueba a realizar los cambios sugeridos sobre la codificación y en caso de que persista me lo comentas y lo atiendo en la siguiente iteración.
# </div>

# #### LightGBM 

# In[23]:


lgb_model = lgb.LGBMRegressor(random_state=42)
param_grid_lgb = {'num_leaves': [31, 50], 'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]}
lgb_search = GridSearchCV(lgb_model, param_grid_lgb, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
rmse_train_lgb, rmse_test_lgb, train_time_lgb = evaluate_model(lgb_search, features_train_2, target_train_2, features_test_2, target_test_2)


# In[24]:


# imprimir los resultados
print("RMSE en entrenamiento:", rmse_train_lgb)
print("RMSE en prueba:", rmse_test_lgb)
print("Tiempo de entrenamiento:", train_time_lgb)


# #### Catboost 

# In[25]:


cb_model = cb.CatBoostRegressor(random_seed=42, silent=True)
param_grid_cb = {'depth': [6, 10], 'iterations': [100, 200], 'learning_rate': [0.1, 0.05]}
cb_search = GridSearchCV(cb_model, param_grid_cb, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
rmse_train_cb, rmse_test_cb, train_time_cb = evaluate_model(cb_search, features_train_2, target_train_2, features_test_2, target_test_2)


# In[26]:


# imprimir los resultados
print("RMSE en entrenamiento:", rmse_train_cb)
print("RMSE en prueba:", rmse_test_cb)
print("Tiempo de entrenamiento:", train_time_cb)


# #### XGBoost

# In[27]:


xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
param_grid_xgb = {'max_depth': [3, 6], 'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]}
xgb_search = GridSearchCV(xgb_model, param_grid_xgb, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
rmse_train_xgb, rmse_test_xgb, train_time_xgb = evaluate_model(xgb_search, features_train_2, target_train_2, features_test_2, target_test_2)


# In[28]:


# imprimir los resultados
print("RMSE en entrenamiento:", rmse_train_xgb)
print("RMSE en prueba:", rmse_test_xgb)
print("Tiempo de entrenamiento:", train_time_xgb)


# ## Análisis del modelo

# In[33]:


# mostar resultados
resultados = pd.DataFrame({
    'Modelo': ['Regresion Lineal', 'SDG Gradiente Estocastico','Arbol de Desicion', 'Bosque Aleatorio', 'LightGBM', 'CatBoost', 'XGBoost'],
    'RMSE Train': [rmse_train_linear, rmse_train_sgd, rmse_train_tree, rmse_train_forest, rmse_train_lgb, rmse_train_cb, rmse_train_xgb],
    'RMSE Test': [rmse_test_linear, rmse_test_sgd, rmse_test_tree, rmse_test_forest, rmse_test_lgb, rmse_test_cb, rmse_test_xgb],
    'Train Time (s)': [train_time_linear, train_time_sgd, train_time_tree, train_time_forest, train_time_lgb, train_time_cb, train_time_xgb]
})


# In[34]:


print(resultados)


# # Lista de control

# Escribe 'x' para verificar. Luego presiona Shift+Enter

# - [x]  Jupyter Notebook está abierto
# - [ ]  El código no tiene errores- [ ]  Las celdas con el código han sido colocadas en orden de ejecución- [ ]  Los datos han sido descargados y preparados- [ ]  Los modelos han sido entrenados
# - [ ]  Se realizó el análisis de velocidad y calidad de los modelos

# In[ ]:




