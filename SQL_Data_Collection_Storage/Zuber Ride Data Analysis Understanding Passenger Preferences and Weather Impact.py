#!/usr/bin/env python
# coding: utf-8

# # ¡Hola Ilse!
# 
# Mi nombre es Ezequiel Ferrario, soy code reviewer en Tripleten y tengo el agrado de revisar el proyecto que entregaste.
# 
# Para simular la dinámica de un ambiente de trabajo, si veo algún error, en primer instancia solo los señalaré, dándote la oportunidad de encontrarlos y corregirlos por tu cuenta. En un trabajo real, el líder de tu equipo hará una dinámica similar. En caso de que no puedas resolver la tarea, te daré una información más precisa en la próxima revisión.
# 
# Encontrarás mis comentarios más abajo - **por favor, no los muevas, no los modifiques ni los borres**.
# 
# ¿Cómo lo voy a hacer? Voy a leer detenidamente cada una de las implementaciones que has llevado a cabo para cumplir con lo solicitado. Verás los comentarios de esta forma:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Si todo está perfecto.
# </div>
# 
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta. Se aceptan uno o dos comentarios de este tipo en el borrador, pero si hay más, deberá hacer las correcciones. Es como una tarea de prueba al solicitar un trabajo: muchos pequeños errores pueden hacer que un candidato sea rechazado.
# </div>
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# Puedes responderme de esta forma:
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# 
# Hola, muchas gracias por tus comentarios y la revisión.
# </div>
# 
# ¡Empecemos!

# -----------------
# 
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario general #1</b> <a class="tocSkip"></a>
# 
# Ilse, has realizado un proyecto completo donde abarcaste todos los puntos importantes.
# 
# 
# A nivel codigo estuviste muy bien, te manejaste perfecto con los graficos y las pruebas.
# 
# La mayoria de correcciones van por el lado de la estructura del proyecto.
# 
# Quedo a la espera de tus correcciones, saludos.</div>
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario general #2</b> <a class="tocSkip"></a>
# 
# Ilse, realizaste unas excelentes correcciones por lo que el proyecto esta practicamente hecho.
# 
# Me gusto mucho las descripciones de los datasets y como te manejaste con ellos.
# 
# Restan dos detalles por corregir
# 
# - Separar los info() de la celda de carga
# 
# - Realizar la prueba testudent con el parametro equal_Var en true.
# 
# Con eso el proyecto quedara aprobado.
# 
# Espero tus correcciones, saludos.</div>
# 
# -----------------

# # Proyecto Sprint 7 (Continuación)
# De: Ilse Natalia Salinas Vázquez
# 
# ## Introducción
# ### Descripción del proyecto
# Se está trabajando como analista para Zuber, una nueva empresa de viajes compartidos que se está lanzando en Chicago. La tarea principal es encontrar patrones en la información disponible. Es importante comprender las preferencias de los pasajeros y el impacto de los factores externos en los viajes.
# Al trabajar con una base de datos, se analizará los datos de los competidores y probarás una hipótesis sobre el impacto del clima en la frecuencia de los viajes.
# 
# Anteriormente se estuvieron trabajando las bases de datos junto con sus ejercicios correspondientes para poder hacer este siguiente análisis de datos con Python. En este siguiente paso se estarán extrayendo bases de datos previamente manejadas parta obtener algunas conclusiones e hipótesis generales.
# 
# # Paso 4. Análisis exploratorio de datos (Python)
# 
# ### Descripción general
# Además de los datos que recuperaste en las tareas anteriores te han dado un segundo archivo. Ahora tienes estos dos CSV:
# 
# /datasets/project_sql_result_01.csv. contiene los siguientes datos:
# 
#     company_name: nombre de la empresa de taxis
# 
#     trips_amount: el número de viajes de cada compañía de taxis el 15 y 16 de noviembre de 2017. 
# 
# /datasets/project_sql_result_04.csv. contiene los siguientes datos:
# 
#     dropoff_location_name: barrios de Chicago donde finalizaron los viajes
# 
#     average_trips: el promedio de viajes que terminaron en cada barrio en noviembre de 2017.
# 
# Para estos dos datasets ahora necesitas:
# 1. Importar los archivos.
# 2. Estudiar los datos que contienen.
# 3. Asegurarte de que los tipos de datos sean correctos.
# 4. Identificar los 10 principales barrios en términos de finalización del recorrido.
# 5. Hacer gráficos: empresas de taxis y número de viajes, los 10 barrios principales por número de finalizaciones.
# 6. Sacar conclusiones basadas en cada gráfico y explicar los resultados.

# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Hace una tabla de contenidos que este  linkeada a las secciones (al clickear debe llevarnos a esa seccion) de esta menera es mas facil desplazarse.
# 
# Como consejo, si realizas bien todas las secciones (con su respectivo #) podes generarlo automáticamente desde jupyter lab. Para hacerlo, en la pestaña de herramientas de jupyter lab clickeas en el **botón de los puntos y barras**  (Table of contents) te generara automáticamente una tabla de contenidos linkeable y estética. A la **derecha** del botón "Validate"
# </div>
#                                                     
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Ilse, recorda realizar una introduccion al proyecto, explicando los objetivos y explicando de que se trata. Esto es muy importante de hacer para asentar las bases del proyecto, ademas de dejar en claro que se busca. A partir de una mirada a la introduccion ya entenderemos que buscamos con el proyecto.</div>
# 
# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor #2</b> <a class="tocSkip"></a>
# 
# 
# Corregido.
# </div>

# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Crea una seccion (Titulo) que se llame Carga de datasets e importacion de librerias (o el titulo que quieras pero que refiera a este punto). Es importante a nivel estructura que quede claro que parte se realizan ambas cosas.</div>
# 
# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor #2</b> <a class="tocSkip"></a>
# 
# 
# Corregido.
# </div>

# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Siguiendo las buenas practicas, deberias  dividir las celdas de importacion por un lado y por otro la de carga de datasets.</div>
# 
# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor #2</b> <a class="tocSkip"></a>
# 
# 
# Corregido.
# </div>

# ## Carga de datasets e importación de librerías
# ### Preprocesamiento y análisis de los datos

# In[1]:


# Aqui se importan las librerías que se van a estar utilizando a lo largo del proyecto de esta sección
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[2]:


# Aquí se lee y se muestra la información cargada de bases de datos en los dataframes
# Se cargan todos los datasets, tanto de la seccion 1 como de la seccion 2
print("Información del DataFrame df_sql_1:")
df_sql_1= pd.read_csv('/datasets/project_sql_result_01.csv')
df_sql_1.info()
print()


# In[3]:


print("Información del DataFrame df_sql_4:")
df_sql_4= pd.read_csv('/datasets/project_sql_result_04.csv')
df_sql_4.info()
print()


# In[4]:


# Este aun no se carga porque no es relevante en esta seccion del proyecto pero en la segunda si 
df_sql_7= pd.read_csv('/datasets/project_sql_result_07.csv')


# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Crea una celda y dedicala exclusivamente a la carga de datasets, por lo que debes cargar **todos** los datasets existentes en esa celda. Esto debido a que constituye una buena practica, ya que al comienzo del notebook ya estaria todo ejecutado y ante cualquier cambio lo tenemos de facil acceso.</div>
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor #2</b> <a class="tocSkip"></a>
# 
# 
# En este caso se mantiene la correccion. Deja libre la celda de importacio nsin utilziar los metodos info() en la misma. Si es necesario que utilices los metodos luego.</div>

# ----------------------
# 
# 
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Explora las primeras filas con el head()</div>
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Recorda utilizar el metodo describe() para una exploracion rapida inicial de aquellas variables numéricas. Siempre es necesario realizarlo ya que de forma rápida tenemos un panorama muy bueno de que nos espera e incluso encontraremos inconsistencias si estas existiencen.
# Describí al respecto lo que ves. </div>
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Verifica si existen duplicados </div>
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Deja una descripcion de lo observado en este dataset en cuanto a como se compone, si hay errores o no  (nulos, duplicados, etc) y que informacion nos provee. Este punto es muy importante porque resumiremos con un analisis en **markdown** sobre como esta compuesto el dataset y sus variables mas importante. </div>
# 
# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor #2</b> <a class="tocSkip"></a>
# 
# 
# Corregido todos los puntos. Excelente.
# </div>

# In[5]:


# Aqui se explora la informacion de cada dataframe con la instruccion head()
print('Informacion df_sql_1')
print(df_sql_1.head())
print()

print('Informacion df_sql_4')
print(df_sql_4.head())


# In[6]:


# Aqui se explora la descripcion de los datos con el metodo describe() en cada dataframe
print('Descripcion de datos df_sql_1')
print(df_sql_1.describe())
print()

print('Descripcion de datos df_sql_4')
print(df_sql_4.describe())


# In[7]:


# Igual que en la primera seccion verificaremos si hay valores duplicados en este dataframe
duplicates = df_sql_1.duplicated()

# Contar el número de duplicados
num_duplicates = duplicates.sum()

# Imprimir el resultado
if num_duplicates == 0:
    print("No hay información duplicada en el DataFrame.")
else:
    print(f"Hay {num_duplicates} filas duplicadas en el DataFrame.")


# In[8]:


# Igual que en la primera seccion verificaremos si hay valores duplicados en este dataframe
duplicates = df_sql_4.duplicated()

# Contar el número de duplicados
num_duplicates = duplicates.sum()

# Imprimir el resultado
if num_duplicates == 0:
    print("No hay información duplicada en el DataFrame.")
else:
    print(f"Hay {num_duplicates} filas duplicadas en el DataFrame.")


# De acuerdo con la información desplegada en las secciones de arriba, se puede observar que la información del dataset está ordenada de forma descendente, es decir, el número de viajes realizados y si promedio va disminuyendo conforme la compañía y tambien la región final del viaje. Esto nos puede dar una idea de como puede distribuirse las gráficas posteriores a realizar junto con la información sobre le media, mínimo, máximo y quantiles.
# 
# Además de que al parecer con la información presentada en cada dataframe, no hay valores nulos pero puede darse el caso en el que esté repetido el nombre de la compañía de taxis o que también en caso de que alguna de estas compañías no haya registrado algún viaje, es por eso que el siguiente código es para verificar esto y excluirlo del análisis y gráficas posteriores.

# In[9]:


# Se cuenta el numero de valores unicos en la columna company_name
num_unique = df_sql_1['company_name'].nunique()

# Se obtiene el numero total de filas no nulas en la columna company_name
num_non_null = df_sql_1['company_name'].notnull().sum()

# Se verifica si no hay valores repetidos
if num_unique == num_non_null:
    print("No hay valores repetidos en la columna company_name.")
else:
    print("Hay valores repetidos en la columna company_name.")
    
# Ahora se filtrar el DataFrame para obtener las compañias con cero viajes
companies_with_zero_trips = df_sql_1[df_sql_1['trips_amount'] == 0]

# Se cuenta el numero de compañias con cero viajes
num_companies_with_zero_trips = len(companies_with_zero_trips)
print("Número de compañías con cero viajes:", num_companies_with_zero_trips)

# En caso de que haya alguna, se eliminan estas compañias del DataFrame
df_sql_1_filtered = df_sql_1[df_sql_1['trips_amount'] != 0]


# Con los resultados anteriores, se puede estar seguro que los datos son idóneos para el análisis ya que los tipos de datos corresponden para cada una de las columnas, no existen valores nulos o repetidos y tambien existen viajes realizados por las compañías. De tal forma, continuamos con los siguientes pasos en la descripción.

# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Crea una seccion (Con #) que haga referencia a la creacion de graficos del top 10 tanto de barrios como de compañias, esto es importante para la estructura del proyecto ya que asi queda clara cada seccion del mismo.</div>
# 
# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor #2</b> <a class="tocSkip"></a>
# 
# 
# Corregido.
# </div>

# ## Creación de gráficos
# 
# Aqui trataremos de verificar la relación entre el top 10 de las principales compañías y barrios de fin de recorrido que tienen mayor preferencia los usuarios. Es por ello que a continuación se muestran los siguientes análisis y gráficos correspondientes.

# In[10]:


# Esto es para verificar el top 10 de los principales barrios en terminos de finalizacion del recorrido 
top_10 = df_sql_4.sort_values(by='average_trips', ascending=False).head(10)
print("Los 10 principales barrios en términos de finalización del recorrido:")
print(top_10)


# ### Gráfico: Número de viajes por Empresa de taxis (general)

# In[11]:


# Grafico de relacion entre empresas de taxi y numero de viajes
# En este primer paso primero se observan todas la compañias y despues nos enfocamos en el top 10 
companies = df_sql_1['company_name']
trips_amount = df_sql_1['trips_amount']

# Paleta de colores única para las barras
num_companies = len(companies)
colors = plt.cm.tab10(np.linspace(0, 1, num_companies))

plt.figure(figsize=(18, 14))
for i in range(num_companies):
    plt.bar(i, trips_amount[i], color=colors[i], label=companies[i])

plt.title('Número de Viajes por Empresa de Taxis')
plt.xlabel('Empresa de Taxis')
plt.ylabel('Número de Viajes')
plt.xticks(range(num_companies), companies, rotation=45, ha='right')
plt.tight_layout()

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.show()


# ### Gráfico: Número de viajes por Empresa de taxis (Top 10)

# In[12]:


# Grafico que relaciona solo los top 10 de compañias con mas viajes
# Se obtienen las 10 compañías con mas viajes
top_companies = df_sql_1.sort_values(by='trips_amount', ascending=False).head(10)
companies = top_companies['company_name']
trips_amount = top_companies['trips_amount']

num_companies = len(companies)
colors = plt.cm.tab10(np.linspace(0, 1, num_companies))

plt.figure(figsize=(18, 14))
for i in range(num_companies):
    plt.bar(i, trips_amount.iloc[i], color=colors[i], label=companies.iloc[i])

plt.title('Número de Viajes por Empresa de Taxis (Top 10)')
plt.xlabel('Empresa de Taxis')
plt.ylabel('Número de Viajes')
plt.xticks(range(num_companies), companies, rotation=45, ha='right')
plt.tight_layout()

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.show()


# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# En este caso el grafico esta **excelente** y aporta buena informacion.
# 
# A nuestro cliente le interesa el top 10, por lo que podes crear otro grafico para visualizar ese top 10 o dejarlo y crear un nuevo para que se puede visualizar ese top 10, de esa manera se eliminaria el ruido a la hora del analisis.</div>
# 
# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor #2</b> <a class="tocSkip"></a>
# 
# 
# Corregido.
# </div>

# ### Gráfico: Top 10 de Barrios por Número de Finalizaciones de Viajes

# In[13]:


#Grafico del top 10 de barrios en donde termina recorrido 
plt.figure(figsize=(10, 6))
plt.bar(top_10['dropoff_location_name'], top_10['average_trips'], color='salmon')
plt.title('Top 10 Barrios por Número de Finalizaciones de Viajes')
plt.xlabel('Barrio')
plt.ylabel('Promedio de Viajes')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ### Conclusión
# Es evidente que algunas compañías de taxis tienen una presencia más destacada en ciertas regiones de Chicago en comparación con otras. Este fenómeno, muy probablemente, está relacionado con los destinos finales preferidos por los usuarios.
# 
# Una estrategia que podría beneficiar a aquellas compañías con menor cantidad de viajes realizados sería incrementar su presencia en las zonas donde existe una mayor demanda de servicios de transporte. Esta estrategia fomentaría una competencia saludable entre todas las compañías, evitando la consolidación de monopolios y promoviendo un entorno favorable para el mercado.
# 
# Es importante tener en cuenta otras variables además de la ubicación, como por ejemplo, las tarifas ofrecidas por cada compañía. Los precios competitivos también pueden jugar un papel significativo en la elección de los usuarios y, por ende, en el éxito comercial de cada compañía.

# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Excelentes conclusiones.</div>

# # Paso 5. Prueba de hipótesis (Python)
# 
# ### Descripción general
# 
# /datasets/project_sql_result_07.csv — el resultado de la última consulta. Contiene datos sobre viajes desde el Loop hasta el Aeropuerto Internacional O'Hare. Recuerda, estos son los valores de campo de la tabla:
# 
#     start_ts: fecha y hora de la recogida
#     weather_conditions: condiciones climáticas en el momento en el que comenzó el viaje
#     duration_seconds: duración del viaje en segundos
#     
# Prueba la hipótesis:
# 
# "La duración promedio de los viajes desde el Loop hasta el Aeropuerto Internacional O'Hare cambia los sábados lluviosos".
# 
# Decide por tu cuenta dónde establecer el nivel de significación (alfa).
# 
# Explica:
# 
#     cómo planteaste las hipótesis nula y alternativa
#     qué criterio usaste para probar las hipótesis y por qué

# In[14]:


# En la primera seccion del proyecto se cargo el dataset correpondiente a esta seccion del proyecto
# Aqui unicamente se vera mas informacion sobre este dataset df_sql_7
df_sql_7.info()
print()

print(df_sql_7.head())
print()

print(df_sql_7.describe())
print()


# In[15]:


# Igual que en la primera seccion verificaremos si hay valores duplicados en este dataframe
duplicates = df_sql_7.duplicated()
num_duplicates = duplicates.sum()

if num_duplicates == 0:
    print("No hay información duplicada en el DataFrame.")
else:
    print(f"Hay {num_duplicates} filas duplicadas en el DataFrame.")


# In[16]:


# Para tratar estos duplicados los filtraremos para poder hacer un buen analisis de la informacion
df_sql_7_sin_duplicados = df_sql_7.drop_duplicates()

duplicates = df_sql_7_sin_duplicados.duplicated()
num_duplicates = duplicates.sum()

if num_duplicates == 0:
    print("No hay información duplicada en el DataFrame después de eliminar duplicados.")
else:
    print(f"Después de eliminar duplicados, quedan {num_duplicates} filas duplicadas en el DataFrame.")


# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Carga este dataset en la celda correspondiente (La del inicio). Esto es considerado una  buena practica ya que siempre queremos tener al principio todo cargado por si hay que hacer una modificacion en el futuro.</div>
# 
# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor #2</b> <a class="tocSkip"></a>
# 
# 
# Corregido.
# </div>

# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Realiza una exploracion inicial de este dataset, el mismo debe contener los siguientes metodos:
# 
# - info()
# - head()
# - describe()
# - Verifica duplicados.
# - Describi el dataset
# 
# </div>
# 
# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor #2</b> <a class="tocSkip"></a>
# 
# 
# Corregido, myu bien..
# </div>

# Para poder probar la hipótesis anterior, se plantean las siguientes hipótesis nula y alternativa:
# 
# Hipótesis nula (H0): La duración promedio de los viajes desde el Loop hasta el Aeropuerto Internacional O'Hare es la misma en los sábados lluviosos que en los sábados no lluviosos.
# 
# Hipótesis alternativa (H1): La duración promedio de los viajes desde el Loop hasta el Aeropuerto Internacional O'Hare es diferente en los sábados lluviosos que en los sábados no lluviosos.
# 
# Para ello es necesario en el siguente código, dividir los datos donde las condiciones climáticas sean con lluvia y sin lluvia para poder comparar los tiempo y verficiar si hay una diferencia significante.

# In[17]:


# Se convierte la columna start_ts a tipo datetime utilizando .loc[]
df_sql_7_sin_duplicados['start_ts'] = pd.to_datetime(df_sql_7_sin_duplicados['start_ts'])

# Como nos interesan los sábados se filtra el dataframe con el número 5 que representa ese día de la semana
df_saturdays = df_sql_7_sin_duplicados[df_sql_7_sin_duplicados['start_ts'].dt.dayofweek == 5] 

# Se dividen los datos en dos grupos: sábados lluviosos y sábados no lluviosos
df_rainy = df_saturdays[df_saturdays['weather_conditions'] == 'Bad']
df_not_rainy = df_saturdays[df_saturdays['weather_conditions'] == 'Good']

# Se realiza la prueba de Levene para igualdad de varianzas
statistic, p_value_levene = stats.levene(df_rainy['duration_seconds'], df_not_rainy['duration_seconds'])

# Nivel de significancia (alfa)
alfa = 0.05

print("Valor p de la prueba de Levene para igualdad de varianzas:", p_value_levene)

# Se compara el valor p con el nivel de significancia
if p_value_levene < alfa:
    print("Hay suficiente evidencia para afirmar que hay una diferencia significativa en las varianzas de los dos grupos.")
else:
    print("No hay suficiente evidencia para afirmar que hay una diferencia significativa en las varianzas de los dos grupos.")

# Si las varianzas son iguales, realizamos la prueba t de Student con equal_var=True
if p_value_levene >= alfa:
    statistic_t, p_value_t = stats.ttest_ind(df_rainy['duration_seconds'], df_not_rainy['duration_seconds'], equal_var=True)
    
    print("\nValor p de la prueba t de Student con equal_var=True:", p_value_t)

    if p_value_t < alfa:
        print("Hay suficiente evidencia para rechazar la hipótesis nula, lo que indica que hay una diferencia significativa entre las medias de los dos grupos.")
    else:
        print("No hay suficiente evidencia para rechazar la hipótesis nula, lo que sugiere que no hay una diferencia significativa entre las medias de los dos grupos.")


# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# 
# Hola, muchas gracias por tus comentarios y tambien por la revisión. En esta parte tengo una duda porque no entiendo el mensaje de error que despliega el cuaderno. Si me puedes ayudar a aclararmelo te lo agradeceria muchisimo. Cambie la prueba a la que me recomendaste en el codigo.
# </div>

# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# En tu caso lo que realizas es una observacion de las dos medias, lo que no esta mal, pero la diferencia en las medias de dos conjuntos de datos no implica automáticamente que las varianzas sean diferentes.
# 
# Las pruebas estadísticas están diseñadas para tomar en cuenta el tamaño de la muestra y calcular si la diferencia observada en las varianzas es estadísticamente significativa o si podría deberse al azar.
# 
# Por lo tanto, aunque los valores de las medias pueden proporcionar una indicación inicial, es importante realizar una prueba estadística para obtener una conclusión más sólida sobre la igualdad de las varianzas.
# 
# Por esto es que se recomienda realizar una prueba, por ejemplo la de **Levene**.</div>
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor #2</b> <a class="tocSkip"></a>
# 
# 
# La prueba de Levene esta excelentemente realizada por lo que el parametro equal_var ira en true.
# 
# Lo que faltaria hacer ahora es la prueba estadistica en si con el t estudent. Que en este caso no se esta haciendo.
# 
# La prueba estaba perfecto y utilizamos levene para extraer el valor de ese parametro por lo que quedaran las dos pruebas.
# 
# El aviso indica que se está intentando modificar una vista de un DataFrame (una "slice") en lugar de una copia del DataFrame original. Para que no salgo podes utilizar el metodo loc, de igual manera no modificara el resultado. </div>
# 
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Simplemente a modo de consejo, cuando realizamos este tipo de prueba, los outliers pueden modificarla. Por lo que generalmente se recomienda tratarlos antes  de realizar la prueba.
# </div>

# ### Conclusión
# 
# Al realizar pruebas estadísticas, es esencial establecer un nivel de significancia (alfa), que representa el umbral de probabilidad que utilizamos para determinar si rechazamos o no la hipótesis nula. En este análisis, se utilizó un nivel de significancia comúnmente aceptado de alfa = 0.05, lo que implica que estamos dispuestos a aceptar un 5% de probabilidad de cometer un error tipo I, es decir, rechazar incorrectamente la hipótesis nula.
# 
# La hipótesis nula para nuestro análisis fue que la duración promedio de los viajes desde el Loop hasta el Aeropuerto Internacional O'Hare no cambia los sábados lluviosos. La hipótesis alternativa, por lo tanto, sería la respuesta al análisis. La presencia de lluvia puede influir en la duración de los viajes debido a diversos factores, como la disminución de la velocidad del tráfico debido a precauciones adicionales de los conductores y una mayor demanda de servicios de taxis. Estos efectos pueden conducir a diferencias significativas en la duración promedio de los viajes entre sábados lluviosos y no lluviosos.

# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Recorda realizar una conclusion general. Esta debe contener todo lo que se hizo en el proyecto de forma enumerada o items.
# 
# Desde la carga e importacion, pasando por los cambios realizado (Y el porque de esas decisiones). Agregando lo que se hizo en cada seccion a modo resumen y las conclusiones del  trabajo.
# 
# El mismo, sirve como resumen de lo realizado en cada proyecto.</div>
# 
# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor #2</b> <a class="tocSkip"></a>
# 
# 
# Corregido, excelente.
# </div>

# ## Conclusión del proyecto final del Sprint 7

# Después de explorar y trabajar con diferentes fuentes de información, he llegado a comprender mejor cómo realizar un análisis de datos efectivo. Desde la etapa inicial de recopilación de datos utilizando herramientas como JSON, requests y BeautifulSoup, hasta el almacenamiento de la información en bases de datos para facilitar su compartición y manipulación, cada paso en el proceso de análisis de datos es crucial.
# 
# Aprendí que estas herramientas son de gran utilidad para buscar información sobre cualquier tema que necesitemos investigar. La capacidad de extraer datos de páginas web, procesarlos y almacenarlos para su posterior análisis proporciona una base sólida para realizar investigaciones en profundidad.
# 
# Además, comprendí la importancia de filtrar y preprocesar los datos antes de analizarlos. Utilizando técnicas como la eliminación de datos duplicados y la aplicación de pruebas estadísticas para verificar la validez de los conjuntos de datos, podemos garantizar la fiabilidad de nuestros análisis y sacar conclusiones precisas.
# 
# En resumen, el conocimiento y la aplicación de herramientas como JSON, requests, BeautifulSoup y bases de datos, junto con técnicas de preprocesamiento y análisis de datos en Python, son fundamentales para realizar un análisis de datos efectivo y obtener conclusiones significativas.

# In[ ]:




