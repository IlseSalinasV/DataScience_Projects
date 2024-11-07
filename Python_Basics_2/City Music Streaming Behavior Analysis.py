#!/usr/bin/env python
# coding: utf-8

# # Hola Ilse! <a class="tocSkip"></a>
# 
# Mi nombre es Oscar Flores y tengo el gusto de revisar tu proyecto. Si tienes algún comentario que quieras agregar en tus respuestas te puedes referir a mi como Oscar, no hay problema que me trates de tú.
# 
# Si veo un error en la primera revisión solamente lo señalaré y dejaré que tú encuentres de qué se trata y cómo arreglarlo. Debo prepararte para que te desempeñes como especialista en Data, en un trabajo real, el responsable a cargo tuyo hará lo mismo. Si aún tienes dificultades para resolver esta tarea, te daré indicaciones más precisas en una siguiente iteración.
# 
# Te dejaré mis comentarios más abajo - **por favor, no los muevas, modifiques o borres**
# 
# Comenzaré mis comentarios con un resumen de los puntos que están bien, aquellos que debes corregir y aquellos que puedes mejorar. Luego deberás revisar todo el notebook para leer mis comentarios, los cuales estarán en rectángulos de color verde, amarillo o rojo como siguen:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
#     
# Muy bien! Toda la respuesta fue lograda satisfactoriamente.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
# 
# Existen detalles a mejorar. Existen recomendaciones.
# </div>
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
# 
# Se necesitan correcciones en el bloque. El trabajo no puede ser aceptado con comentarios en rojo sin solucionar.
# </div>
# 
# Cualquier comentario que quieras agregar entre iteraciones de revisión lo puedes hacer de la siguiente manera:
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta estudiante.</b> <a class="tocSkip"></a>
# </div>
# 

# ## Resumen de la revisión 1 <a class="tocSkip"></a>

# <div class="alert alert-block alert-danger">
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
# 
# Buen trabajo! Tu notebook está casi terminado. Tan solo falta que corrijas algunos puntos importantes, como ordenar alfabéticamente los géneros antes y después de la corrección de nombre de género y resetear el índice al remover filas. Realiza las correcciones señaladas en rojo y revisaré tu notebook en la siguiente iteración.
#     
# Saludos!    
# </div>

# ## Resumen de la revisión 2 <a class="tocSkip"></a>

# <div class="alert alert-block alert-success">
# <b>Comentario de Reviewer v2</b> <a class="tocSkip"></a>
# 
# Bien hecho! Has completado todos los puntos necesarios del notebook, tu proyecto está aprobado!
#     
# Saludos!    
# </div>

# ----

# # Déjame escuchar la música

# # Contenido <a id='back'></a>
# 
# * [Introducción](#intro)
# * [Etapa 1. Descripción de los datos](#data_review)
#     * [Conclusiones](#data_review_conclusions)
# * [Etapa 2. Preprocesamiento de datos](#data_preprocessing)
#     * [2.1 Estilo del encabezado](#header_style)
#     * [2.2 Valores ausentes](#missing_values)
#     * [2.3 Duplicados](#duplicates)
#     * [2.4 Conclusiones](#data_preprocessing_conclusions)
# * [Etapa 3. Prueba de hipótesis](#hypothesis)
#     * [3.1 Hipótesis 1: actividad de los usuarios y las usuarias en las dos ciudades](#activity)
# * [Conclusiones](#end)

# ## Introducción <a id='intro'></a>
# Como analista de datos, tu trabajo consiste en analizar datos para extraer información valiosa y tomar decisiones basadas en ellos. Esto implica diferentes etapas, como la descripción general de los datos, el preprocesamiento y la prueba de hipótesis.
# 
# Siempre que investigamos, necesitamos formular hipótesis que después podamos probar. A veces aceptamos estas hipótesis; otras veces, las rechazamos. Para tomar las decisiones correctas, una empresa debe ser capaz de entender si está haciendo las suposiciones correctas.
# 
# En este proyecto, compararás las preferencias musicales de las ciudades de Springfield y Shelbyville. Estudiarás datos reales de transmisión de música online para probar la hipótesis a continuación y comparar el comportamiento de los usuarios y las usuarias de estas dos ciudades.
# 
# ### Objetivo:
# Prueba la hipótesis:
# 1. La actividad de los usuarios y las usuarias difiere según el día de la semana y dependiendo de la ciudad.
# 
# 
# ### Etapas
# Los datos del comportamiento del usuario se almacenan en el archivo `/datasets/music_project_en.csv`. No hay ninguna información sobre la calidad de los datos, así que necesitarás examinarlos antes de probar la hipótesis.
# 
# Primero, evaluarás la calidad de los datos y verás si los problemas son significativos. Entonces, durante el preprocesamiento de datos, tomarás en cuenta los problemas más críticos.
# 
# Tu proyecto consistirá en tres etapas:
#  1. Descripción de los datos.
#  2. Preprocesamiento de datos.
#  3. Prueba de hipótesis.
# 
# 
# 
# 
# 
# 
# 

# [Volver a Contenidos](#back)

# ## Etapa 1. Descripción de los datos <a id='data_review'></a>
# 
# Abre los datos y examínalos.

# Necesitarás `pandas`, así que impórtalo.

# In[1]:


# Importar pandas
import pandas as pd


# Lee el archivo `music_project_en.csv` de la carpeta `/datasets/` y guárdalo en la variable `df`:

# In[2]:


# Leer el archivo y almacenarlo en df
df = pd.read_csv('/datasets/music_project_en.csv')


# Muestra las 10 primeras filas de la tabla:

# In[3]:


# Obtener las 10 primeras filas de la tabla df
df.head(10)


# <div class="alert alert-block alert-warning">
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
# 
# Nota que no es necesario usar print() para mostrar las primeras 10 filas de la tabla. Prueba usando solamente df.head(10) y compara la diferencia, verás que es visualmente mejor.
# </div>

# Obtén la información general sobre la tabla con un comando. Conoces el método que muestra la información general que necesitamos.

# In[4]:


# Obtener la información general sobre nuestros datos
print(df.info())


# Estas son nuestras observaciones sobre la tabla. Contiene siete columnas. Almacenan los mismos tipos de datos: `object`.
# 
# Según la documentación:
# - `' userID'`: identificador del usuario o la usuaria;
# - `'Track'`: título de la canción;
# - `'artist'`: nombre del artista;
# - `'genre'`: género de la pista;
# - `'City'`: ciudad del usuario o la usuaria;
# - `'time'`: la hora exacta en la que se reprodujo la canción;
# - `'Day'`: día de la semana.
# 
# Podemos ver tres problemas con el estilo en los encabezados de la tabla:
# 1. Algunos encabezados están en mayúsculas, otros en minúsculas.
# 2. Hay espacios en algunos encabezados.
# 3. Algunos encabezados deberían de ser más descriptivos tales como time and day que pueden llamarse time_listened y day_listened. De esta forma se puede entender que se refieren éstos dos al tiempo y día en el que se reprodució la canción.

# <div class="alert alert-block alert-success">
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
# 
# Correcto. Esta primera parte es la exploración básica de la data y ayuda a tener una idea de su composición.
# </div>

# ### Escribe observaciones de tu parte. Estas son algunas de las preguntas que pueden ser útiles: <a id='data_review_conclusions'></a>
# 
# 1.   La información que se nos propociona en las columnas y filas la podemos entender fácilmente a partir de la impresión del código print(df.info()) que nos muestra información de los índidces, el nombre de las columnas, la cantidad de valores no nulos, los tipos de datos, el tipo de datos que entran en el dataset y el uso de memoria. 
# 
# 2.   De momento, parece que tenemos la información suficiente para poder responder a la hipótesis, no obstante hubiera sido también útil saber el número de veces que fueron reproducidas las canciones en ambas ciudades con el fin de tener un mejor monitoreo de la actividad de los días. 
# 
# 3.   Al imprimir solo las 10 filas del DataFrame, se puede observar en la fila 9 un dato ausente NaN en la columna de artista. Esto puede generar errores ya que puede darse el caso en el que esta misma canción sea repetida más adelante. Se recomienda que para identificar el total de valores ausentes se escriba este fragemento de código val_aus = df.isna().sum().

# [Volver a Contenidos](#back)

# ## Etapa 2. Preprocesamiento de datos <a id='data_preprocessing'></a>
# 
# El objetivo aquí es preparar los datos para que sean analizados.
# El primer paso es resolver cualquier problema con los encabezados. Luego podemos avanzar a los valores ausentes y duplicados. Empecemos.
# 
# Corrige el formato en los encabezados de la tabla.
# 

# ### Estilo del encabezado <a id='header_style'></a>
# Muestra los encabezados de la tabla (los nombres de las columnas):

# In[5]:


# Muestra los nombres de las columnas
print(df.columns)


# Cambia los encabezados de la tabla de acuerdo con las reglas del buen estilo:
# * Todos los caracteres deben ser minúsculas.
# * Elimina los espacios.
# * Si el nombre tiene varias palabras, utiliza snake_case.

# Anteriormente, aprendiste acerca de la forma automática de cambiar el nombre de las columnas. Vamos a aplicarla ahora. Utiliza el bucle for para iterar sobre los nombres de las columnas y poner todos los caracteres en minúsculas. Cuando hayas terminado, vuelve a mostrar los encabezados de la tabla:

# In[6]:


# Bucle en los encabezados poniendo todo en minúsculas
new_columns = []
for i in df.columns:
    i_low = i.lower()
    new_columns.append(i_low)
    
df.columns = new_columns    
print(df.columns)


# Ahora, utilizando el mismo método, elimina los espacios al principio y al final de los nombres de las columnas e imprime los nombres de las columnas nuevamente:

# In[7]:


# Bucle en los encabezados eliminando los espacios
new_columns = []
for i in df.columns:
    i_stripped = i.strip()
    new_columns.append(i_stripped)
    
df.columns = new_columns    
print(df.columns)


# Necesitamos aplicar la regla de snake_case a la columna `userid`. Debe ser `user_id`. Cambia el nombre de esta columna y muestra los nombres de todas las columnas cuando hayas terminado.

# In[8]:


# Cambiar el nombre de la columna "userid"
columns_new ={
    'userid':'user_id',
    }

df.rename(columns = columns_new, inplace = True)

print(df.columns)


# Comprueba el resultado. Muestra los encabezados una vez más:

# In[9]:


# Comprobar el resultado: la lista de encabezados
print(df.columns)


# [Volver a Contenidos](#back)

# <div class="alert alert-block alert-success">
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
# 
# Muy bien! Buen uso del método rename de los DataFrame de pandas.
# </div>

# ### Valores ausentes <a id='missing_values'></a>
#  Primero, encuentra el número de valores ausentes en la tabla. Debes utilizar dos métodos en una secuencia para obtener el número de valores ausentes.

# In[10]:


# Calcular el número de valores ausentes
print(df.isna().sum())


# No todos los valores ausentes afectan a la investigación. Por ejemplo, los valores ausentes en `track` y `artist` no son cruciales. Simplemente puedes reemplazarlos con valores predeterminados como el string `'unknown'` (desconocido).
# 
# Pero los valores ausentes en `'genre'` pueden afectar la comparación entre las preferencias musicales de Springfield y Shelbyville. En la vida real, sería útil saber las razones por las cuales hay datos ausentes e intentar recuperarlos. Pero no tenemos esa oportunidad en este proyecto. Así que tendrás que:
# * rellenar estos valores ausentes con un valor predeterminado;
# * evaluar cuánto podrían afectar los valores ausentes a tus cómputos;

# Reemplazar los valores ausentes en las columnas `'track'`, `'artist'` y `'genre'` con el string `'unknown'`. Como mostramos anteriormente en las lecciones, la mejor forma de hacerlo es crear una lista que almacene los nombres de las columnas donde se necesita el reemplazo. Luego, utiliza esta lista e itera sobre las columnas donde se necesita el reemplazo haciendo el propio reemplazo.

# In[11]:


# Bucle en los encabezados reemplazando los valores ausentes con 'unknown'
columns_to_replace = ['track','artist','genre']

for i in columns_to_replace:
    df[i].fillna('unknown', inplace=True) 
    
print(df.isna().sum())


# Ahora comprueba el resultado para asegurarte de que después del reemplazo no haya valores ausentes en el conjunto de datos. Para hacer esto, cuenta los valores ausentes nuevamente.

# In[12]:


# Contar valores ausentes
print(df.isna().sum())


# [Volver a Contenidos](#back)

# <div class="alert alert-block alert-success">
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
# 
# Correcto!
# </div>

# ### Duplicados <a id='duplicates'></a>
# Encuentra el número de duplicados explícitos en la tabla. Una vez más, debes aplicar dos métodos en una secuencia para obtener la cantidad de duplicados explícitos.

# In[13]:


# Contar duplicados explícitos
print(df.duplicated().sum())


# Ahora, elimina todos los duplicados. Para ello, llama al método que hace exactamente esto.

# In[14]:


# Eliminar duplicados explícitos
df = df.drop_duplicates().reset_index(drop=True) 

print(df.duplicated().sum()) 


# <div class="alert alert-block alert-danger">
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
# 
# Después de borrar filas te recomiendo resetear el índice. 
# </div>

# <div class="alert alert-block alert-success">
# <b>Comentario de Reviewer v2</b> <a class="tocSkip"></a>
# 
# Muy bien! Hacer esto evita posibles errores al trabajar con la data.
# </div>

# Comprobemos ahora si eliminamos con éxito todos los duplicados. Cuenta los duplicados explícitos una vez más para asegurarte de haberlos eliminado todos:

# In[15]:


# Comprobar de nuevo si hay duplicados

print(df.duplicated().sum()) 


# Ahora queremos deshacernos de los duplicados implícitos en la columna `genre`. Por ejemplo, el nombre de un género se puede escribir de varias formas. Dichos errores también pueden afectar al resultado.

# Para hacerlo, primero mostremos una lista de nombres de género únicos, ordenados en orden alfabético. Para ello:
# * Extrae la columna `genre` del DataFrame.
# * Llama al método que devolverá todos los valores únicos en la columna extraída.
# 

# In[16]:


#Ordenar alfabeticamente generos
df_ordenados = df.sort_values(by='genre')

# Obtener valores únicos del df ordenado alfabeticamente
df_unicos_ordenados = df_ordenados['genre'].unique()

# Imprimir los valores únicos ordenados
print(df_unicos_ordenados)


# <div class="alert alert-block alert-danger">
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
# 
# No olvides de mostrar los géneros en orden alfabético. Para ello puedes usar sort_values()
# </div>

# <div class="alert alert-block alert-success">
# <b>Comentario de Reviewer v2</b> <a class="tocSkip"></a>
# 
# Muy bien! Correcto!
# </div>

# Busca en la lista para encontrar duplicados implícitos del género `hiphop`. Estos pueden ser nombres escritos incorrectamente o nombres alternativos para el mismo género.
# 
# Verás los siguientes duplicados implícitos:
# * `hip`
# * `hop`
# * `hip-hop`
# 
# Para deshacerte de ellos, crea una función llamada `replace_wrong_genres()` con dos parámetros:
# * `wrong_genres=`: esta es una lista que contiene todos los valores que necesitas reemplazar.
# * `correct_genre=`: este es un string que vas a utilizar como reemplazo.
# 
# Como resultado, la función debería corregir los nombres en la columna `'genre'` de la tabla `df`, es decir, remplazar cada valor de la lista `wrong_genres` por el valor en `correct_genre`.
# 
# Dentro del cuerpo de la función, utiliza un bucle `'for'` para iterar sobre la lista de géneros incorrectos, extrae la columna `'genre'` y aplica el método `replace` para hacer correcciones.

# In[17]:


# Función para reemplazar duplicados implícitos
def replace_wrong_genres(df, wrong_genres, correct_genre): #la instruccion dice dos parametros pero tambien es necesario enviar df
    for i in wrong_genres:
        df['genre'] = df['genre'].replace(i, correct_genre) #recorre lista de hip, hop y hip-hop para cambiarlo por hiphop
    return df


# Ahora, llama a `replace_wrong_genres()` y pásale tales argumentos para que retire los duplicados implícitos (`hip`, `hop` y `hip-hop`) y los reemplace por `hiphop`:

# In[18]:


# Eliminar duplicados implícitos
duplicates = ['hip', 'hop','hip-hop']
correct_name = 'hiphop' # el nombre correcto
df = replace_wrong_genres(df, duplicates, correct_name)  


# Asegúrate de que los nombres duplicados han sido eliminados. Muestra la lista de valores únicos de la columna `'genre'` una vez más:

# In[19]:


# Comprobación de duplicados implícitos
#Ordenar alfabeticamente generos
df_ordenados = df.sort_values(by='genre')

# Obtener valores únicos del df ordenado alfabeticamente
df_unicos_ordenados = df_ordenados['genre'].unique()

# Imprimir los valores únicos ordenados
print(df_unicos_ordenados)

#Contar cuantos son los generos unicos
print(df['genre'].nunique())


# <div class="alert alert-block alert-danger">
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
# 
# No olvides de mostrar los géneros en orden alfabético. Para ello puedes usar sort_values()
# </div>

# <div class="alert alert-block alert-success">
# <b>Comentario de Reviewer v2</b> <a class="tocSkip"></a>
# 
# Muy bien! Ahora se aprecia la corrección realizada
# </div>

# [Volver a Contenidos](#back)

# ### Tus observaciones <a id='data_preprocessing_conclusions'></a>
# 
# Después de haber analizado los datos sobre los géneros de música existentes en el DataFrame, observé que es muy fácil que un género pueda ser almacenado por los usuarios de diferentes maneras y ahí radica la importancia de identificar que es el mismo, que esta duplicado y debe ser resumido en uno mismo para evitar que la información adquirida interrumpa el desarrollo de la hipótesis inicial. 

# [Volver a Contenidos](#back)

# ## Etapa 3. Prueba de hipótesis <a id='hypothesis'></a>

# ### Hipótesis: comparar el comportamiento del usuario o la usuaria en las dos ciudades <a id='activity'></a>

# La hipótesis afirma que existen diferencias en la forma en que los usuarios y las usuarias de Springfield y Shelbyville consumen música. Para comprobar esto, usa los datos de tres días de la semana: lunes, miércoles y viernes.
# 
# * Agrupa a los usuarios y las usuarias por ciudad.
# * Compara el número de canciones que cada grupo reprodujo el lunes, el miércoles y el viernes.
# 

# Realiza cada cálculo por separado.
# 
# El primer paso es evaluar la actividad del usuario en cada ciudad. Recuerda las etapas dividir-aplicar-combinar de las que hablamos anteriormente en la lección. Tu objetivo ahora es agrupar los datos por ciudad, aplicar el método apropiado para contar durante la etapa de aplicación y luego encontrar la cantidad de canciones reproducidas en cada grupo especificando la columna para obtener el recuento.
# 
# A continuación se muestra un ejemplo de cómo debería verse el resultado final:
# `df.groupby(by='....')['column'].method()`Realiza cada cálculo por separado.
# 
# Para evaluar la actividad de los usuarios y las usuarias en cada ciudad, agrupa los datos por ciudad y encuentra la cantidad de canciones reproducidas en cada grupo.
# 
# 

# In[20]:


# Contar las canciones reproducidas en cada ciudad
print(df.groupby('city')['track'].count())


# <div class="alert alert-block alert-success">
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
# 
# Correcto
# </div>

# De acuerdo con la información presentada, se observa que en la ciudad de Springfield el número de canciones reproducidas es mayor por lo que se puede deducir que hay mayor cantidad de actividad aquí. Sin embargo, hay que analizar que día de la semana es en la que más hay actividad en cada ciudad ya que puede darse el caso de que en un día específico haya más actividad en un lugar que en otro. 

# Ahora agrupemos los datos por día de la semana y encontremos el número de canciones reproducidas el lunes, miércoles y viernes. Utiliza el mismo método que antes, pero ahora necesitamos una agrupación diferente.
# 

# In[21]:


# Calcular las canciones reproducidas en cada uno de los tres días
print(df.groupby('day')['track'].count())


# <div class="alert alert-block alert-success">
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
# 
# Correcto
# </div>

# La información muestra que los viernes son los días en los que se reproducen más canciones. Ahora es necesario hacer una intersección entre este resultado y el anterior, para saber cual de las dos ciudades es la que escucha más canciones los viernes.

# Ya sabes cómo contar entradas agrupándolas por ciudad o día. Ahora necesitas escribir una función que pueda contar entradas según ambos criterios simultáneamente.
# 
# Crea la función `number_tracks()` para calcular el número de canciones reproducidas en un determinado día **y** ciudad. La función debe aceptar dos parámetros:
# 
# - `day`: un día de la semana para filtrar. Por ejemplo, `'Monday'` (lunes).
# - `city`: una ciudad para filtrar. Por ejemplo, `'Springfield'`.
# 
# Dentro de la función, aplicarás un filtrado consecutivo con indexación lógica.
# 
# Primero filtra los datos por día y luego filtra la tabla resultante por ciudad.
# 
# Después de filtrar los datos por dos criterios, cuenta el número de valores de la columna 'user_id' en la tabla resultante. Este recuento representa el número de entradas que estás buscando. Guarda el resultado en una nueva variable y devuélvelo desde la función.

# In[22]:


# Declara la función number_tracks() con dos parámetros: day= y city=.
def number_tracks(df, day, city):
    
    df_filtrado = df[(df['day'] == day) & (df['city'] == city)]
    users= df_filtrado['user_id'].count()
    return users


# Llama a `number_tracks()` seis veces, cambiando los valores de los parámetros para que recuperes los datos de ambas ciudades para cada uno de los tres días.

# In[23]:


# El número de canciones reproducidas en Springfield el lunes
result = number_tracks(df,'Monday','Springfield')
print(result)


# In[24]:


# El número de canciones reproducidas en Shelbyville el lunes
result = number_tracks(df,'Monday','Shelbyville')
print(result)


# In[25]:


# El número de canciones reproducidas en Springfield el miércoles
result = number_tracks(df,'Wednesday','Springfield')
print(result)


# In[26]:


# El número de canciones reproducidas en Shelbyville el miércoles
result = number_tracks(df,'Wednesday','Shelbyville')
print(result)


# In[27]:


# El número de canciones reproducidas en Springfield el viernes
result = number_tracks(df,'Friday','Springfield')
print(result)


# In[28]:


# El número de canciones reproducidas en Shelbyville el viernes
result = number_tracks(df,'Friday','Shelbyville')
print(result)


# **Conclusiones**
# 
# La hipótesis planteada es parcialmente correcta ya que la actividad de los usuarios en efecto depende de la ciudad pero varía en cuestión del día de la semana. Anteriormente se demostró que los viernes aparentemente son los días donde hay mayor cantidad de actividad registrada, no obstante después de comparar el resultado de ambas ciudades, el resultado solo dedpende de la ciudad de Springfield la cual tuvo mayor actividad mientras que Shelbyville registró mayor actividad los miércoles. Además se puede observar que si se hace una comparación entre ambas ciudades, Springfield cuenta con la mayor cantidad de usuarios que Springfield por lo que hacer un análisis de ambas ciudades no sería completamente equitativo. 

# [Volver a Contenidos](#back)

# <div class="alert alert-block alert-success">
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
# 
# Muy bien! Aunque hubiese sido mejor mostrar esto en una tabla
# </div>

# # Conclusiones <a id='end'></a>

# Considero que con el fin de obtener resultados más verídicos se puede considerar la cantidad de población en ambas ciudades para saber la razón por la que en Shelbyville la actividad es menor que en Springfield y mejorar la publicidad para que existan más usuarios interesados en la plataforma musical y aumentar de esta manera el número ded clientes. 

# ### Nota
# En proyectos de investigación reales, la prueba de hipótesis estadística es más precisa y cuantitativa. También ten en cuenta que no siempre se pueden sacar conclusiones sobre una ciudad entera a partir de datos de una sola fuente.
# 
# Aprenderás más sobre la prueba de hipótesis en el sprint de análisis estadístico de datos.

# [Volver a Contenidos](#back)
