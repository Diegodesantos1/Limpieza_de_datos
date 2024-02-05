<h1 align="center">Limpieza y Análisis de Datos</h1>

En este [repositorio](https://github.com/Diegodesantos1/Limpieza_de_datos) queda resuelta la práctica de los análisis y limpieza de datos.

Tras la limpieza de datos he realizado el siguiente análisis:

En primer lugar un análisis de valores interesantes como la varianza, desviación típica etc:

```python
def estadisticos(self):
        columnas_numericas = ["puntaje"]

        for columna in columnas_numericas:
            estadisticos = {
                'media': self.df[columna].mean(),
                'mediana': self.df[columna].median(),
                'desviacion estandar': self.df[columna].std(),
                'varianza': self.df[columna].var(),
                'minimo': self.df[columna].min(),
                'maximo': self.df[columna].max()
            }
            print(f"Estadisticos de la columna {columna}: {estadisticos}")
```


El código es el siguiente:

```python
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from sklearn.linear_model import LinearRegression

conexion = sqlite3.connect('bookmaker.db')

consulta_sql_equipos = "SELECT * FROM equipos;"

df_equipos = pd.read_sql(consulta_sql_equipos, conexion)

conexion = sqlite3.connect('bookmaker.db')

consulta_sql = "SELECT * FROM partidos;"

df = pd.read_sql(consulta_sql, conexion)

conexion.close()

plt.rcParams['figure.figsize'] = (12, 8)


class Analisis_Partidos:
    def __init__(self, df):
        self.df = df

    def estadisticos(self):
        columnas_numericas = ["equipo_local", "equipo_visistante"]

        for columna in columnas_numericas:

            estadisticos = {
                'media': self.df[columna].mean(),
                'mediana': self.df[columna].median(),
                'desviacion estandar': self.df[columna].std(),
                'varianza': self.df[columna].var(),
                'minimo': self.df[columna].min(),
                'maximo': self.df[columna].max()
            }
            print(f"Estadisticos de la columna {columna}: {estadisticos}")

    def grafico_puntos(self):
        self.df['equipo_local'].plot()
        self.df['equipo_visistante'].plot()
        plt.axhline(self.df['equipo_local'].mean(), color='b', linestyle='--')
        plt.axhline(self.df['equipo_visistante'].mean(),
                    color='orange', linestyle='--')
        plt.legend()
        plt.title("Diferencia de puntos entre equipos")
        plt.show()
        return self.df


class AnalisisEquipos:
    def __init__(self, df):
        self.df = df

    def estadisticos(self):
        columnas_numericas = ["puntaje"]

        for columna in columnas_numericas:
            estadisticos = {
                'media': self.df[columna].mean(),
                'mediana': self.df[columna].median(),
                'desviacion estandar': self.df[columna].std(),
                'varianza': self.df[columna].var(),
                'minimo': self.df[columna].min(),
                'maximo': self.df[columna].max()
            }
            print(f"Estadisticos de la columna {columna}: {estadisticos}")

    def grafico_puntos(self):
        # Gráfico de barras para mostrar el puntaje de cada equipo
        fig, ax = plt.subplots()
        bars = ax.bar(self.df['nombre'], self.df['puntaje'], color='skyblue')

        # Añadir escudos a la barra correspondiente
        for i, bar in enumerate(bars):
            equipo_nombre = self.df.loc[i, 'nombre']
            escudo_url = self.df.loc[i, 'escudo']
            img = Image.open("Escudos/" + escudo_url)
            img = img.resize((50, 50))

            imagebox = OffsetImage(img, zoom=0.4)
            ab = AnnotationBbox(
                imagebox, (i, 0), frameon=False, boxcoords="data", pad=0.5)
            ax.add_artist(ab)

        plt.title("Puntaje de cada equipo con escudos")
        plt.xlabel("Equipos")
        plt.ylabel("Puntaje")
        plt.xticks(rotation=45, ha="right")
        plt.show()

        pais_counts = self.df['pais'].value_counts()
        pais_counts.plot.pie(autopct='%1.1f%%', startangle=90)
        plt.title("Distribución de equipos por país")
        plt.show()

        return self.df

    def regresion_lineal(self):
        # Extraer las variables independientes (X) y dependientes (y)
        # Usaremos el índice como variable independiente
        X = self.df.index.values.reshape(-1, 1)
        # Ordenar los valores de menor a mayor
        y = self.df['puntaje'].sort_values().values.reshape(-1, 1)

        # Crear un modelo de regresión lineal
        modelo = LinearRegression()

        # Ajustar el modelo a los datos
        modelo.fit(X, y)

        # Predecir los puntajes para los valores existentes y futuros
        predicciones = modelo.predict(X)

        # Graficar los datos y la línea de regresión
        plt.scatter(X, y, label='Datos reales')
        plt.plot(X, predicciones, color='red', label='Regresión lineal')
        plt.xlabel('Equipos')
        plt.ylabel('Puntaje')
        plt.title('Regresión Lineal de Puntajes de Equipos')
        plt.legend()
        plt.show()


analisis = Analisis_Partidos(df)
df = analisis.grafico_puntos()
analisis.estadisticos()

analisis_equipos = AnalisisEquipos(df_equipos)
df_equipos = analisis_equipos.grafico_puntos()
analisis_equipos.estadisticos()
analisis_equipos.regresion_lineal()
```

Después gráficas como la diferencia de puntos entre el equipo local tanto visitante:

![image](https://github.com/Diegodesantos1/Limpieza_de_datos/assets/91721855/68f5a32d-95e7-483f-9821-f9246e6cfb08)

También una gráfica con la distribución de puntos de cada equipo de la Champions con su respectivo escudo:

![image](https://github.com/Diegodesantos1/Limpieza_de_datos/assets/91721855/9891bade-72e1-4665-9e17-25ab2ae382c3)

Además una gráfica con la distribución por país de los diferentes equipos en la competición Europea:

![image](https://github.com/Diegodesantos1/Limpieza_de_datos/assets/91721855/4df58005-388d-4211-adab-50beb1efd320)

Y por último una regresión lineal de los datos:

![image](https://github.com/Diegodesantos1/Limpieza_de_datos/assets/91721855/e53b22b2-2349-45d1-96c0-d38e418b95b5)

