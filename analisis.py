import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

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


analisis = Analisis_Partidos(df)
df = analisis.grafico_puntos()
analisis.estadisticos()

analisis_equipos = AnalisisEquipos(df_equipos)
df_equipos = analisis_equipos.grafico_puntos()
analisis_equipos.estadisticos()
