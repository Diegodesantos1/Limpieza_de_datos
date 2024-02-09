import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://kassiesa.net/uefa/data/method5/trank10-2024.html'

page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# Buscar la tabla con los datos
table = soup.find_all('table')[0]

# Crear un DataFrame con los datos directamente del contenido HTML
df = pd.read_html(str(table), header=[0, 1])[0]

# Asignar nombres a las columnas
columnas = ["Position", "-", "Club", "Country", "14/15", "15/16", "16/17", "17/18", "18/19",
            "19/20", "20/21", "21/22", "22/23", "23/24", "Title Points", "Total Points", "Country Part"]
df.columns = columnas

# Eliminar las columnas que no interesan
df = df.drop(df.columns[[1, 3, 16]], axis=1)

# Guardar la primera fila
primera_fila = df.iloc[0]

# Eliminar las filas que contienen los países especificados
paises = ['Spain', 'England', 'Germany', 'Italy', 'France', 'Portugal', 'Netherlands', 'Russia', 'Belgium', 'Ukraine', 'Turkey', 'Austria', 'Czech Republic', 'Switzerland', 'Scotland', 'Denmark', 'Scotland', 'Greece', 'Croatia', 'Norway', 'Serbia', 'Israel', 'Cyprus', 'Poland', 'Sweden', 'Azerbaijan', 'Bulgaria', 'Romania', 'Slovakia', 'Hungary',
          'Kazakhstan', 'Belarus', 'Slovenia', 'Liechtenstein', 'Moldova', 'Finland', 'Ireland', 'Bosnia-Herzegovina', 'Iceland', 'Latvia', 'Armenia', 'Lithuania', 'Albania', 'Faroe Islands', 'Luxembourg', 'Kosovo', 'North Macedonia', 'Malta', 'Northern Ireland', 'Georgia', 'Estonia', 'Wales', 'Montenegro', 'Gibraltar', 'Andorra', 'San Marino']
df = df[~df['Club'].isin(paises)]

# Convertir los valores en la columna 'Position' a numéricos
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')

# Rellenar los valores NaN con el método ffill
df['Position'] = df['Position'].fillna(method='ffill')

# Convertir los valores a enteros
df['Position'] = df['Position'].astype('Int64')

# Añadir la fila específica al DataFrame
nueva_fila = pd.DataFrame([[1, '-', 'Real Madrid', '-', 29.0, 33.0, 33.0, 32.0,
                          19.0, 17.0, 26.0, 30.0, 29.0, 21.0, 94.0, 363.0, '-']], columns=columnas)
df = pd.concat([nueva_fila, df], ignore_index=True)

# Eliminar las columnas que no interesan
df = df.drop(df.columns[[1, 3, 16]], axis=1)

# Guardar el DataFrame en un archivo CSV
df.to_csv('Datos_Limpios/UEFA_Ranking.csv', index=False)