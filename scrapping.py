import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://kassiesa.net/uefa/data/method5/trank10-2024.html'

page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# Busco la tabla con los datos
table = soup.find_all('table')[0]

# Creo un dataframe con los datos directamente del contenido HTML
df = pd.read_html(str(table), header=[0, 1])[0]

# Guardo el dataframe en un archivo csv
df.to_csv('Datos/UEFA_Ranking.csv', index=False)

# Ahora nombro las columnas
columnas = ["Position","-","Club","Country","14/15","15/16","16/17","17/18","18/19","19/20","20/21","21/22","22/23","23/24","Title Points","Total Points","Country Part"]

df.columns = columnas

# Borro las columnas que no me interesan

df = df.drop(df.columns[[1, 3, 16]], axis=1)

# Ahora elimino las filas que contienen los siguientes paises

paises = ['Spain', 'England', 'Germany', 'Italy', 'France', 'Portugal', 'Netherlands', 'Russia', 'Belgium', 'Ukraine', 'Turkey', 'Austria', 'Czech Republic', 'Switzerland', 'Scotland', 'Denmark', 'Scotland', 'Greece', 'Croatia', 'Norway', 'Serbia', 'Israel', 'Cyprus', 'Poland','Sweden','Azerbaijan','Bulgaria', 'Romania','Slovakia','Hungary', 'Kazakhstan', 'Belarus','Slovenia','Liechtenstein','Moldova','Finland','Ireland','Bosnia-Herzegovina','Iceland','Latvia','Armenia','Lithuania','Albania','Faroe Islands','Luxembourg','Kosovo','North Macedonia','Malta','Northern Ireland','Georgia','Estonia','Wales','Montenegro','Gibraltar','Andorra','San Marino']

df = df[~df['Club'].isin(paises)]


# Ahora relleno los valores que faltan en la columna "Position"

df['Position'] = df['Position'].fillna(method='ffill')

df['Position'] = df['Position'].astype(int)

# Guardo el dataframe en un archivo csv
df.to_csv('Datos/UEFA_Ranking2.csv', index=False)