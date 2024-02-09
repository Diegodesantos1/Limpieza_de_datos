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
