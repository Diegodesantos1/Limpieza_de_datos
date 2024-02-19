import os
from scrapping import scrapeo
from limpieza import limpieza

os.system("pip install -r requirements.txt")
scrapeo()
limpieza()


