import numpy as np
import pandas as pd


datos = ["Datos/champions-league-2017-UTC.csv", "Datos/champions-league-2018-UTC.csv", "Datos/champions-league-2019-UTC.csv",
         "Datos/champions-league-2020-UTC.csv", "Datos/champions-league-2021-UTC.csv", "Datos/champions-league-2022-UTC.csv"]

nombres = ["2017-2018", "2018-2019", "2019-2020",
           "2020-2021", "2021-2022", "2022-2023"]


for i in range(len(datos)):
    # Leer el archivo CSV
    df = pd.read_csv(datos[i])

    # Crear una copia del dataframe
    df_trabajar = df.copy()

    # Eliminar las columnas Match Number, Date, Location, Group
    df_trabajar = df_trabajar.drop(
        ['Match Number', 'Date', 'Location', 'Group'], axis=1)

    # Guardar el dataframe en un archivo CSV
    df_trabajar.to_csv("Datos_Limpios/UCL" +
                       nombres[i] + ".csv", index=False)


datos_limpios = ["Datos_Limpios/UCL2014-2015.csv", "Datos_Limpios/UCL2015-2016.csv", "Datos_Limpios/UCL2016-2017.csv", "Datos_Limpios/UCL2017-2018.csv", "Datos_Limpios/UCL2018-2019.csv", "Datos_Limpios/UCL2019-2020.csv",
                 "Datos_Limpios/UCL2020-2021.csv", "Datos_Limpios/UCL2021-2022.csv", "Datos_Limpios/UCL2022-2023.csv"]

nombres_goles = ["2014-2015", "2015-2016", "2016-2017", "2017-2018",
                 "2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023"]


# Printeo todos los nombres de los equipos desde 2014-2015 hasta 2022-2023

equipos = set()
for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    equipos = equipos.union(set(df['Home Team']))
    equipos = equipos.union(set(df['Away Team']))

equipos = sorted(equipos)
print("\nEquipos\n")

# Primero cambio todos Atletico Madrid', 'Atlético', 'Atlético Madrid', 'Atlético de Madrid' por 'Atletico de Madrid'

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Atletico Madrid', 'Atlético', 'Atlético Madrid', 'Atlético de Madrid'], 'Atletico de Madrid')
    df['Away Team'] = df['Away Team'].replace(
        ['Atletico Madrid', 'Atlético', 'Atlético Madrid', 'Atlético de Madrid'], 'Atletico de Madrid')
    df.to_csv(datos_limpios[i], index=False)

# Segundo cambio todos los 'FC Bayern Munich', 'Bayern Munich', 'Bayern' por 'Bayern Munich'

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['FC Bayern Munich', 'Bayern Munich', 'Bayern'], 'Bayern Munich')
    df['Away Team'] = df['Away Team'].replace(
        ['FC Bayern Munich', 'Bayern Munich', 'Bayern'], 'Bayern Munich')
    df.to_csv(datos_limpios[i], index=False)

# Tercero cambio todos los 'Borussia Dortmund', 'Dortmund' por 'Borussia Dortmund'

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Borussia Dortmund', 'Dortmund'], 'Borussia Dortmund')
    df['Away Team'] = df['Away Team'].replace(
        ['Borussia Dortmund', 'Dortmund'], 'Borussia Dortmund')
    df.to_csv(datos_limpios[i], index=False)

# Cuarto cambio todos los 'CSKA Moscow', 'CSKA Moskva' por 'CSKA Moscow'

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['CSKA Moscow', 'CSKA Moskva'], 'CSKA Moscow')
    df['Away Team'] = df['Away Team'].replace(
        ['CSKA Moscow', 'CSKA Moskva'], 'CSKA Moscow')
    df.to_csv(datos_limpios[i], index=False)

# Quinto cambio todos los 'Galatasaray', 'Galatasaray Istanbul' por 'Galatasaray'

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Galatasaray', 'Galatasaray Istanbul'], 'Galatasaray')
    df['Away Team'] = df['Away Team'].replace(
        ['Galatasaray', 'Galatasaray Istanbul'], 'Galatasaray')
    df.to_csv(datos_limpios[i], index=False)

# Sexto cambio todos los 'Leverkusen', 'Bayer Leverkusen' por 'Bayer Leverkusen'

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Leverkusen', 'Bayer Leverkusen'], 'Bayer Leverkusen')
    df['Away Team'] = df['Away Team'].replace(
        ['Leverkusen', 'Bayer Leverkusen'], 'Bayer Leverkusen')
    df.to_csv(datos_limpios[i], index=False)

# Septimo cambio todos los FC Porto', 'Porto' por 'FC Porto'

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['FC Porto', 'Porto'], 'FC Porto')
    df['Away Team'] = df['Away Team'].replace(
        ['FC Porto', 'Porto'], 'FC Porto')
    df.to_csv(datos_limpios[i], index=False)

# Octavo cambio todos los 'FC Barcelona', 'Barcelona' por 'FC Barcelona'

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['FC Barcelona', 'Barcelona'], 'FC Barcelona')
    df['Away Team'] = df['Away Team'].replace(
        ['FC Barcelona', 'Barcelona'], 'FC Barcelona')
    df.to_csv(datos_limpios[i], index=False)

# Noveno cambio todos los 'FC Basel', 'Basel' por 'FC Basel'

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['FC Basel', 'Basel'], 'FC Basel')
    df['Away Team'] = df['Away Team'].replace(
        ['FC Basel', 'Basel'], 'FC Basel')
    df.to_csv(datos_limpios[i], index=False)

# Decimo cambio todos los 'FC Schalke 04', 'Schalke 04', 'Schalke' por 'FC Schalke 04'

for i in range(len(datos_limpios)):

    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['FC Schalke 04', 'Schalke 04', 'Schalke'], 'FC Schalke 04')
    df['Away Team'] = df['Away Team'].replace(
        ['FC Schalke 04', 'Schalke 04', 'Schalke'], 'FC Schalke 04')
    df.to_csv(datos_limpios[i], index=False)
# Onceavo cambio todos los 'FC Zenit', 'Zenit' por 'FC Zenit'

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['FC Zenit', 'Zenit'], 'FC Zenit')
    df['Away Team'] = df['Away Team'].replace(
        ['FC Zenit', 'Zenit'], 'FC Zenit')
    df.to_csv(datos_limpios[i], index=False)

# Doceavo cambio todos los 'Juventus', 'Juventus Turin' por 'Juventus'

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Juventus', 'Juventus Turin'], 'Juventus')
    df['Away Team'] = df['Away Team'].replace(
        ['Juventus', 'Juventus Turin'], 'Juventus')
    df.to_csv(datos_limpios[i], index=False)

# Treceavo cambio todos los 'Manchester City', 'Manchester City FC', 'Man City', 'Man. City' por 'Manchester City'

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Manchester City', 'Manchester City FC', 'Man City', 'Man. City'], 'Manchester City')
    df['Away Team'] = df['Away Team'].replace(
        ['Manchester City', 'Manchester City FC', 'Man City', 'Man. City'], 'Manchester City')
    df.to_csv(datos_limpios[i], index=False)

# Catorceavo cambio todos los 'Manchester United', 'Man United', 'Man. United', 'Manchester United FC' por 'Manchester United'

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Manchester United', 'Man United', 'Man. United', 'Manchester United FC'], 'Manchester United')
    df['Away Team'] = df['Away Team'].replace(
        ['Manchester United', 'Man United', 'Man. United', 'Manchester United FC'], 'Manchester United')
    df.to_csv(datos_limpios[i], index=False)

# Quinceavo cambio todos los 'Olympiakos', 'Olympiakos Piraeus' por 'Olympiakos'

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Olympiakos', 'Olympiakos Piraeus'], 'Olympiakos')
    df['Away Team'] = df['Away Team'].replace(
        ['Olympiakos', 'Olympiakos Piraeus'], 'Olympiakos')
    df.to_csv(datos_limpios[i], index=False)

# Dieciseisavo cambio todos los 'Inter", "Internazionale' por 'Internazionale"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Inter', 'Internazionale'], 'Internazionale')
    df['Away Team'] = df['Away Team'].replace(
        ['Inter', 'Internazionale'], 'Internazionale')
    df.to_csv(datos_limpios[i], index=False)

# Diecisieteavo cambio todos los 'Milan', 'AC Milan' por 'AC Milan"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Milan', 'AC Milan'], 'AC Milan')
    df['Away Team'] = df['Away Team'].replace(
        ['Milan', 'AC Milan'], 'AC Milan')
    df.to_csv(datos_limpios[i], index=False)

# Dieciochoavo cambio todos los 'Tottenham', 'Tottenham Hotspur' por 'Tottenham'

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Tottenham', 'Tottenham Hotspur'], 'Tottenham')
    df['Away Team'] = df['Away Team'].replace(
        ['Tottenham', 'Tottenham Hotspur'], 'Tottenham')
    df.to_csv(datos_limpios[i], index=False)

# Diecinueveavo cambio todos los Monaco', 'AS Monaco' por 'AS Monaco"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Monaco', 'AS Monaco'], 'AS Monaco')
    df['Away Team'] = df['Away Team'].replace(
        ['Monaco', 'AS Monaco'], 'AS Monaco')
    df.to_csv(datos_limpios[i], index=False)

# Veinteavo cambio todos los 'Borussia Mönchengladbach', 'Mönchengladbach' por 'Borussia Mönchengladbach"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Borussia Mönchengladbach', 'Mönchengladbach'], 'Borussia Mönchengladbach')
    df['Away Team'] = df['Away Team'].replace(
        ['Borussia Mönchengladbach', 'Mönchengladbach'], 'Borussia Mönchengladbach')
    df.to_csv(datos_limpios[i], index=False)

# Veintiunavo cambio todos los 'Paris Saint-Germain', 'Paris SG', 'PSG' por 'Paris Saint-Germain"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Paris Saint-Germain', 'Paris SG', 'PSG', 'Paris'], 'Paris Saint-Germain')
    df['Away Team'] = df['Away Team'].replace(
        ['Paris Saint-Germain', 'Paris SG', 'PSG', 'Paris'], 'Paris Saint-Germain')
    df.to_csv(datos_limpios[i], index=False)

# Veintidosavo cambio todos los 'Real Madrid', 'Real Madrid CF' por 'Real Madrid"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Real Madrid', 'Real Madrid CF'], 'Real Madrid')
    df['Away Team'] = df['Away Team'].replace(
        ['Real Madrid', 'Real Madrid CF'], 'Real Madrid')
    df.to_csv(datos_limpios[i], index=False)

# Veintitresavo cambio todos los 'Roma', 'AS Roma' por 'AS Roma"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Roma', 'AS Roma'], 'AS Roma')
    df['Away Team'] = df['Away Team'].replace(
        ['Roma', 'AS Roma'], 'AS Roma')
    df.to_csv(datos_limpios[i], index=False)

# Veinticuatroavo cambio todos los 'Sevilla', 'Sevilla FC' por 'Sevilla"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Sevilla', 'Sevilla FC'], 'Sevilla')
    df['Away Team'] = df['Away Team'].replace(
        ['Sevilla', 'Sevilla FC'], 'Sevilla')
    df.to_csv(datos_limpios[i], index=False)

# Veinticincoavo cambio todos los 'Shakhtar Donetsk', 'Shakhtar' por 'Shakhtar Donetsk"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Shakhtar Donetsk', 'Shakhtar'], 'Shakhtar Donetsk')
    df['Away Team'] = df['Away Team'].replace(
        ['Shakhtar Donetsk', 'Shakhtar'], 'Shakhtar Donetsk')
    df.to_csv(datos_limpios[i], index=False)

# Veintiseisavo cambio todos los 'Valencia', 'Valencia CF' por 'Valencia"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Valencia', 'Valencia CF'], 'Valencia')
    df['Away Team'] = df['Away Team'].replace(
        ['Valencia', 'Valencia CF'], 'Valencia')
    df.to_csv(datos_limpios[i], index=False)

# Veintisieteavo cambio todos los 'Lepizig', 'RB Leipzig' por 'RB Leipzig"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Lepizig', 'RB Leipzig'], 'RB Leipzig')
    df['Away Team'] = df['Away Team'].replace(
        ['Lepizig', 'RB Leipzig'], 'RB Leipzig')
    df.to_csv(datos_limpios[i], index=False)

# Veintiochoavo cambio todos los 'Atalanta', 'Atalanta BC' por 'Atalanta"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Atalanta', 'Atalanta BC'], 'Atalanta')
    df['Away Team'] = df['Away Team'].replace(
        ['Atalanta', 'Atalanta BC'], 'Atalanta')
    df.to_csv(datos_limpios[i], index=False)

# Veintinueveavo cambio todos los 'Ajax', 'AFC Ajax' por 'Ajax"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(['Ajax', 'AFC Ajax'], 'Ajax')
    df['Away Team'] = df['Away Team'].replace(['Ajax', 'AFC Ajax'], 'Ajax')
    df.to_csv(datos_limpios[i], index=False)

# Treintaavo cambio todos los 'Benfica', 'SL Benfica' por 'Benfica"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Benfica', 'SL Benfica'], 'Benfica')
    df['Away Team'] = df['Away Team'].replace(
        ['Benfica', 'SL Benfica'], 'Benfica')
    df.to_csv(datos_limpios[i], index=False)

# Treinta y unoavo cambio todos los 'APOEL', 'APOEL Nicosia', 'apoel' por 'Apoel"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['APOEL', 'APOEL Nicosia', 'apoel'], 'Apoel')
    df['Away Team'] = df['Away Team'].replace(
        ['APOEL', 'APOEL Nicosia', 'apoel'], 'Apoel')
    df.to_csv(datos_limpios[i], index=False)

# Treinta y dosavo cambio todos los 'PSV', 'PSV Eindhoven' por 'PSV"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['PSV', 'PSV Eindhoven'], 'PSV')
    df['Away Team'] = df['Away Team'].replace(
        ['PSV', 'PSV Eindhoven'], 'PSV')
    df.to_csv(datos_limpios[i], index=False)

# Treinta y tresavo cambio todos los 'Malmo', 'Malmö', 'Malmö FF' por 'Malmo FF"

for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    df['Home Team'] = df['Home Team'].replace(
        ['Malmo', 'Malmö', 'Malmö FF'], 'Malmo FF')
    df['Away Team'] = df['Away Team'].replace(
        ['Malmo', 'Malmö', 'Malmö FF'], 'Malmo FF')
    df.to_csv(datos_limpios[i], index=False)

print("Equipos limpios\n")
equipos = set()
for i in range(len(datos_limpios)):
    df = pd.read_csv(datos_limpios[i])
    equipos = equipos.union(set(df['Home Team']))
    equipos = equipos.union(set(df['Away Team']))

equipos = sorted(equipos)
print(equipos)


# crea la columna "Home Goals" y "Away Goals" en cada archivo CSV que sea la división de la columna "Result" en dos columnas
for archivo in datos_limpios:
    df = pd.read_csv(archivo)
    if 'Result' in df.columns:
        df[['Home Goals', 'Away Goals']
           ] = df['Result'].str.split('-', expand=True)
        df = df.drop(['Result'], axis=1)
        df.to_csv(archivo, index=False)
    else:
        print(f"El archivo {archivo} no tiene una columna 'Result'.")

# crear un csv de equipos con goles totales según los datos
diccionario_equipos_goles = {}

# Cálculo de los goles totales por equipo
for archivo in datos_limpios:
    df = pd.read_csv(archivo)
    for index, row in df.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        home_goals = int(row['Home Goals'])
        away_goals = int(row['Away Goals'])

        if home_team not in diccionario_equipos_goles:
            diccionario_equipos_goles[home_team] = 0
        if away_team not in diccionario_equipos_goles:
            diccionario_equipos_goles[away_team] = 0

        diccionario_equipos_goles[home_team] += home_goals
        diccionario_equipos_goles[away_team] += away_goals

# Guardar el diccionario de equipos con goles totales en un archivo CSV y ordenar el diccionario de mayor goles a menor
df_equipos_goles = pd.DataFrame(diccionario_equipos_goles.items(), columns=[
                                'Equipo', 'Goles'])
df_equipos_goles = df_equipos_goles.sort_values(
    by='Goles', ascending=False).reset_index(drop=True)
df_equipos_goles.to_csv('Datos_Limpios/Goles.csv', index=False)


# Guardar el dataframe en un archivo CSV
df_trabajar.to_csv("Datos_Limpios/UCL" +
                   nombres_goles[i] + ".csv", index=False)
