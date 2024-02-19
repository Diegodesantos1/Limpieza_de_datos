import numpy as np
import pandas as pd


def limpieza():

    datos = ["Datos/champions-league-2017-UTC.csv", "Datos/champions-league-2018-UTC.csv", "Datos/champions-league-2019-UTC.csv",
             "Datos/champions-league-2020-UTC.csv", "Datos/champions-league-2021-UTC.csv", "Datos/champions-league-2022-UTC.csv",
             "Datos/champions-league-2023-UTC.csv"]

    nombres = ["2017-2018", "2018-2019", "2019-2020",
               "2020-2021", "2021-2022", "2022-2023", "2023-2024"]

    for i in range(len(datos)):
        # Leer el archivo CSV
        df = pd.read_csv(datos[i])

        # Crear una copia del dataframe
        df_trabajar = df.copy()

        # Eliminar las columnas Match Number, Date, Location, Group
        df_trabajar = df_trabajar.drop(
            ['Match Number', 'Date', 'Location', 'Group'], axis=1)

        # Definir una función para obtener el ganador
        def obtener_ganador(row):
            # Comprobar si el valor en 'Result' es NaN
            if pd.isna(row['Result']):
                return 'Draw'

            # Dividir el resultado en el signo "-"
            resultados = row['Result'].split('-')
            # Convertir los resultados a enteros
            resultados = [int(resultado) for resultado in resultados]
            # Obtener el máximo resultado
            max_resultado = max(resultados)
            # Determinar el ganador según el índice del máximo resultado
            ganador = row['Home Team'] if resultados.index(
                max_resultado) == 0 else row['Away Team']
            return ganador if resultados[0] != resultados[1] else 'Draw'

        # Aplicar la función para obtener el ganador y crear la columna "Winner"
        df_trabajar['Winner'] = df_trabajar.apply(obtener_ganador, axis=1)

        # Guardar el dataframe en un archivo CSV
        df_trabajar.to_csv("Datos_Limpios/UCL" +
                           nombres[i] + ".csv", index=False)

    datos_limpios = ["Datos_Limpios/UCL2014-2015.csv", "Datos_Limpios/UCL2015-2016.csv", "Datos_Limpios/UCL2016-2017.csv", "Datos_Limpios/UCL2017-2018.csv", "Datos_Limpios/UCL2018-2019.csv", "Datos_Limpios/UCL2019-2020.csv",
                     "Datos_Limpios/UCL2020-2021.csv", "Datos_Limpios/UCL2021-2022.csv", "Datos_Limpios/UCL2022-2023.csv", "Datos_Limpios/UCL2023-2024.csv"]

    # printeame todos los nombres de los equipos desde 2014-2015 hasta 2023-2024

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
            ['Paris Saint-Germain', 'Paris SG', 'PSG'], 'Paris Saint-Germain')
        df['Away Team'] = df['Away Team'].replace(
            ['Paris Saint-Germain', 'Paris SG', 'PSG'], 'Paris Saint-Germain')
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

    # crea la columna "Home Goals" y "Away Goals" en cada archivo CSV

    for i in range(len(datos_limpios)):
        df = pd.read_csv(datos_limpios[i])
        df['Home Goals'] = np.nan
        df['Away Goals'] = np.nan
        df.to_csv(datos_limpios[i], index=False)

    # Ahora voy a crear un diccionario con los goles de cada equipo

    goles = {}
    for equipo in equipos:
        goles[equipo] = 0

    for i in range(len(datos_limpios)):
        df = pd.read_csv(datos_limpios[i])
        for index, row in df.iterrows():
            goles[row['Home Team']] += row['Home Goals']
            goles[row['Away Team']] += row['Away Goals']

    print("\nGoles de los equipos ordenados\n")

    goles_ordenados = sorted(goles.items(), key=lambda x: x[1], reverse=True)

    print(goles_ordenados)

    # Ahora voy a guardar los goles en un archivo CSV

    equipos = []
    goles = []
    for equipo, gol in goles_ordenados:
        equipos.append(equipo)
        goles.append(gol)

    df = pd.DataFrame({'Equipo': equipos, 'Goles': goles})

    df.to_csv("Datos_Limpios/Goles.csv", index=False)

    # Voy a crear un sistema de puntuación para los equipos dando 2 puntos por victoria, 1 por empate y -1 por derrota

    puntos = {}
    for equipo in equipos:
        puntos[equipo] = 0

    for i in range(len(datos_limpios)):
        df = pd.read_csv(datos_limpios[i])
        for index, row in df.iterrows():
            if row['Winner'] == 'Draw':
                puntos[row['Home Team']] += 1
                puntos[row['Away Team']] += 1
            elif row['Winner'] == row['Home Team']:
                puntos[row['Home Team']] += 2
                puntos[row['Away Team']] -= 1
            else:
                puntos[row['Away Team']] += 2
                puntos[row['Home Team']] -= 1

    print("\nPuntos de los equipos ordenados\n")

    puntos_ordenados = sorted(puntos.items(), key=lambda x: x[1], reverse=True)

    print(puntos_ordenados)

    # Ahora voy a guardar los puntos en un archivo CSV

    equipos = []
    puntos = []
    for equipo, punto in puntos_ordenados:
        equipos.append(equipo)
        puntos.append(punto)

    df = pd.DataFrame({'Equipo': equipos, 'Puntos': puntos})
    df.to_csv("Datos_Limpios/Puntaje.csv", index=False)

    datos = ["Datos_Limpios/UCL2014-2015.csv","Datos_Limpios/UCL2015-2016.csv","Datos_Limpios/UCL2016-2017.csv","Datos/champions-league-2017-UTC.csv", "Datos/champions-league-2018-UTC.csv", "Datos/champions-league-2019-UTC.csv",
             "Datos/champions-league-2020-UTC.csv", "Datos/champions-league-2021-UTC.csv", "Datos/champions-league-2022-UTC.csv",
             "Datos/champions-league-2023-UTC.csv"]

    nombres = ["2014-2015","2015-2016","2016-2017","2017-2018", "2018-2019", "2019-2020",
               "2020-2021", "2021-2022", "2022-2023", "2023-2024"]

    # Diccionario para almacenar los goles totales de cada equipo
    goles_totales = {}

    for i in range(len(datos)):
        df = pd.read_csv(datos[i])

        # Crear una copia del dataframe
        df_trabajar = df.copy()

        # Eliminar las columnas Match Number, Date, Location, Group si existen

        if 'Match Number' in df_trabajar.columns:
            df_trabajar = df_trabajar.drop(['Match Number'], axis=1)

        if 'Date' in df_trabajar.columns:
            df_trabajar = df_trabajar.drop(['Date'], axis=1)

        if 'Location' in df_trabajar.columns:
            df_trabajar = df_trabajar.drop(['Location'], axis=1)

        if 'Group' in df_trabajar.columns:
            df_trabajar = df_trabajar.drop(['Group'], axis=1)

        # Dividir el resultado en goles de local y visitante
        df_trabajar[['Home Goals', 'Away Goals']
                    ] = df_trabajar['Result'].str.split('-', expand=True)
        df_trabajar['Home Goals'] = df_trabajar['Home Goals'].astype(float)
        df_trabajar['Away Goals'] = df_trabajar['Away Goals'].astype(float)

        # Calcular los goles totales de cada equipo
        for index, row in df_trabajar.iterrows():
            home_team = row['Home Team']
            away_team = row['Away Team']
            if home_team not in goles_totales:
                goles_totales[home_team] = {'a_favor': 0, 'en_contra': 0}
            if away_team not in goles_totales:
                goles_totales[away_team] = {'a_favor': 0, 'en_contra': 0}
            goles_totales[home_team]['a_favor'] += row['Home Goals']
            goles_totales[home_team]['en_contra'] += row['Away Goals']
            goles_totales[away_team]['a_favor'] += row['Away Goals']
            goles_totales[away_team]['en_contra'] += row['Home Goals']

        # Guardar el dataframe en un archivo CSV
        df_trabajar.to_csv("Datos_Limpios/UCL" +
                           nombres[i] + ".csv", index=False)

    # Crear un dataframe para los goles totales
    goles_df = pd.DataFrame(goles_totales).transpose().reset_index()
    goles_df.columns = ['Equipo', 'Goles a Favor', 'Goles en Contra']

    # Guardar los goles totales en un archivo CSV
    goles_df.to_csv("Datos_Limpios/Goles.csv", index=False)

    # Resto del código para limpiar los nombres de los equipos y calcular puntos, etc.
