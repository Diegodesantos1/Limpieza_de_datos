import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


class ChampionsLeaguePredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()

    def load_data(self, file_path):
        data = pd.read_csv(file_path)

        # Codificar los nombres de los equipos
        data['Home Team'] = self.label_encoder.fit_transform(data['Home Team'])
        data['Away Team'] = self.label_encoder.transform(data['Away Team'])

        # Crear una columna para representar el resultado del partido
        data['Result'] = data['Home Goals'] - data['Away Goals']
        data['Result'] = data['Result'].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        # Dividir los datos en características (X) y variable objetivo (y)
        X = data[['Home Team', 'Away Team']]
        y = data['Result']

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train, X_test, y_test, epochs=100):
        # Definir el modelo de red neuronal
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='tanh', input_shape=(2,)),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compilar el modelo
        self.model.compile(optimizer='sgd', loss='binary_crossentropy',
                           metrics=['accuracy'])

        # Entrenar el modelo y guardar el historial del entrenamiento en una variable
        history = self.model.fit(X_train, y_train, epochs=epochs,
                                 batch_size=32, validation_data=(X_test, y_test))

        return history

    def predict_match_result(self, home_team, away_team):
        home_team_encoded = self.label_encoder.transform([home_team])
        away_team_encoded = self.label_encoder.transform([away_team])
        match_data = pd.DataFrame({'Home Team': home_team_encoded, 'Away Team': away_team_encoded})
        match_data = match_data.values.reshape(1, -1)
        result_probability = self.model.predict(match_data)
        return result_probability[0][0]

    def determine_winner(self, home_team, away_team, home_goals, away_goals):
        if home_goals > away_goals:
            return home_team
        elif away_goals > home_goals:
            return away_team
        else:
            return "Empate"

    def plot_training_history(self, history):
        plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
        plt.plot(history.history['val_loss'], label='Pérdida en validación')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.show()

        plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
        plt.plot(history.history['val_accuracy'], label='Precisión en validación')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()
        plt.show()


# Ejemplo de uso:
predictor = ChampionsLeaguePredictor()
X_train, X_test, y_train, y_test = predictor.load_data('Datos_Limpios/UCL2014-2015.csv')
history = predictor.train_model(X_train, y_train, X_test, y_test, epochs=100)
predictor.plot_training_history(history)
