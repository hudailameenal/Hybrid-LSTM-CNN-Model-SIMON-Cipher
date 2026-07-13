from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout

def build_model(hidden_units=64, dropout_rate=0.5):
    model = Sequential([
        Bidirectional(LSTM(hidden_units, return_sequences=True, activation='tanh'), input_shape=(8, 1)),
        Dropout(dropout_rate),
        Bidirectional(LSTM(hidden_units, activation='tanh')),
        Dropout(dropout_rate),
        Dense(8, activation='linear')  # Predict plaintext bytes
    ])
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
