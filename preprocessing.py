import os
import numpy as np
import csv
from sklearn.model_selection import train_test_split

def hex_to_byte_matrix(hex_list):
    """
    Converts list of hex strings (8 bytes each) into m×8 numpy array of uint8.
    """
    byte_matrix = []
    for h in hex_list:
        # Ensure 16-character hex string (8 bytes)
        h = h.zfill(16)
        b = bytes.fromhex(h)
        if len(b) != 8:
            raise ValueError(f"Invalid block length for {h} - expected 8 bytes, got {len(b)}")
        byte_matrix.append(list(b))
    return np.array(byte_matrix, dtype=np.uint8)

def preprocess_csv(dataset_path, add_noise=False, noise_std=0.0):
    """
    Load CSV dataset, convert hex strings to byte matrices, normalize, reshape for LSTM.
    Returns: (X, y) where:
        X = ciphertexts (samples, 8, 1)
        y = plaintexts (samples, 8)
    """
    plaintexts = []
    ciphertexts = []

    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            plaintexts.append(row['plaintext'])
            ciphertexts.append(row['ciphertext'])

    # Convert hex → bytes → matrix (m×8)
    X_bytes = hex_to_byte_matrix(ciphertexts)
    y_bytes = hex_to_byte_matrix(plaintexts)

    # Normalize to [0,1]
    X = X_bytes.astype(np.float32) / 255.0
    y = y_bytes.astype(np.float32) / 255.0

    # Optional noise
    if add_noise:
        noise = np.random.normal(0, noise_std, X.shape)
        X = np.clip(X + noise, 0, 1)

    # Reshape for LSTM input: (samples, timesteps=8, features=1)
    X = X.reshape((X.shape[0], 8, 1))

    return X, y

def load_all_data(data_dir=None):
    """
    Load all datasets (Dataset-1 to Dataset-7) from CSV format.
    Returns a dict: { "dataset1": (X_train, X_test, y_train, y_test), ... }
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "datasets")

    datasets = {}
    dataset_names = [f"Dataset-{i}" for i in range(1, 8)]

    for name in dataset_names:
        file_path = os.path.join(data_dir, f"{name}.csv")
        print(f"📂 Loading {file_path} ...")
        X, y = preprocess_csv(file_path)

        if name == "Dataset-1":
            # Sequential 90–10 split for CTT dataset (incremental)
            split_index = int(len(X) * 0.9)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
        else:
            # Random 90–10 split for all others
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, shuffle=True, random_state=42
            )

        datasets[name.lower()] = (X_train, X_test, y_train, y_test)

    return datasets

if __name__ == "__main__":
    # Test loading
    all_data = load_all_data()
    for name, (X_train, X_test, y_train, y_test) in all_data.items():
        print(f"{name}:")
        print(f"  Train shapes: X={X_train.shape}, y={y_train.shape}")
        print(f"  Test shapes: X={X_test.shape}, y={y_test.shape}")
        print(f"  Sample X_train[0]: {X_train[0].flatten()[:5]}...")
        print(f"  Sample y_train[0]: {y_train[0][:5]}...\n")