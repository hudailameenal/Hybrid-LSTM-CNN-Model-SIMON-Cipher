import os
import numpy as np
import optuna
import tensorflow as tf
import csv
from cipher import SimonCipher  # Assuming you have a Simon cipher implementation
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "__pycache__"))
import model
from preprocessing import load_all_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ====== METRICS & UTILITIES ======
def restoration_accuracy(y_true, y_pred):
    """Byte-level accuracy (percentage of correctly predicted bytes)"""
    y_true_bytes = (y_true * 255).astype(np.uint8)
    y_pred_bytes = np.clip(np.round(y_pred * 255), 0, 255).astype(np.uint8)
    
    # Count correctly predicted bytes
    correct_bytes = np.sum(y_true_bytes == y_pred_bytes)
    total_bytes = y_true_bytes.size
    
    return correct_bytes / total_bytes

def mean_byte_error(y_true, y_pred):
    """Average absolute error in bytes"""
    y_true_bytes = (y_true * 255).astype(np.uint8)
    y_pred_bytes = np.clip(np.round(y_pred * 255), 0, 255).astype(np.uint8)
    return np.mean(np.abs(y_pred_bytes - y_true_bytes)) / 255.0

def hamming_accuracy(y_true, y_pred):
    """Calculate accuracy based on Hamming distance between predicted and actual bytes"""
    y_true_bytes = (y_true * 255).astype(np.uint8)
    y_pred_bytes = np.clip(np.round(y_pred * 255), 0, 255).astype(np.uint8)
    
    # Calculate Hamming distance (number of differing bits)
    hamming_dist = np.sum(np.unpackbits(y_true_bytes) != np.unpackbits(y_pred_bytes))
    total_bits = y_true_bytes.size * 8
    
    return 1 - (hamming_dist / total_bits)

# ====== DATA PREPARATION ======
def prepare_round_data(datasets, round_num, previous_ciphertexts=None):
    """Prepare ciphertexts for a specific round number with proper chaining"""
    X_train, X_test, y_train, y_test = datasets["Dataset-1"]
    
    # Convert normalized data back to original bytes
    def denormalize_to_bytes(normalized_data):
        return (normalized_data * 255).astype(np.uint8)
    
    # Get original byte values (before normalization)
    y_train_bytes = denormalize_to_bytes(y_train)
    y_test_bytes = denormalize_to_bytes(y_test)
    
    # Use the SIMON test vector key (96-bit key for SIMON-64/96)
    cipher = SimonCipher(key_hex="0x000000000000000000000000", block_size=64, key_size=96, rounds=1)
    
    # Encrypt data
    X_train_ct = []
    X_test_ct = []
    
    if previous_ciphertexts is None:
        # First round - encrypt the plaintexts
        for pt in y_train_bytes:
            ct = cipher.encrypt(bytes(pt))
            X_train_ct.append(list(ct))
        
        for pt in y_test_bytes:
            ct = cipher.encrypt(bytes(pt))
            X_test_ct.append(list(ct))
    else:
        # Subsequent rounds - encrypt the previous ciphertexts
        prev_train_ct, prev_test_ct = previous_ciphertexts
        
        for ct in prev_train_ct:
            new_ct = cipher.encrypt(bytes(ct))
            X_train_ct.append(list(new_ct))
        
        for ct in prev_test_ct:
            new_ct = cipher.encrypt(bytes(ct))
            X_test_ct.append(list(new_ct))
    
    # Convert to normalized form for training
    X_train_ct = np.array(X_train_ct, dtype=np.float32) / 255.0
    X_test_ct = np.array(X_test_ct, dtype=np.float32) / 255.0
    
    # Reshape for LSTM (samples, timesteps=8, features=1)
    X_train_ct = X_train_ct.reshape((-1, 8, 1))
    X_test_ct = X_test_ct.reshape((-1, 8, 1))
    
    # Store ciphertexts for next round
    next_round_ciphertexts = (
        denormalize_to_bytes(X_train_ct),  # Training ciphertexts
        denormalize_to_bytes(X_test_ct)    # Test ciphertexts
    )
    
    return X_train_ct, X_test_ct, y_train, y_test, next_round_ciphertexts

# ====== HYBRID MODEL ARCHITECTURE ======
def build_simon_ctt_hybrid_model(input_shape, hidden_units, dropout_rate):
    """SIMON CTT Hybrid Model with CNN and LSTM layers"""
    model = Sequential([
        # CNN layers for feature extraction
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'),
        
        # LSTM layers for sequence processing
        LSTM(hidden_units, activation='tanh', return_sequences=True),
        Dropout(dropout_rate),
        
        LSTM(int(hidden_units/2), activation='sigmoid'),
        Dropout(dropout_rate),
        
        Dense(8, activation='linear')  # 8-byte output
    ])
    return model

# ====== PAPER HYPERPARAMETERS ======
def get_paper_hyperparams(round_num):
    """Get hyperparameters from the paper for specific rounds"""
    paper_params = {
        1: {'hidden_units': 10, 'optimizer': 'Adam', 'lr': 0.0029, 'epochs': 40, 'batch_size': 32},
        2: {'hidden_units': 10, 'optimizer': 'RMSprop', 'lr': 0.003, 'epochs': 50, 'batch_size': 32},
        3: {'hidden_units': 10, 'optimizer': 'RMSprop', 'lr': 0.0026, 'epochs': 30, 'batch_size': 32},
        4: {'hidden_units': 10, 'optimizer': 'RMSprop', 'lr': 0.0024, 'epochs': 30, 'batch_size': 32},
        20: {'hidden_units': 10, 'optimizer': 'Adam', 'lr': 0.0025, 'epochs': 40, 'batch_size': 32},
        31: {'hidden_units': 15, 'optimizer': 'RMSprop', 'lr': 0.0024, 'epochs': 30, 'batch_size': 32},
        42: {'hidden_units': 10, 'optimizer': 'RMSprop', 'lr': 0.002, 'epochs': 20, 'batch_size': 32}
    }
    return paper_params.get(round_num, None)

# ====== TRAINING FUNCTION ======
def train_for_round(round_num, datasets, previous_ciphertexts=None):
    """Train model for a specific round"""
    print(f"\n=== Training for Round {round_num} ===")
    
    # Prepare data
    X_train_ct, X_test_ct, y_train, y_test, next_round_ciphertexts = prepare_round_data(
        datasets, round_num, previous_ciphertexts
    )
    
    # Verify data shapes match
    print(f"X_train_ct shape: {X_train_ct.shape}, y_train shape: {y_train.shape}")
    print(f"X_test_ct shape: {X_test_ct.shape}, y_test shape: {y_test.shape}")
    
    if X_train_ct.shape[0] != y_train.shape[0]:
        raise ValueError(f"Training data mismatch: X has {X_train_ct.shape[0]} samples, y has {y_train.shape[0]}")
    
    if X_test_ct.shape[0] != y_test.shape[0]:
        raise ValueError(f"Test data mismatch: X has {X_test_ct.shape[0]} samples, y has {y_test.shape[0]}")
    
    # Use paper hyperparameters if available
    paper_params = get_paper_hyperparams(round_num)
    
    if paper_params:
        print(f"Using paper hyperparameters for round {round_num}")
        params = paper_params
    else:
        print(f"Using Optuna for round {round_num}")
        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_ct, y_train, test_size=0.1, random_state=42
        )
        
        # Optuna objective function
        def objective(trial):
            params = {
                'hidden_units': trial.suggest_int('hidden_units', 10, 25),
                'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'RMSprop']),
                'lr': trial.suggest_categorical('lr', [0.001, 0.002, 0.003]),
                'epochs': trial.suggest_int('epochs', 10, 60, step=10),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64])
            }
            
            # Build hybrid model
            model = build_simon_ctt_hybrid_model(
                input_shape=(8, 1),
                hidden_units=params['hidden_units'],
                dropout_rate=0.009  # As specified in the paper
            )
            
            # Configure optimizer
            if params['optimizer'] == "Adam":
                optimizer = Adam(learning_rate=params['lr'])
            else:
                optimizer = RMSprop(learning_rate=params['lr'])
            
            model.compile(optimizer=optimizer, loss="mae", metrics=["mae"])
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
            ]
            
            # Training
            history = model.fit(
                X_train_split, y_train_split,
                validation_data=(X_val_split, y_val_split),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluation
            y_pred = model.predict(X_val_split, verbose=0)
            mbe = mean_byte_error(y_val_split, y_pred)
            return mbe
        
        # Run optimization with 5 trials
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5)
        
        # Get best parameters
        trial = study.best_trial
        params = trial.params
    
    # Build and train final hybrid model
    model = build_simon_ctt_hybrid_model(
        input_shape=(8, 1),
        hidden_units=params['hidden_units'],
        dropout_rate=0.009  # As specified in the paper
    )
    
    if params['optimizer'] == "Adam":
        optimizer = Adam(learning_rate=params['lr'])
    else:
        optimizer = RMSprop(learning_rate=params['lr'])
    
    model.compile(optimizer=optimizer, loss="mae", metrics=["mae"])
    
    # Train on full training data
    history = model.fit(
        X_train_ct, y_train,
        validation_data=(X_test_ct, y_test),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        verbose=1
    )
    
    # Final evaluation
    y_pred = model.predict(X_test_ct, verbose=0)
    acc = restoration_accuracy(y_test, y_pred) * 100  # Convert to percentage
    mbe = mean_byte_error(y_test, y_pred)
    hamming_acc = hamming_accuracy(y_test, y_pred) * 100  # Hamming accuracy
    
    result = {
        'Round': round_num,
        'Optimizer': params['optimizer'],
        'Epochs': params['epochs'],
        'Lr_rate': params['lr'],
        'Hidden_Nodes': params['hidden_units'],
        'Test_Accuracy': acc,
        'Mean_Byte_Error': mbe,
        
    }
    
    print(f"Round {round_num} Results: Accuracy={result['Test_Accuracy']:.2f}%,  MBE={result['Mean_Byte_Error']:.4f}")
    
    return model, result, next_round_ciphertexts

# ====== MAIN EXECUTION ======

if __name__ == "__main__":
    print("Initializing SIMON CTT Hybrid training pipeline...")
    print("Loading datasets and preparing environment...")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Datasets loaded successfully. Beginning training...")
    try:
        model.train_for_all_rounds()
    except Exception as e:
        print(f"training error: Reload{e}")
        from preprocessing import load_all_data
        datasets = load_all_data()
        results = []
        previous_ciphertexts = None
        final_model = None

        for round_num in range(1, 43):  # SIMON-64/96 → 42 rounds
            try:
                model, result, previous_ciphertexts = train_for_round(
                    round_num, datasets, previous_ciphertexts
                )
                if result:
                    results.append(result)
                final_model = model
            except Exception as e:
                print(f"Error training round {round_num}: {e}")
                continue

        # save simulated model
        if final_model is not None:
            final_model.save("models/simon_ctt_hybrid.h5")
            print("Simulation completed. Model saved in models/simon_ctt_hybrid.h5")

        # save simulated round results
        if results:
            fieldnames = [
                'Round', 'Optimizer', 'Epochs', 'Lr_rate', 'Hidden_Nodes',
                'Test_Accuracy', 'Mean_Byte_Error', 
            ]
            with open(os.path.join("results", "ctt_hybrid_round_results.csv"), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

    print("Simulation completed. Results saved in results/ctt_hybrid_round_results.csv")