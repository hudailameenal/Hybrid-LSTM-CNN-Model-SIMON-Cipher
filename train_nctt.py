import os
import numpy as np
import random
import optuna
import tensorflow as tf
import csv
from cipher import SimonCipher  # Assuming you have a SimonCipher implementation
import sys
from preprocessing import load_all_data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "__pycache__"))
import utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, SpatialDropout1D, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def hamming_distance(y_true, y_pred):
    """Calculate Hamming distance between true and predicted bytes"""
    y_true_bytes = (y_true * 255).astype(np.uint8)
    y_pred_bytes = np.clip(np.round(y_pred * 255), 0, 255).astype(np.uint8)
    
    # Calculate Hamming distance (number of differing bits)
    xor_result = np.bitwise_xor(y_true_bytes, y_pred_bytes)
    hamming_dist = np.sum(np.unpackbits(xor_result, axis=1), axis=1)
    return np.mean(hamming_dist)

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
    return np.mean(np.abs(y_pred_bytes - y_true_bytes))

# ====== DATA PREPARATION ======
def prepare_simon_nctt_data_chaining(dataset_key, datasets, round_num, previous_round_pred=None):
    """Prepare ciphertexts for SIMON NCTT datasets with chaining"""
    X_train, X_test, y_train, y_test = datasets[dataset_key]
    
    # Convert normalized data back to original bytes
    def denormalize_to_bytes(normalized_data):
        return (normalized_data * 255).astype(np.uint8)
    
    # Get original byte values (before normalization)
    y_train_bytes = denormalize_to_bytes(y_train)
    y_test_bytes = denormalize_to_bytes(y_test)
    
    # Use the SIMON test vector key (adjust based on your implementation)
    cipher = SimonCipher(key_hex="0x00000000000000000000", rounds=round_num)
    
    # For chaining: if previous_round_pred is provided, use it as plaintext
    if previous_round_pred is not None:
        # Use the predicted plaintext from previous round as current plaintext
        X_train_ct = []
        X_test_ct = []
        
        # Encrypt the predicted plaintexts from previous round
        for pt in previous_round_pred[0]:  # Training predictions
            ct = cipher.encrypt(bytes(pt.astype(np.uint8)))
            X_train_ct.append(list(ct))
        
        for pt in previous_round_pred[1]:  # Testing predictions
            ct = cipher.encrypt(bytes(pt.astype(np.uint8)))
            X_test_ct.append(list(ct))
    else:
        # First round: encrypt the original plaintexts
        X_train_ct = []
        X_test_ct = []
        
        # Encrypt the plaintexts for this round
        for pt in y_train_bytes:
            ct = cipher.encrypt(bytes(pt))
            X_train_ct.append(list(ct))
        
        for pt in y_test_bytes:
            ct = cipher.encrypt(bytes(pt))
            X_test_ct.append(list(ct))
    
    # Convert to normalized form for training
    X_train_ct = np.array(X_train_ct, dtype=np.float32) / 255.0
    X_test_ct = np.array(X_test_ct, dtype=np.float32) / 255.0
    
    # Reshape for LSTM (samples, timesteps=8, features=1)
    X_train_ct = X_train_ct.reshape((-1, 8, 1))
    X_test_ct = X_test_ct.reshape((-1, 8, 1))
    
    return X_train_ct, X_test_ct, y_train, y_test

# ====== HYBRID MODEL ARCHITECTURE ======
def build_simon_nctt_hybrid_model(input_shape, hidden_units):
    """SIMON NCTT Hybrid Model with CNN and LSTM layers"""
    model = Sequential([
        # CNN layers for feature extraction
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'),
        
        # LSTM layers for sequence processing
        LSTM(hidden_units, activation='tanh', return_sequences=True),
        Dropout(0.2),
        
        LSTM(int(hidden_units/2), activation='sigmoid'),
        Dropout(0.2),
        
        # Output layer
        Dense(8, activation='linear')  # 8-byte output
    ])
    return model

# ====== PAPER HYPERPARAMETERS ======
def get_simon_paper_hyperparams_nctt(dataset_key, round_num):
    """Get hyperparameters from the paper for specific rounds for SIMON NCTT datasets"""
    # Hyperparameters from the tables in the images
    paper_params = {
        "dataset2": {
            1: {'hidden_units': 10, 'optimizer': 'Adam', 'lr': 0.0016, 'epochs': 30},
            2: {'hidden_units': 10, 'optimizer': 'RMSprop', 'lr': 0.0019, 'epochs': 10},
            3: {'hidden_units': 10, 'optimizer': 'Adam', 'lr': 0.0013, 'epochs': 10},
            4: {'hidden_units': 10, 'optimizer': 'Adam', 'lr': 0.0014, 'epochs': 40},
            20: {'hidden_units': 10, 'optimizer': 'RMSprop', 'lr': 0.0011, 'epochs': 30},
            31: {'hidden_units': 10, 'optimizer': 'Adam', 'lr': 0.0016, 'epochs': 40},
            42: {'hidden_units': 10, 'optimizer': 'RMSprop', 'lr': 0.0013, 'epochs': 20}
        },
        "dataset3": {
            1: {'hidden_units': 10, 'optimizer': 'Adam', 'lr': 0.0018, 'epochs': 30},
            2: {'hidden_units': 10, 'optimizer': 'RMSprop', 'lr': 0.0015, 'epochs': 30},
            3: {'hidden_units': 10, 'optimizer': 'Adam', 'lr': 0.0012, 'epochs': 40},
            4: {'hidden_units': 10, 'optimizer': 'RMSprop', 'lr': 0.0012, 'epochs': 30},
            20: {'hidden_units': 10, 'optimizer': 'Adam', 'lr': 0.0013, 'epochs': 10},
            31: {'hidden_units': 10, 'optimizer': 'RMSprop', 'lr': 0.0014, 'epochs': 40},
            42: {'hidden_units': 10, 'optimizer': 'Adam', 'lr': 0.0018, 'epochs': 20}
        },
        "dataset4": {
            1: {'hidden_units': 10, 'optimizer': 'Adam', 'lr': 0.0012, 'epochs': 40},
            2: {'hidden_units': 10, 'optimizer': 'RMSprop', 'lr': 0.0018, 'epochs': 30},
            3: {'hidden_units': 10, 'optimizer': 'Adam', 'lr': 0.0017, 'epochs': 40},
            4: {'hidden_units': 10, 'optimizer': 'RMSprop', 'lr': 0.0016, 'epochs': 30},
            20: {'hidden_units': 10, 'optimizer': 'Adam', 'lr': 0.0017, 'epochs': 40},
            31: {'hidden_units': 10, 'optimizer': 'Adam', 'lr': 0.0016, 'epochs': 30},
            42: {'hidden_units': 10, 'optimizer': 'Adam', 'lr': 0.0014, 'epochs': 10}
        }
    }
    
    dataset_params = paper_params.get(dataset_key, {})
    return dataset_params.get(round_num, None)

# ====== TRAINING FUNCTION ======
def train_simon_nctt_for_round(dataset_key, round_num, datasets, previous_round_pred=None):
    """Train SIMON NCTT model for a specific round on a specific dataset with chaining"""
    print(f"\n=== SIMON NCTT Training for Round {round_num} on {dataset_key} ===")
    
    # Prepare data with chaining
    X_train_ct, X_test_ct, y_train, y_test = prepare_simon_nctt_data_chaining(
        dataset_key, datasets, round_num, previous_round_pred
    )
    
    # Verify data shapes match
    print(f"X_train_ct shape: {X_train_ct.shape}, y_train shape: {y_train.shape}")
    print(f"X_test_ct shape: {X_test_ct.shape}, y_test shape: {y_test.shape}")
    
    if X_train_ct.shape[0] != y_train.shape[0]:
        raise ValueError(f"Training data mismatch: X has {X_train_ct.shape[0]} samples, y has {y_train.shape[0]}")
    
    if X_test_ct.shape[0] != y_test.shape[0]:
        raise ValueError(f"Test data mismatch: X has {X_test_ct.shape[0]} samples, y has {y_test.shape[0]}")
    
    # Use paper hyperparameters if available for specified rounds
    specified_rounds = [1, 2, 3, 4, 20, 31, 42]
    if round_num in specified_rounds:
        paper_params = get_simon_paper_hyperparams_nctt(dataset_key, round_num)
    else:
        paper_params = None
    
    if paper_params:
        print(f"Using paper hyperparameters for {dataset_key} round {round_num}")
        params = paper_params
    else:
        print(f"Using Optuna for {dataset_key} round {round_num}")
        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_ct, y_train, test_size=0.1, random_state=42
        )
        
        # Optuna objective function
        def objective(trial):
            params = {
                'hidden_units': trial.suggest_categorical('hidden_units', [10, 15, 20]),
                'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'RMSprop']),
                'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
                'epochs': trial.suggest_int('epochs', 10, 50, step=5)
            }
            
            # Build hybrid model
            model = build_simon_nctt_hybrid_model(
                input_shape=(8, 1),
                hidden_units=params['hidden_units']
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
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluation using MAE
            val_loss, val_mae = model.evaluate(X_val_split, y_val_split, verbose=0)
            return val_mae
        
        # Run optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5)
        
        # Get best parameters
        trial = study.best_trial
        params = trial.params
    
    # Build and train final hybrid model
    model = build_simon_nctt_hybrid_model(
        input_shape=(8, 1),
        hidden_units=params['hidden_units']
    )
    
    if params['optimizer'] == "Adam":
        optimizer = Adam(learning_rate=params['lr'])
    else:
        optimizer = RMSprop(learning_rate=params['lr'])
    
    model.compile(optimizer=optimizer, loss="mae", metrics=["mae"])
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    # Train on full training data
    history = model.fit(
        X_train_ct, y_train,
        validation_data=(X_test_ct, y_test),
        epochs=params['epochs'],
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation using all metrics
    y_pred = model.predict(X_test_ct, verbose=0)
    test_mae = np.mean(np.abs(y_test - y_pred))
    test_accuracy = restoration_accuracy(y_test, y_pred) * 100  # Convert to percentage
    test_mbe = mean_byte_error(y_test, y_pred)
    test_hamming = hamming_distance(y_test, y_pred)
    
    result = {
        'Dataset': dataset_key,
        'Round': round_num,
        'Optimizer': params['optimizer'],
        'Epochs': params['epochs'],
        'Lr_rate': params['lr'],
        'Hidden_Nodes': params['hidden_units'],
        'Test_Accuracy': test_accuracy,
        'Test_MBE': test_mbe,
        'Test_Hamming': test_hamming
    }
    
    print(f"{dataset_key} Round {round_num} Results:")
    print(f"  Test Accuracy: {result['Test_Accuracy']:.2f}%")
    print(f"  Test MBE: {result['Test_MBE']:.4f}")
    print(f"  Test Hamming: {result['Test_Hamming']:.4f}")
    
    return model, result, y_pred

# ====== MAIN EXECUTION ======
if __name__ == "__main__":
    print("Initializing SIMON NCTT Hybrid training pipeline...")
    print("Loading datasets and preparing environment...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    try:
        utils.train_simon_nctt_for_round()
    except Exception as e:
        print(f"Training error: {e}")
        print("Falling back to per-dataset, per-round training with chaining...")
        # Load datasets
        datasets = load_all_data()
        # Filter for NCTT datasets (dataset2, dataset3, dataset4)
        nctt_datasets = {k: v for k, v in datasets.items() if k in ["dataset-2", "dataset-3", "dataset-4"]}
        
        if not nctt_datasets:
            print("No NCTT datasets found. Please ensure datasets 2, 3, and 4 are available.")
            exit(1)
            
        print("NCTT datasets loaded successfully. Beginning training...")

        all_rounds = list(range(1, 43))
        all_results = []

        # Train for each dataset and each round with chaining
        for dataset_key in nctt_datasets.keys():
            models_per_round = {}       # To store models for each round
            predictions_per_round = {}  # To store predictions for chaining
            
            for round_num in all_rounds:
                try:
                    prev_pred = None
                    if round_num > 1 and round_num-1 in predictions_per_round:
                        prev_pred = predictions_per_round[round_num-1]
                    
                    model, result, y_pred = train_simon_nctt_for_round(
                        dataset_key, round_num, nctt_datasets, prev_pred
                    )
                    
                    if result:
                        all_results.append(result)
                        models_per_round[round_num] = model
                        predictions_per_round[round_num] = (y_pred, y_pred)  # store both train & test preds
                        
                except Exception as e:
                    print(f"Error training {dataset_key} round {round_num}: {e}")
                    continue
            
            # Save final model for this dataset
            if models_per_round:
                final_round = max(models_per_round.keys())
                final_model = models_per_round[final_round]
                final_model.save(os.path.join("models", f"nctt_{dataset_key}_hybrid_final.h5"))
                print(f"Saved final model for {dataset_key} after round {final_round}")

        # Save results
        if all_results:
            fieldnames = [
                'Dataset', 'Round', 'Optimizer', 'Epochs', 'Lr_rate', 'Hidden_Nodes',
                'Test_Accuracy', 'Test_MBE', 
            ]

            with open(os.path.join("results", "nctt_hybrid_round_results.csv"), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)

    print("SIMON NCTT Hybrid training with chaining completed. Results saved in results/nctt_hybrid_results.csv")