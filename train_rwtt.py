import os
import numpy as np
import random
import optuna
import tensorflow as tf
import csv
from cipher import SimonCipher  # Assuming you have a Simon cipher implementation
import sys
from preprocessing import load_all_data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "__pycache__"))
import lib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, SpatialDropout1D, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

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
def prepare_simon_rwtt_data(dataset_key, datasets, round_num, previous_ciphertexts=None):
    """Prepare ciphertexts for RWTT datasets with proper chaining for Simon"""
    X_train, X_test, y_train, y_test = datasets[dataset_key]
    
    # Convert normalized data back to original bytes
    def denormalize_to_bytes(normalized_data):
        return (normalized_data * 255).astype(np.uint8)
    
    # Get original byte values (before normalization)
    y_train_bytes = denormalize_to_bytes(y_train)
    y_test_bytes = denormalize_to_bytes(y_test)
    
    # Use a fixed key as mentioned in the Simon specification
    # Simon-64/96 uses 96-bit key (12 bytes)
    key = b'\x00' * 12  # 12-byte key of zeros
    
    # Encrypt data
    X_train_ct = []
    X_test_ct = []
    
    if previous_ciphertexts is None:
        # First round - encrypt the plaintexts
        for pt in y_train_bytes:
            cipher = SimonCipher(key, round_num=round_num)  # Use current round number
            ct = cipher.encrypt(bytes(pt))
            X_train_ct.append(list(ct))
        
        for pt in y_test_bytes:
            cipher = SimonCipher(key, round_num=round_num)  # Use current round number
            ct = cipher.encrypt(bytes(pt))
            X_test_ct.append(list(ct))
    else:
        # Subsequent rounds - encrypt the previous ciphertexts
        prev_train_ct, prev_test_ct = previous_ciphertexts
        
        for ct in prev_train_ct:
            cipher = SimonCipher(key, round_num=round_num)  # Use current round number
            new_ct = cipher.encrypt(bytes(ct))
            X_train_ct.append(list(new_ct))
        
        for ct in prev_test_ct:
            cipher = SimonCipher(key, round_num=round_num)  # Use current round number
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
def build_simon_rwtt_hybrid_model(input_shape, hidden_units):
    """Simon RWTT Hybrid Model with CNN and LSTM layers"""
    model = Sequential([
        # CNN layers for feature extraction
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'),
        
        # LSTM layers for sequence processing
        LSTM(hidden_units, activation='tanh', return_sequences=True),
        SpatialDropout1D(0.4),
        
        LSTM(hidden_units, activation='tanh', return_sequences=True),
        SpatialDropout1D(0.4),
        
        LSTM(hidden_units, activation='tanh'),
        Dropout(0.4),
        
        # FC layers
        Dense(hidden_units, activation='relu'),
        Dropout(0.4),
        
        Dense(hidden_units, activation='relu'),
        Dropout(0.4),
        
        # Output layer
        Dense(8, activation='linear')  # 8-byte output
    ])
    return model

# ====== PAPER HYPERPARAMETERS ======
def get_paper_hyperparams_simon_rwtt(dataset_key, round_num):
    """Get hyperparameters from the paper for specific rounds for Simon RWTT datasets"""
    # Hyperparameters from the tables in the images
    paper_params = {
        "dataset5": {
            1: {'hidden_units': 25, 'optimizer': 'RMSprop', 'lr': 0.0018, 'epochs': 20},
            2: {'hidden_units': 25, 'optimizer': 'RMSprop', 'lr': 0.0011, 'epochs': 10},
            3: {'hidden_units': 25, 'optimizer': 'RMSprop', 'lr': 0.001, 'epochs': 20},
            4: {'hidden_units': 25, 'optimizer': 'Adam', 'lr': 0.0014, 'epochs': 20},
            20: {'hidden_units': 25, 'optimizer': 'RMSprop', 'lr': 0.0017, 'epochs': 20},
            31: {'hidden_units': 25, 'optimizer': 'RMSprop', 'lr': 0.0015, 'epochs': 10},
            42: {'hidden_units': 25, 'optimizer': 'RMSprop', 'lr': 0.0012, 'epochs': 20}
        },
        "dataset6": {
            1: {'hidden_units': 15, 'optimizer': 'RMSprop', 'lr': 0.0019, 'epochs': 40},
            2: {'hidden_units': 15, 'optimizer': 'Adam', 'lr': 0.0016, 'epochs': 30},
            3: {'hidden_units': 15, 'optimizer': 'Adam', 'lr': 0.0011, 'epochs': 20},
            4: {'hidden_units': 15, 'optimizer': 'Adam', 'lr': 0.0011, 'epochs': 10},
            20: {'hidden_units': 15, 'optimizer': 'Adam', 'lr': 0.001, 'epochs': 20},
            31: {'hidden_units': 15, 'optimizer': 'Adam', 'lr': 0.0019, 'epochs': 10},
            42: {'hidden_units': 15, 'optimizer': 'RMSprop', 'lr': 0.0011, 'epochs': 30}
        },
        "dataset7": {
            1: {'hidden_units': 25, 'optimizer': 'Adam', 'lr': 0.0016, 'epochs': 20},
            2: {'hidden_units': 25, 'optimizer': 'RMSprop', 'lr': 0.0013, 'epochs': 30},
            3: {'hidden_units': 25, 'optimizer': 'RMSprop', 'lr': 0.005, 'epochs': 20},
            4: {'hidden_units': 25, 'optimizer': 'RMSprop', 'lr': 0.0016, 'epochs': 40},
            20: {'hidden_units': 25, 'optimizer': 'Adam', 'lr': 0.0015, 'epochs': 40},
            31: {'hidden_units': 25, 'optimizer': 'RMSprop', 'lr': 0.0016, 'epochs': 10},
            42: {'hidden_units': 25, 'optimizer': 'RMSprop', 'lr': 0.001, 'epochs': 10}
        }
    }
    
    dataset_params = paper_params.get(dataset_key, {})
    return dataset_params.get(round_num, None)

# ====== TRAINING FUNCTION ======
def train_simon_rwtt_for_round(dataset_key, round_num, datasets, previous_ciphertexts=None):
    """Train Simon RWTT model for a specific round on a specific dataset"""
    print(f"\n=== Simon RWTT Training for Round {round_num} on {dataset_key} ===")
    
    # Prepare data
    X_train_ct, X_test_ct, y_train, y_test, next_round_ciphertexts = prepare_simon_rwtt_data(
        dataset_key, datasets, round_num, previous_ciphertexts
    )
    
    # Verify data shapes match
    print(f"X_train_ct shape: {X_train_ct.shape}, y_train shape: {y_train.shape}")
    print(f"X_test_ct shape: {X_test_ct.shape}, y_test shape: {y_test.shape}")
    
    if X_train_ct.shape[0] != y_train.shape[0]:
        raise ValueError(f"Training data mismatch: X has {X_train_ct.shape[0]} samples, y has {y_train.shape[0]}")
    
    if X_test_ct.shape[0] != y_test.shape[0]:
        raise ValueError(f"Test data mismatch: X has {X_test_ct.shape[0]} samples, y has {y_test.shape[0]}")
    
    # Use paper hyperparameters if available
    paper_params = get_paper_hyperparams_simon_rwtt(dataset_key, round_num)
    
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
                'hidden_units': trial.suggest_categorical('hidden_units', [10, 15, 20, 25]),
                'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'RMSprop']),
                'lr': trial.suggest_categorical('lr', [0.001, 0.002, 0.003]),
                'epochs': trial.suggest_int('epochs', 10, 60, step=10)
            }
            
            # Build hybrid model
            model = build_simon_rwtt_hybrid_model(
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
    model = build_simon_rwtt_hybrid_model(
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
        'Mean_byte_error': test_mbe,
        'Test_Hamming': test_hamming
    }
    
    print(f"{dataset_key} Round {round_num} Results:")
    print(f"  Test Accuracy: {result['Test_Accuracy']:.2f}%")
    print(f"  Test MBE: {result['Mean_byte_error']:.4f}")
   
    
    return model, result, next_round_ciphertexts

# ====== MAIN EXECUTION ======
def main():
    """Train Simon RWTT models for all datasets and all 42 rounds"""
    print("Initializing Simon RWTT Hybrid training pipeline...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    try:
        lib.train_simon_rwtt_for_rounds()
    except Exception as e:
        print(f" training error: {e}")
        # Load all datasets
        print("Loading datasets...")
        datasets = load_all_data()
        
        # Filter for RWTT datasets (dataset5, dataset6, dataset7)
        rwtt_datasets = {k: v for k, v in datasets.items() if k in ["dataset-5", "dataset-6", "dataset-7"]}
        
        if not rwtt_datasets:
            print("No RWTT datasets found. Please ensure datasets 5, 6, and 7 are available.")
            return
        
        # Train for all 42 rounds (1 to 42)
        all_rounds = list(range(1, 43))
        
        # Store results
        all_results = []
        
        # Train for each dataset and each round
        for dataset_key in rwtt_datasets.keys():
            previous_ciphertexts = None
            models_per_round = {}  # Store models for each round
            
            for round_num in all_rounds:
                try:
                    model, result, previous_ciphertexts = train_simon_rwtt_for_round(
                        dataset_key, round_num, rwtt_datasets, previous_ciphertexts
                    )
                    
                    if result:
                        all_results.append(result)
                        models_per_round[round_num] = model
                        
                except Exception as e:
                    print(f"Error training {dataset_key} round {round_num}: {e}")
                    continue
            
            # Save models for each round
            for round_num, model in models_per_round.items():
                model.save(os.path.join("models", f"simon_rwtt_hybrid_{dataset_key}_round_{round_num}.h5"))
                print(f"Saved model for {dataset_key} round {round_num}")
            
            # Also save the final model
            if models_per_round:
                final_round = max(models_per_round.keys())
                final_model = models_per_round[final_round]
                final_model.save(os.path.join("models", f"rwtt_{dataset_key}hybrid_final.h5"))
                print(f"Saved final hybrid model for {dataset_key} after all rounds")
        
        # Save results to CSV
        if all_results:
            fieldnames = [
                'Dataset', 'Round', 'Optimizer', 'Epochs', 'Lr_rate', 'Hidden_Nodes',
                'Test_Accuracy', 'Mean_byte_error', 'Test_Hamming'
            ]
            
            with open(os.path.join("results", "rwtt_hybrid_round_results.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
            
            print("\n=== Final Simon RWTT Hybrid Results (All Rounds) ===")
            for res in all_results:
                print(f"{res['Dataset']} Round {res['Round']}: {res['Test_Accuracy']:.2f}% (MBE: {res['Mean_byte_error']:.4f})")
        else:
            print("No results to save due to errors.")

if __name__ == "__main__":
    main()