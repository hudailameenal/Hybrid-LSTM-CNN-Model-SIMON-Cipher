import os
import csv
import numpy as np
from tensorflow.keras.models import load_model

def display_saved_results():
    """Display the saved results from the CSV file in a table"""
    results_file = os.path.join("results", "ctt_hybrid_round_results.csv")
    
    # Check if results file exists
    if not os.path.exists(results_file):
        print(f"Results file not found at {results_file}")
        return
    
    print("\n=== Saved Model Results (SIMON CTT) ===\n")
    
    # Load CSV rows
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("No results found in CSV.")
        return
    
    # Define headers
    headers = [
        "Round", "Optimizer", "Epochs", "Lr_rate", "Hidden_Nodes",
        "Test_Accuracy", "Mean_Byte_Error"
    ]
    
    # Prepare data
    table_data = []
    for row in rows:
        table_data.append([
            row["Round"],
            row["Optimizer"],
            row["Epochs"],
            row["Lr_rate"],
            row["Hidden_Nodes"],
            f"{row['Test_Accuracy']}%",
            row["Mean_Byte_Error"],
        ])
    
    # Find column widths
    col_widths = [max(len(str(item)) for item in [header] + [r[i] for r in table_data]) for i, header in enumerate(headers)]
    
    # Function to format rows
    def format_row(row_items):
        return "│ " + " │ ".join(f"{item:<{col_widths[i]}}" for i, item in enumerate(row_items)) + " │"
    
    # Print top border
    total_width = sum(col_widths) + (3 * len(col_widths)) + 1
    print("┌" + "─" * (total_width - 2) + "┐")
    
    # Print header
    print(format_row(headers))
    
    # Print separator
    print("├" + "─" * (total_width - 2) + "┤")
    
    # Print rows
    for row in table_data:
        print(format_row(row))
    
    # Print bottom border
    print("└" + "─" * (total_width - 2) + "┘\n")


def display_model_info():
    """Display information about the saved SIMON model"""
    model_path = os.path.join("models", "simon_ctt_hybrid.h5")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
    
    print("\n=== SIMON CTT Model Information ===\n")
    
    try:
        model = load_model(model_path)
        print("✓ Model loaded successfully\n")
        
        # Display model configuration in a box
        info = [
            ["Optimizer", model.optimizer.__class__.__name__],
            ["Loss Function", model.loss],
            ["Metrics", ", ".join(model.metrics_names) if hasattr(model, "metrics_names") else "None"]
        ]
        
        # Find widths
        col_width = max(len(i[0]) for i in info)
        val_width = max(len(str(i[1])) for i in info)
        
        total_width = col_width + val_width + 7
        print("┌" + "─" * (total_width - 2) + "┐")
        for key, value in info:
            print(f"│ {key:<{col_width}} │ {value:<{val_width}} │")
        print("└" + "─" * (total_width - 2) + "┘\n")
        
    except Exception as e:
        print(f"Error loading model: {e}")


if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Show results & model info
    display_saved_results()
   