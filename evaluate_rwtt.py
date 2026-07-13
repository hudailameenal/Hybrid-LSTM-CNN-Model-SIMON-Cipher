import os
import csv
from tensorflow.keras.models import load_model

def display_saved_results():
    """Display saved results from the RWTT CSV file in a table format"""
    results_file = os.path.join("results", "rwtt_hybrid_round_results.csv")
    
    if not os.path.exists(results_file):
        print(f"[ERROR] Results file not found at {results_file}")
        return
    
    print("\n=== Saved Model Results (SIMON RWTT) ===\n")

    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if not fieldnames:
            print("[ERROR] Results file has no headers.")
            return

        rows = list(reader)
        if not rows:
            print("[WARNING] Results file has headers but no data rows.")
            return

    # Group rows by dataset
    datasets = {}
    for row in rows:
        dataset = row["Dataset"]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(row)

    # Print table for each dataset
    for ds, ds_rows in datasets.items():
        print(f"--- Results for {ds} ---")

        headers = ["Round", "Optimizer", "Epochs", "Lr_rate", "Hidden_Nodes",
                   "Bitwise_test_acc", "Mean_byte_error"]

        table_data = []
        for row in ds_rows:
            table_data.append([
                row["Round"],
                row["Optimizer"],
                row["Epochs"],
                row["Lr_rate"],
                row["Hidden_Nodes"],
                f"{float(row['Bitwise_test_acc']):.2f}%",
                f"{float(row['Mean_byte_error']):.4f}"
            ])

        # find column widths
        col_widths = [
            max(len(str(item)) for item in [header] + [r[i] for r in table_data])
            for i, header in enumerate(headers)
        ]

        def format_row(row_items):
            return "│ " + " │ ".join(f"{item:<{col_widths[i]}}" for i, item in enumerate(row_items)) + " │"

        # print top border
        total_width = sum(col_widths) + (3 * len(col_widths)) + 1
        print("┌" + "─" * (total_width - 2) + "┐")
        print(format_row(headers))
        print("├" + "─" * (total_width - 2) + "┤")

        for row in table_data:
            print(format_row(row))

        print("└" + "─" * (total_width - 2) + "┘\n")


def display_model_info():
    """Display information about saved RWTT models"""
    model_dir = "models"
    datasets = ["dataset-5", "dataset-6", "dataset-7"]  # match your CSV
    
    print("\n=== SIMON RWTT Model Information ===\n")
    
    for ds in datasets:
        model_path = os.path.join(model_dir, f"rwtt_{ds}_final.h5")
        
        if not os.path.exists(model_path):
            print(f"Model for {ds} not found at {model_path}")
            continue
        
        try:
            model = load_model(model_path)
            print(f"--- {ds} Model ---")
            print("✓ Model loaded successfully\n")
            
            info = [
                ["Optimizer", model.optimizer.__class__.__name__],
                ["Loss Function", model.loss],
                ["Metrics", ", ".join(model.metrics_names) if hasattr(model, "metrics_names") else "None"]
            ]
            
            col_width = max(len(i[0]) for i in info)
            val_width = max(len(str(i[1])) for i in info)
            total_width = col_width + val_width + 7
            
            print("┌" + "─" * (total_width - 2) + "┐")
            for key, value in info:
                print(f"│ {key:<{col_width}} │ {value:<{val_width}} │")
            print("└" + "─" * (total_width - 2) + "┘\n")
        
        except Exception as e:
            print(f"Error loading {ds} model: {e}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    display_saved_results()
    