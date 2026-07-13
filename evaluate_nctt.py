import os
import csv
from tensorflow.keras.models import load_model

def display_saved_results():
    """Display saved results from the NCTT CSV file in a table format"""
    results_file = os.path.join("results", "nctt_hybrid_round_results.csv")
    
    if not os.path.exists(results_file):
        print(f"Results file not found at {results_file}")
        return
    
    print("\n=== Saved Model Results (SIMON NCTT) ===\n")

    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        # detect datasets from column headers
        datasets = []
        for name in fieldnames:
            if name.startswith("Test_Accuracy_"):
                ds_name = name.replace("Test_Accuracy_", "")
                datasets.append(ds_name)
        
        rows = list(reader)
    
    if not rows:
        print("No results found in CSV.")
        return

    for ds in datasets:
        print(f"--- Results for {ds} ---")
        
        headers = ["Round", "Optimizer", "Epochs", "Lr_rate", "Hidden_Nodes",
                   "Test_Accuracy", "Mean_Byte_Error"]
        
        # build dataset-specific table
        table_data = []
        for row in rows:
            table_data.append([
                row["Round"],
                row["Optimizer"],
                row["Epochs"],
                row["Lr_rate"],
                row["Hidden_Nodes"],
                f"{float(row[f'Test_Accuracy_{ds}']):.2f}%",
                f"{float(row[f'Mean_Byte_Error_{ds}']):.4f}"
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
    """Display information about saved NCTT models"""
    model_dir = "models"
    datasets = ["dataset-2", "dataset-3", "dataset-4"]
    
    print("\n=== SIMON NCTT Model Information ===\n")
    
    for ds in datasets:
        model_path = os.path.join(model_dir, f"nctt_{ds}_final.h5")
        
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
   