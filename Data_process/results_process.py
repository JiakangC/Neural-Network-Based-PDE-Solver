import csv
import json
import os

# Load data from a JSON file (replace "your_data.json" with the actual path to your JSON file)

# ========= path list =========
# Results/Infinite_Potential_Well_1D/results_IPW_1D.json
# Results/Infinite_Potential_Well_2D/results_IPW_2D.json
# Results/Quantum_Harmonic_Oscillator_1D/results_QHO_1D.json
# Results/Quantum_Harmonic_Oscillator_2D/results_QHO_2D.json
save_path = 'Results/Infinite_Potential_Well_1D'
results_file = os.path.join(save_path, 'results_IPW_1D.json')

# ========== load data ==========
with open(results_file, "r") as f:
    data = json.load(f)



# filter for the results with 'layers' key this one is for ablation study on layers
data = [entry for entry in data if 'layers' in entry]

# # Group data by method and find the entry with the lowest L2_error for each method
# best_entries = {}
# for entry in data:
#     method = entry["method"]
#     if method not in best_entries or entry["L2_error"] < best_entries[method]["L2_error"]:
#         best_entries[method] = entry

# Define the fields for the CSV, including weight terms, percentage, and technique
# fields = [
#     "method", "L2_error", "time", "time_of_best_model", "n", "technique",
#     "weight_pde", "weight_drm", "weight_norm", "weight_bc", "weight_orth", "weight_data", "epochs", "percentage", "timestamp"
# ]
fields = [
    "method", "L2_error", "time", "time_of_best_model", "n", "technique", "layers"
]
# Write to CSV file
csv_file = "method_comparison_IPW_1D_FN_layers.csv"
output_file = os.path.join(save_path, csv_file)
with open(output_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    # for entry in best_entries.values():
    for entry in data:
        writer.writerow({
            "method": entry["method"],
            "L2_error": entry["L2_error"],
            "time": entry["time"],
            "time_of_best_model": entry["time_of_best_model"],
            "n": entry["n"],
            "technique": entry["technique"],
            "layers": entry.get("layers", ""),
        })

print(f"CSV file '{output_file}' has been created successfully.")