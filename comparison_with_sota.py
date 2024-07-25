import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the results for each model (example values)
dra_results = {
    # Det
    "HIS_det": [72.91, 68.73, 74.33, 79.16],
    "chest_Xray_det": [72.22, 75.81, 82.70, 85.01],
    "OCT_17_det": [98.08, 99.06, 99.13, 99.87],
    # Det & Seg
    "brain_MRI_det": [71.78, 80.62, 85.94, 82.99],
    "brain_MRI_seg": [72.09, 74.77, 75.32, 80.45],
    "liver_CT_det": [57.17, 59.64, 72.53, 80.89],
    "liver_CT_seg": [63.13, 71.79, 81.78, 93.00],
    "RESC_det": [85.69, 90.90, 93.06, 94.88],
    "RESC_seg": [65.59, 77.28, 83.07, 80.01],
}

bgad_results = {
    # Det
    "HIS_det": [
        None, 
        None,
        None,
        None,
    ],
    "chest_Xray_det": [
        None,
        None,
        None,
        None,
    ],
    "OCT_17_det": [
        None,
        None,
        None,
        None,
    ],
    # Det & Seg
    "brain_MRI_det": [78.70, 83.56, 88.01, 88.05],
    "brain_MRI_seg": [92.42, 92.68, 94.32, 95.29],
    "liver_CT_det": [72.27, 72.48, 74.60, 78.79],
    "liver_CT_seg": [98.71, 98.88, 99.00, 99.25],
    "RESC_det": [83.58, 86.22, 89.96, 91.29],
    "RESC_seg": [92.10, 93.84, 96.06, 97.07],
}

april_gan_results = {
    # Det
    "HIS_det": [69.57, 76.11, 81.70, 81.16],
    "chest_Xray_det": [69.84, 77.43, 73.69, 78.62],
    "OCT_17_det": [99.21, 99.41, 99.75, 99.93],
    # Det & Seg
    "brain_MRI_det": [78.45, 89.18, 88.41, 94.03],
    "brain_MRI_seg": [94.02, 94.67, 95.50, 96.17],
    "liver_CT_det": [57.80, 53.05, 62.38, 82.94],
    "liver_CT_seg": [95.87, 96.24, 97.56, 99.64],
    "RESC_det": [89.44, 94.70, 91.36, 95.96],
    "RESC_seg": [96.39, 97.98, 97.36, 98.47],
}

mvfa_ad_results = {
    # Det
    "HIS_det": [82.61, 82.71, 85.10, 82.62],
    "chest_Xray_det": [81.32, 81.95, 883.89, 85.72],
    "OCT_17_det": [97.98, 99.38, 99.64, 99.66],
    # Det & Seg
    "brain_MRI_det": [92.72, 92.44, 92.61, 94.40],
    "brain_MRI_seg": [96.55, 97.30, 97.21, 97.70],
    "liver_CT_det": [81.08, 81.18, 85.90, 83.85],
    "liver_CT_seg": [96.57, 99.73, 99.79, 99.73],
    "RESC_det": [91.36, 96.18, 96.57, 97.25],
    "RESC_seg": [98.11, 98.97, 99.00, 99.07],
}

my_model_results = {
    # Det
    "HIS_det": [71.45, 83.05, 87.38, 85.56],
    "chest_Xray_det": [86.07, 82.13, 84.12, 87.39],
    "OCT_17_det": [98.16, 99.66, 99.15, 99.93],
    # Det & Seg
    "brain_MRI_det": [92.73, 91.34, 91.9, 94.51],
    "brain_MRI_seg": [96.98, 97.10, 96.95, 97.91],
    "liver_CT_det": [81.81, 84.50, 89.39, 92.75],
    "liver_CT_seg": [98.02, 99.63, 99.65, 99.64],
    "RESC_det": [92.17, 93.47, 97.14, 97.37],
    "RESC_seg": [97.63, 99.02, 98.92, 99.34],
}

# Define shots
shots = [2, 4, 8, 16]


# Prepare DataFrame for results
def create_results_df(model_name, results_dict):
    rows = []
    for task, auc_values in results_dict.items():
        dataset, task_type = task.rsplit("_", 1)
        for shot, auc in zip(shots, auc_values):
            if auc is not None:
                rows.append([model_name, dataset, task_type, shot, auc])
    return pd.DataFrame(rows, columns=["Model", "Dataset", "Task", "Shots", "AUC"])


# Combine all results into a single DataFrame
dfs = [
    create_results_df("DRA", dra_results),
    create_results_df("BGAD", bgad_results),
    create_results_df("April-GAN", april_gan_results),
    create_results_df("MVFA-AD", mvfa_ad_results),
    create_results_df("Ours", my_model_results),
]
results_df = pd.concat(dfs, ignore_index=True)

# Create directory to save images
output_dir = "results_plots/paper"
os.makedirs(output_dir, exist_ok=True)

# Define markers and colors for each model
model_styles = {
    "DRA": {
        "color": "blue",
        "marker_det": "^",
        "marker_seg": "*",
        "line_style_det": "solid",
        "line_style_seg": "dashed",
    },
    "BGAD": {
        "color": "green",
        "marker_det": "^",
        "marker_seg": "*",
        "line_style_det": "solid",
        "line_style_seg": "dashed",
    },
    "April-GAN": {
        "color": "red",
        "marker_det": "^",
        "marker_seg": "*",
        "line_style_det": "solid",
        "line_style_seg": "dashed",
    },
    "MVFA-AD": {
        "color": "purple",
        "marker_det": "^",
        "marker_seg": "*",
        "line_style_det": "solid",
        "line_style_seg": "dashed",
    },
    "Ours": {
        "color": "orange",
        "marker_det": "^",
        "marker_seg": "*",
        "line_style_det": "solid",
        "line_style_seg": "dashed",
    },
}


# Plot results and save images
def plot_results(results_df, dataset):
    plt.figure(figsize=(6, 6))
    x_positions = np.arange(len(shots))
    task_name = {"det": "AD", "seg": "AS"}
    for task in ["det", "seg"]:
        subset = results_df[
            (results_df["Dataset"] == dataset) & (results_df["Task"] == task)
        ]
        for model in subset["Model"].unique():
            model_data = subset[subset["Model"] == model]
            color = model_styles[model]["color"]
            marker = model_styles[model][f"marker_{task.lower()}"]
            line_style = model_styles[model][f"line_style_{task.lower()}"]
            plt.plot(
                x_positions,
                model_data["AUC"],
                linestyle=line_style,
                marker=marker,
                color=color,
                label=f"{model} ({task_name[task]})",
            )
    # plt.title(f"{dataset}")
    plt.xlabel("Shots")
    plt.ylabel("AUC (%)")
    plt.ylim(50, 100)
    plt.xticks(x_positions, ["2", "4", "8", "16"])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset}_results.png"))
    plt.close()


# Plot results for relevant datasets
datasets = ["brain_MRI", "liver_CT", "RESC"]
for dataset in datasets:
    plot_results(results_df, dataset)

# Plot individual classification results
datasets_det_only = ["OCT_17", "chest_Xray", "HIS"]
for dataset in datasets_det_only:
    plot_results(results_df, dataset)
