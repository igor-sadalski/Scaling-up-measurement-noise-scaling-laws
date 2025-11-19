import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Model
import warnings
import os


def info_scaling(u, u_bar, I_max):
    """
    Information scaling function based on:
        I(u) = I_max - 0.5 * log2( (1 + u/u_bar) / (u/u_bar + 2**(-2*I_max)) )
    """
    # Avoid division by zero and invalid values
    u = np.asarray(u)
    u_bar = np.asarray(u_bar)
    I_max = np.asarray(I_max)
    # Set a small epsilon to avoid division by zero
    epsilon = 1e-12
    u_bar_safe = np.where(u_bar == 0, epsilon, u_bar)
    u_over_u_bar = u / u_bar_safe
    numerator = 1 + u_over_u_bar
    denominator = u_over_u_bar + 2 ** (-2 * I_max)
    # Avoid division by zero in denominator
    denominator = np.where(denominator == 0, epsilon, denominator)
    # Avoid negative or zero values inside log2
    ratio = numerator / denominator
    ratio = np.where(ratio <= 0, epsilon, ratio)
    return I_max - 0.5 * np.log2(ratio)


def plot_noise_scaling_results(csv_path, output_dir=None, show_plot=True):
    """
    Load noise scaling CSV and plot fitted curves with uncertainty bands for each row.
    
    Parameters:
        csv_path: path to noise_scaling.csv
        output_dir: directory to save plots (default: current directory)
        show_plot: whether to display the plot
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Filter out rows with invalid data
    df = df.dropna(subset=['fitted_u_bar', 'fitted_I_max'])
    
    # Get unique combinations of method, metric, and dataset for plotting
    unique_combinations = df.groupby(["method", "metric", "dataset"]).size().reset_index(name="count")
    
    # Get unique metrics and methods for subplot layout
    metrics = df["metric"].unique()
    methods = df["method"].unique()
    
    fig, axes = plt.subplots(len(metrics), len(methods), figsize=(4 * len(methods), 4 * len(metrics)))
    
    if len(metrics) == 1 and len(methods) == 1:
        axes = np.array([[axes]])
    elif len(metrics) == 1:
        axes = axes.reshape(1, -1)
    elif len(methods) == 1:
        axes = axes.reshape(-1, 1)
    
    # Get unique sizes for color mapping
    sizes = sorted(df["size"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(sizes)))
    size_color_map = dict(zip(sizes, colors))
    
    # Dataset styles
    datasets = df["dataset"].unique()
    dataset_styles = ["-", "--", "-.", ":"]
    dataset_markers = ["o", "s", "^", "v", "D", "p", "*", "h"]
    
    for i, metric in enumerate(metrics):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            subset = df[(df["metric"] == metric) & (df["method"] == method)]
            
            if len(subset) == 0:
                ax.set_title(f"{method}\n{metric}\n(No data)")
                continue
            
            legend_labels = []
            legend_handles = []
            
            # Get unique datasets for this method-metric combination
            subset_datasets = subset["dataset"].unique()
            
            for dataset_idx, dataset in enumerate(subset_datasets):
                dataset_data = subset[subset["dataset"] == dataset]
                
                for size in sizes:
                    size_data = dataset_data[dataset_data["size"] == size]
                    
                    if len(size_data) == 0:
                        continue
                    
                    # Get fitted parameters from the first row (should be same for same size)
                    row = size_data.iloc[0]
                    u_bar = row["fitted_u_bar"]
                    I_max = row["fitted_I_max"]
                    u_bar_err = row["u_bar_error"]
                    I_max_err = row["I_max_error"]
                    avg_uncertainty_err = row["avg_uncertainty_error"]
                    
                    # Skip if parameters are invalid
                    if pd.isna(u_bar) or pd.isna(I_max) or u_bar <= 0 or I_max <= 0:
                        continue
                    
                    # Generate a reasonable range of u values
                    # Use a range around u_bar for better visualization
                    u_min = max(1, u_bar * 0.01)
                    u_max = u_bar * 100
                    u_range = np.logspace(np.log10(u_min), np.log10(u_max), 100)
                    
                    # Calculate the fitted curve
                    fitted_curve = info_scaling(u_range, u_bar, I_max)
                    
                    # Calculate uncertainty bands using lmfit
                    # Create a model and result object to compute uncertainty
                    linestyle = dataset_styles[dataset_idx % len(dataset_styles)]
                    color = size_color_map[size]
                    
                    try:
                        model = Model(info_scaling)
                        params = model.make_params(u_bar=u_bar, I_max=I_max)
                        params['u_bar'].value = u_bar
                        params['I_max'].value = I_max
                        
                        # Set parameter uncertainties if available
                        if not pd.isna(u_bar_err):
                            params['u_bar'].stderr = u_bar_err
                        if not pd.isna(I_max_err):
                            params['I_max'].stderr = I_max_err
                        
                        # Compute uncertainty using eval_uncertainty
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            try:
                                y_uncertainty = model.eval_uncertainty(
                                    params=params, 
                                    u=u_range, 
                                    sigma=2
                                )
                                max_uncertainty = np.max(np.abs(y_uncertainty))
                                
                                # Only plot uncertainty bands if error is reasonable
                                if max_uncertainty <= 0.5:
                                    y_upper = y_uncertainty
                                    y_lower = -y_uncertainty
                                    
                                    # Plot uncertainty bands
                                    ax.fill_between(
                                        u_range,
                                        fitted_curve + y_upper,
                                        fitted_curve + y_lower,
                                        color=color,
                                        alpha=0.1,
                                    )
                            except Exception:
                                # Fallback to avg_uncertainty_error if eval_uncertainty fails
                                if not pd.isna(avg_uncertainty_err) and avg_uncertainty_err <= 0.5:
                                    y_upper = np.full_like(u_range, avg_uncertainty_err)
                                    y_lower = -y_upper
                                    ax.fill_between(
                                        u_range,
                                        fitted_curve + y_upper,
                                        fitted_curve + y_lower,
                                        color=color,
                                        alpha=0.1,
                                    )
                    except Exception:
                        # Fallback if model creation fails - use avg_uncertainty_error
                        if not pd.isna(avg_uncertainty_err) and avg_uncertainty_err <= 0.5:
                            y_upper = np.full_like(u_range, avg_uncertainty_err)
                            y_lower = -y_upper
                            ax.fill_between(
                                u_range,
                                fitted_curve + y_upper,
                                fitted_curve + y_lower,
                                color=color,
                                alpha=0.1,
                            )
                    
                    # Plot the fitted curve
                    label = f"{dataset} Size {size}"
                    ax.plot(
                        u_range,
                        fitted_curve,
                        color=color,
                        linestyle=linestyle,
                        lw=1.5,
                        alpha=0.7,
                        label=label,
                    )
                    
                    # Add a marker at a representative point (e.g., at u_bar)
                    marker = dataset_markers[dataset_idx % len(dataset_markers)]
                    mi_at_u_bar = info_scaling(u_bar, u_bar, I_max)
                    handle = ax.scatter(
                        u_bar,
                        mi_at_u_bar,
                        color=color,
                        marker=marker,
                        s=50,
                        alpha=0.8,
                        zorder=5,
                    )
                    legend_handles.append(handle)
                    legend_labels.append(label)
            
            ax.set_xlabel("UMIs per cell")
            ax.set_ylabel("MI value")
            ax.set_title(f"{method}\n{metric}")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            
            # Add legend only for first subplot
            if i == 0 and j == 0 and legend_handles:
                ax.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    fontsize=8,
                )
    
    plt.tight_layout()
    
    # Save plot if output directory is specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "noise_scaling_visualization.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Path to the CSV file
    csv_path = "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/final_results/noise_scaling.csv"
    
    # Output directory for saving plots
    output_dir = "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/final_results"
    
    # Plot the results
    plot_noise_scaling_results(csv_path, output_dir=output_dir, show_plot=True)

