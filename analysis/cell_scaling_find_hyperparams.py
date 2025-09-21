import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from lmfit import Model, Parameters


def cell_number_scaling(x, N0, s, I_inf):
    """
    Cell number scaling function: I(x) = I_inf - (x / N0) ** (-s)
    """
    return I_inf - (x / N0) ** (-s)


def fit_cell_number_scaling_model(cell_values, mi_values, method, initial_N0, initial_s, initial_I_inf):
    """
    Fit the cell number scaling model to data and return parameters and mean residual.

    Parameters:
    cell_values: array of cell number values
    mi_values: array of mutual information values
    method: string indicating the method for parameter initialization
    initial_N0: initial guess for N0 parameter
    initial_s: initial guess for s parameter
    initial_I_inf: initial guess for I_inf parameter

    Returns:
    dict with N0, s, I_inf, mean_residual, fit_success status, and lmfit result
    """

    # Define the cell number scaling function for fitting
    def cell_number_scaling_local(x, N0, s, I_inf):
        """
        Cell number scaling function: I(x) = I_inf - (x / N0) ** (-s)
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(x > 0, I_inf - (x / N0) ** (-s), np.nan)
        return result

    # Create lmfit model
    model = Model(cell_number_scaling_local)

    # Set up parameters with initial values and bounds
    if method in ["PCA", "RandomProjection"]:
        params = model.make_params(
            N0=dict(value=initial_N0, min=1e-6),
            s=dict(value=initial_s, min=1e-3),
            I_inf=dict(value=initial_I_inf, min=mi_values.max(), max=mi_values.max() * 1.5),
        )
    else:  # SCVI or Geneformer
        params = model.make_params(N0=initial_N0, s=initial_s, I_inf=initial_I_inf)
        params["N0"].min = 10
        params["N0"].max = 10**6
        params["s"].min = 0.01
        params["s"].max = 2.0
        params["I_inf"].min = 0.1
        params["I_inf"].max = 5.0

    # Fit the curve
    try:
        result = model.fit(mi_values, params, x=cell_values)
        N0_val = result.params["N0"].value
        s_val = result.params["s"].value
        I_inf_val = result.params["I_inf"].value

        # Compute mean residual
        residuals = result.residual
        mean_residual = np.mean(np.abs(residuals))

        # Compute uncertainty bands (optional, not used in output)
        x_fit = cell_values
        y_cell_upper = result.eval_uncertainty(params=result.params, x=x_fit, sigma=2)
        y_cell_lower = -y_cell_upper

        return {
            "N0": N0_val,
            "s": s_val,
            "I_inf": I_inf_val,
            "mean_residual": mean_residual,
            "fit_success": True,
            "result": result,
            "y_cell_upper": y_cell_upper,
            "y_cell_lower": y_cell_lower,
        }

    except Exception as e:
        print(f"Fitting algorithm failed for method {method}: {str(e)}")
        return {
            "N0": np.nan,
            "s": np.nan,
            "I_inf": np.nan,
            "mean_residual": np.nan,
            "fit_success": False,
            "result": None,
            "y_cell_upper": np.nan,
            "y_cell_lower": np.nan,
        }


def plot_cell_scaling_fits(
    df, initial_N0=10**4, initial_s=1.0, initial_I_inf=2.5, save_plots=True, output_dir=None, show_plot=True
):
    """
    Fit and plot cell scaling model for all (dataset, method, metric, quality) groups in df.

    Parameters:
        df: DataFrame with columns including 'dataset', 'method', 'metric', 'quality', 'size', 'mi_value'
        initial_N0: initial guess for N0 parameter
        initial_s: initial guess for s parameter
        initial_I_inf: initial guess for I_inf parameter
        save_plots: whether to save plots to file
        output_dir: directory to save plots (default: cell_scaling_different_hyperparams)
        show_plot: whether to display the plot

    Returns:
        DataFrame with fit results including dataset, method, metric, quality, size, N0, s, I_inf, mean_residual
    """

    # Create output directory if not provided
    if output_dir is None:
        output_dir = "/home/jupyter/igor_repos/exploration/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/cell_scaling_different_hyperparams"

    # Clear the output directory if it exists
    if os.path.exists(output_dir):
        import shutil

        shutil.rmtree(output_dir)
        print(f"Cleared existing directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Get unique combinations of dataset, method, metric, and quality
    unique_combinations = df.groupby(["dataset", "method", "metric", "quality"]).size().reset_index(name="count")

    # Create a separate dataframe to store all cell scaling results
    cell_scaling_results_df = []

    # Loop through each combination
    failed_fits = []
    for idx, row in unique_combinations.iterrows():
        dataset = row["dataset"]
        method = row["method"]
        metric = row["metric"]
        quality = row["quality"]

        # Filter data for current combination
        filtered_data = df[
            (df["dataset"] == dataset)
            & (df["method"] == method)
            & (df["metric"] == metric)
            & (df["quality"] == quality)
        ]

        # Skip if not enough data points
        if len(filtered_data) < 3:
            continue

        # Extract data for fitting
        cell_values = filtered_data["size"].values
        mi_values = filtered_data["mi_value"].values

        # Fit the model
        fit_results = fit_cell_number_scaling_model(
            cell_values, mi_values, method, initial_N0, initial_s, initial_I_inf
        )

        # Store results in separate dataframe - create one row per combination
        result_row = {
            "dataset": dataset,
            "method": method,
            "metric": metric,
            "quality": quality,
            "N0": fit_results["N0"],
            "s": fit_results["s"],
            "I_inf": fit_results["I_inf"],
            "mean_residual": fit_results["mean_residual"],
        }
        cell_scaling_results_df.append(result_row)

        if not fit_results["fit_success"]:
            # Record failed fit
            failed_fits.append(f"{dataset}/{method}/{metric}/quality={quality:.3f}")
            print(f"Fitting failed for: {dataset}, {method}, {metric}, quality={quality:.3f}")

    # Convert the cell scaling results to a DataFrame
    cell_scaling_results_summary = pd.DataFrame(cell_scaling_results_df)

    # Print summary of failed fits
    if failed_fits:
        print(f"\nTotal failed fits: {len(failed_fits)}")
        print("Failed combinations:")
        for failed in failed_fits:
            print(f"  - {failed}")
    else:
        print("All fits successful!")

    # Create a subset for visualization (only successful fits - those with valid N0 values)
    cell_scaling_results = cell_scaling_results_summary[cell_scaling_results_summary["N0"].notna()].copy()

    # Create the plot
    if len(cell_scaling_results) > 0 and save_plots:
        # Get unique combinations of metric and method
        metrics = cell_scaling_results["metric"].unique()
        methods = cell_scaling_results["method"].unique()

        # Create subplots with methods as columns and metrics as rows
        fig, axes = plt.subplots(len(metrics), len(methods), figsize=(4 * len(methods), 4 * len(metrics)))

        # Handle case where there's only one metric or method
        if len(metrics) == 1 and len(methods) == 1:
            axes = np.array([[axes]])
        elif len(metrics) == 1:
            axes = axes.reshape(1, -1)
        elif len(methods) == 1:
            axes = axes.reshape(-1, 1)

        # Color scheme for different qualities
        qualities = sorted(cell_scaling_results["quality"].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(qualities)))
        quality_color_map = dict(zip(qualities, colors))

        for i, metric in enumerate(metrics):
            for j, method in enumerate(methods):
                ax = axes[i, j]

                # Filter data for this metric and method combination
                subset = cell_scaling_results[
                    (cell_scaling_results["metric"] == metric) & (cell_scaling_results["method"] == method)
                ]

                if len(subset) > 0:
                    # Plot points for each quality
                    for quality in qualities:
                        quality_data = subset[subset["quality"] == quality]
                        if len(quality_data) > 0:
                            # Get the first row for this quality (they should all have same fitted parameters)
                            first_row = quality_data.iloc[0]

                            # Get the original data for plotting
                            original_data = df[
                                (df["dataset"] == first_row["dataset"])
                                & (df["method"] == first_row["method"])
                                & (df["metric"] == first_row["metric"])
                                & (df["quality"] == first_row["quality"])
                            ]

                            # Group by size (cell number) to handle multiple seeds
                            grouped = original_data.groupby("size")

                            x_vals = []
                            y_means = []
                            y_stds = []

                            for size, group in grouped:
                                x_vals.append(size)
                                mi_values = group["mi_value"]
                                y_means.append(mi_values.mean())
                                y_stds.append(mi_values.std() if len(mi_values) > 1 else 0)

                            x_vals = np.array(x_vals)
                            y_means = np.array(y_means)
                            y_stds = np.array(y_stds)

                            # Plot error bars if there are multiple seeds, otherwise scatter points
                            if np.any(y_stds > 0):
                                ax.errorbar(
                                    x_vals,
                                    y_means,
                                    yerr=y_stds,
                                    color=quality_color_map[quality],
                                    fmt="o",
                                    capsize=5,
                                    capthick=2,
                                    alpha=0.7,
                                    markersize=6,
                                    label=f"Quality {quality:.3f}",
                                )
                            else:
                                ax.scatter(
                                    x_vals,
                                    y_means,
                                    color=quality_color_map[quality],
                                    alpha=0.7,
                                    s=50,
                                    label=f"Quality {quality:.3f}",
                                )

                            # Plot fitted line for this quality
                            cell_range = np.linspace(original_data["size"].min(), original_data["size"].max(), 5000)
                            N0_fit = first_row["N0"]
                            s_fit = first_row["s"]
                            I_inf_fit = first_row["I_inf"]

                            # Calculate fitted curve
                            fitted_curve = cell_number_scaling(cell_range, N0_fit, s_fit, I_inf_fit)

                            # Plot fitted line
                            ax.plot(
                                cell_range,
                                fitted_curve,
                                color=quality_color_map[quality],
                                linestyle="--",
                                lw=2,
                                alpha=0.8,
                            )

                ax.set_xlabel("Number of cells")
                ax.set_ylabel("MI value")
                ax.set_title(f"{method}\n{metric}")
                ax.set_xscale("log")
                ax.grid(True, alpha=0.3)

                # Add legend only to the first subplot
                if i == 0 and j == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            else:
                ax.set_title(f"{method}\n{metric}\n(No data)")

        plt.tight_layout()

        # Save plot if requested
        if save_plots:
            # Create filename with hyperparameters (3 decimal places)
            filename = f"cell_scaling_N0_{initial_N0:.3f}_s_{initial_s:.3f}_I_inf_{initial_I_inf:.3f}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {filepath}")

        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the figure to free memory when not showing

    # Save the results to CSV
    if save_plots:
        csv_filename = f"cell_scaling_N0_{initial_N0:.3f}_s_{initial_s:.3f}_I_inf_{initial_I_inf:.3f}.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)
        cell_scaling_results_summary.to_csv(csv_filepath, index=False)
        print(f"Results saved to: {csv_filepath}")

    return cell_scaling_results_summary


def generate_hyperparameter_combinations(
    n_combinations=50, N0_range=(1000, 50000), s_range=(0.1, 2.0), I_inf_range=(0.5, 5.0)
):
    """
    Generate random hyperparameter combinations for testing.

    Parameters:
        n_combinations: number of combinations to generate
        N0_range: tuple of (min, max) for N0 values
        s_range: tuple of (min, max) for s values
        I_inf_range: tuple of (min, max) for I_inf values

    Returns:
        list of tuples (N0, s, I_inf)
    """
    import random

    combinations = []
    for _ in range(n_combinations):
        N0 = random.uniform(N0_range[0], N0_range[1])
        s = random.uniform(s_range[0], s_range[1])
        I_inf = random.uniform(I_inf_range[0], I_inf_range[1])
        combinations.append((N0, s, I_inf))
    return combinations


def plot_single_hyperparameter(args):
    """Plot function for multiprocessing."""
    df, N0, s, I_inf, output_dir = args

    try:
        # Call the plotting function with show_plot=False for multiprocessing
        results_df = plot_cell_scaling_fits(
            df,
            initial_N0=N0,
            initial_s=s,
            initial_I_inf=I_inf,
            save_plots=True,
            output_dir=output_dir,
            show_plot=False,
        )

        png_filename = f"cell_scaling_N0_{N0:.3f}_s_{s:.3f}_I_inf_{I_inf:.3f}.png"
        png_filepath = os.path.join(output_dir, png_filename)

        csv_filename = f"cell_scaling_N0_{N0:.3f}_s_{s:.3f}_I_inf_{I_inf:.3f}.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)

        return (N0, s, I_inf, True, png_filepath, csv_filepath)
    except Exception as e:
        print(f"Error with N0={N0:.3f}, s={s:.3f}, I_inf={I_inf:.3f}: {str(e)}")
        return (N0, s, I_inf, False, None, None)


def test_multiple_cell_scaling_hyperparameters(df, hyperparameter_combinations, n_processes=None):
    """
    Test multiple hyperparameter combinations for cell scaling and save plots for each.

    Parameters:
        df: DataFrame with the data
        hyperparameter_combinations: list of tuples (N0, s, I_inf)
        n_processes: number of processes to use (default: number of CPU cores)
    """
    import multiprocessing as mp
    from multiprocessing import Pool
    from tqdm import tqdm

    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(hyperparameter_combinations))

    output_dir = "/home/jupyter/igor_repos/exploration/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/cell_scaling_different_hyperparams"

    # Clear the output directory if it exists
    if os.path.exists(output_dir):
        import shutil

        shutil.rmtree(output_dir)
        print(f"Cleared existing directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Prepare arguments for multiprocessing
    args_list = [(df, N0, s, I_inf, output_dir) for N0, s, I_inf in hyperparameter_combinations]

    print(f"Generating {len(hyperparameter_combinations)} plots using {n_processes} processes...")

    # Use multiprocessing with progress bar
    with Pool(processes=n_processes) as pool:
        results = list(
            tqdm(pool.imap(plot_single_hyperparameter, args_list), total=len(args_list), desc="Generating plots")
        )

    # Print summary
    successful = sum(1 for _, _, _, success, _, _ in results if success)
    failed = len(results) - successful

    print(f"\nPlot generation complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("Failed combinations:")
        for N0, s, I_inf, success, _, _ in results:
            if not success:
                print(f"  N0={N0:.3f}, s={s:.3f}, I_inf={I_inf:.3f}")

    # Print summary of saved files
    png_files = [png_path for _, _, _, success, png_path, _ in results if success and png_path]
    csv_files = [csv_path for _, _, _, success, _, csv_path in results if success and csv_path]
    print(f"PNG files saved: {len(png_files)}")
    print(f"CSV files saved: {len(csv_files)}")


# Example usage and testing
if __name__ == "__main__":
    import random

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Load the data
    df = pd.read_csv(
        "/home/jupyter/igor_repos/exploration/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/collect_mi_results.csv"
    )

    # Rename columns for consistency
    df = df.rename(columns={"algorithm": "method", "signal": "metric"})

    # Test with default parameters
    print("Testing with default parameters...")
    results = plot_cell_scaling_fits(df, initial_N0=10**4, initial_s=1.0, initial_I_inf=2.5)
    print(f"Results shape: {results.shape}")

    # Generate some test hyperparameter combinations
    def generate_hyperparameter_combinations(
        n_combinations=10, N0_range=(1000, 50000), s_range=(0.1, 2.0), I_inf_range=(0.5, 5.0)
    ):
        """Generate random hyperparameter combinations for testing."""
        combinations = []
        for _ in range(n_combinations):
            N0 = random.uniform(N0_range[0], N0_range[1])
            s = random.uniform(s_range[0], s_range[1])
            I_inf = random.uniform(I_inf_range[0], I_inf_range[1])
            combinations.append((N0, s, I_inf))
        return combinations

    # Generate test combinations
    print("\nGenerating test hyperparameter combinations...")
    hyperparameter_combinations = generate_hyperparameter_combinations(
        n_combinations=5, N0_range=(1000, 50000), s_range=(0.1, 2.0), I_inf_range=(0.5, 5.0)
    )

    print("Hyperparameter combinations:")
    for i, (N0, s, I_inf) in enumerate(hyperparameter_combinations):
        print(f"  {i+1:2d}: N0={N0:8.3f}, s={s:.3f}, I_inf={I_inf:.3f}")

    # Test with multiple hyperparameters
    print("\nTesting with multiple hyperparameters...")
    test_multiple_cell_scaling_hyperparameters(df, hyperparameter_combinations)
    print("All plots saved successfully!")
