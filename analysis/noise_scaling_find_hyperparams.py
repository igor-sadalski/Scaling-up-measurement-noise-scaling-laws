import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Model
import warnings
import os
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import random


def plot_noise_scaling_fits(
    df, initial_u_bar=540, initial_I_max=3, save_plots=True, output_dir=None, show_plot=True, save_uncertainty=True
):
    """
    Fit and plot noise scaling model for all (dataset, size, method, metric) groups in df.
    Parameters:
        df: DataFrame with columns in`cl`uding 'dataset', 'size', 'method', 'metric', 'umis_per_cell', 'mi_value'
        initial_u_bar: initial guess for u_bar (A_info)
        initial_I_max: initial guess for I_max (B_info)
        save_plots: whether to save plots to file
        output_dir: directory to save plots (default: different_starting_params)
        show_plot: whether to display the plot (set to False for multiprocessing)
        save_uncertainty: whether to save uncertainty data to CSV
    Output:
        Shows a combined plot of curves with their fits and uncertainty bands.
        Returns: DataFrame with uncertainty information if save_uncertainty=True
    """

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

    def fit_noise_scaling_model(u_values, mi_values, initial_u_bar, initial_I_max, method=None, metric=None):
        """
        Fit the noise scaling model to data and return u_bar and I_max with uncertainties.
        """

        def info_scaling_local(u, u_bar, I_max):
            # Avoid division by zero and invalid values
            u = np.asarray(u)
            u_bar = np.asarray(u_bar)
            I_max = np.asarray(I_max)
            epsilon = 1e-12
            u_bar_safe = np.where(u_bar == 0, epsilon, u_bar)
            u_over_u_bar = u / u_bar_safe
            numerator = 1 + u_over_u_bar
            denominator = u_over_u_bar + 2 ** (-2 * I_max)
            denominator = np.where(denominator == 0, epsilon, denominator)
            ratio = numerator / denominator
            ratio = np.where(ratio <= 0, epsilon, ratio)
            return I_max - 0.5 * np.log2(ratio)

        model = Model(info_scaling_local)
        params = model.make_params(u_bar=initial_u_bar, I_max=initial_I_max)
        params["u_bar"].min = 0
        params["I_max"].min = 0

        try:
            result = model.fit(mi_values, params, u=u_values)
            u_bar_fit = result.params["u_bar"].value
            I_max_fit = result.params["I_max"].value
            u_bar_err = result.params["u_bar"].stderr
            I_max_err = result.params["I_max"].stderr

            u_fit = u_values
            # Suppress warnings from eval_uncertainty if any
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    y_info = result.eval_uncertainty(params=result.params, u=u_fit, sigma=2)
                except Exception:
                    y_info = np.zeros_like(u_fit)
            y_info_lower = -y_info

            return {
                "fit_success": True,
                "result": result,
                "u_bar": u_bar_fit,
                "I_max": I_max_fit,
                "u_bar_err": u_bar_err,
                "I_max_err": I_max_err,
                "y_info": y_info,
                "y_info_lower": y_info_lower,
            }
        except Exception as e:
            return {
                "fit_success": False,
                "result": None,
                "u_bar": np.nan,
                "I_max": np.nan,
                "u_bar_err": np.nan,
                "I_max_err": np.nan,
                "y_info": np.nan,
                "y_info_lower": np.nan,
            }

    # Get unique combinations of method, metric, and dataset for plotting
    unique_combinations = df.groupby(["method", "metric", "dataset"]).size().reset_index(name="count")
    all_fit_results = []
    uncertainty_data = []  # Store uncertainty information

    for idx, row in unique_combinations.iterrows():
        method = row["method"]
        metric = row["metric"]
        dataset = row["dataset"]

        # Get all data for this method-metric-dataset combination
        filtered_data = df[(df["method"] == method) & (df["metric"] == metric) & (df["dataset"] == dataset)]

        if len(filtered_data) < 3:
            continue

        # Group by size for fitting
        sizes = filtered_data["size"].unique()
        for size in sizes:
            size_data = filtered_data[filtered_data["size"] == size]

            if len(size_data) < 3:
                continue

            u_values = size_data["umis_per_cell"].values
            mi_values = size_data["mi_value"].values

            fit_results = fit_noise_scaling_model(u_values, mi_values, initial_u_bar, initial_I_max, method, metric)

            if fit_results["fit_success"]:
                fitted_mi_values = info_scaling(u_values, fit_results["u_bar"], fit_results["I_max"])
                size_data_with_fits = size_data.copy()
                size_data_with_fits["fitted_mi_value"] = fitted_mi_values
                size_data_with_fits["u_bar"] = fit_results["u_bar"]
                size_data_with_fits["I_max"] = fit_results["I_max"]
                size_data_with_fits["u_bar_err"] = fit_results["u_bar_err"]
                size_data_with_fits["I_max_err"] = fit_results["I_max_err"]
                size_data_with_fits["y_info"] = fit_results["y_info"]
                size_data_with_fits["y_info_lower"] = fit_results["y_info_lower"]
                all_fit_results.append(size_data_with_fits)

                # Store uncertainty information
                uncertainty_data.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "size": size,
                        "fitted_u_bar": round(fit_results["u_bar"], 3),
                        "fitted_I_max": round(fit_results["I_max"], 3),
                        "u_bar_error": round(fit_results["u_bar_err"], 3),
                        "I_max_error": round(fit_results["I_max_err"], 3),
                    }
                )
            else:
                # Store failed fit information
                uncertainty_data.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "size": size,
                        "fitted_u_bar": np.nan,
                        "fitted_I_max": np.nan,
                        "u_bar_error": np.nan,
                        "I_max_error": np.nan,
                    }
                )

    if not all_fit_results:
        print("No successful fits to plot.")
        return None

    # Combine all results into a single DataFrame
    combined_results = pd.concat(all_fit_results, ignore_index=True)

    # Create uncertainty DataFrame
    uncertainty_df = pd.DataFrame(uncertainty_data)

    # Get unique combinations of metric, method, and dataset
    metrics = combined_results["metric"].unique()
    methods = combined_results["method"].unique()
    datasets = combined_results["dataset"].unique()

    fig, axes = plt.subplots(len(metrics), len(methods), figsize=(4 * len(methods), 4 * len(metrics)))

    if len(metrics) == 1 and len(methods) == 1:
        axes = np.array([[axes]])
    elif len(metrics) == 1:
        axes = axes.reshape(1, -1)
    elif len(methods) == 1:
        axes = axes.reshape(-1, 1)

    sizes = combined_results["size"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(sizes)))
    size_color_map = dict(zip(sizes, colors))

    for i, metric in enumerate(metrics):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            subset = combined_results[(combined_results["metric"] == metric) & (combined_results["method"] == method)]
            legend_labels = []
            legend_handles = []
            if len(subset) > 0:
                # Get unique datasets for this method-metric combination
                subset_datasets = subset["dataset"].unique()
                dataset_styles = ["-", "--", "-.", ":"]
                dataset_markers = ["o", "s", "^", "v", "D", "p", "*", "h"]

                for dataset in subset_datasets:
                    dataset_data = subset[subset["dataset"] == dataset]
                    for size in sizes:
                        size_data = dataset_data[dataset_data["size"] == size]
                        if len(size_data) > 0:
                            # Group by umis_per_cell to handle multiple seeds
                            umis_vals = size_data["umis_per_cell"].unique()
                            x_vals = []
                            y_means = []
                            y_stds = []
                            for umis in umis_vals:
                                group = size_data[size_data["umis_per_cell"] == umis]
                                x_vals.append(umis)
                                mi_values = group["mi_value"]
                                y_means.append(np.mean(mi_values))
                                y_stds.append(np.std(mi_values) if len(mi_values) > 1 else 0)
                            x_vals = np.array(x_vals)
                            y_means = np.array(y_means)
                            y_stds = np.array(y_stds)

                            # Choose style based on dataset
                            dataset_idx = list(subset_datasets).index(dataset)
                            linestyle = dataset_styles[dataset_idx % len(dataset_styles)]
                            marker = dataset_markers[dataset_idx % len(dataset_markers)]

                            if np.any(y_stds > 0):
                                handle = ax.errorbar(
                                    x_vals,
                                    y_means,
                                    yerr=y_stds,
                                    color=size_color_map[size],
                                    fmt=marker,
                                    capsize=5,
                                    capthick=2,
                                    alpha=0.7,
                                    markersize=6,
                                    label=f"{dataset} Size {size}",
                                )
                                legend_handles.append(handle)
                                legend_labels.append(f"{dataset} Size {size}")
                            else:
                                handle = ax.scatter(
                                    x_vals,
                                    y_means,
                                    color=size_color_map[size],
                                    marker=marker,
                                    alpha=0.7,
                                    s=50,
                                    label=f"{dataset} Size {size}",
                                )
                                legend_handles.append(handle)
                                legend_labels.append(f"{dataset} Size {size}")
                            # Plot fitted line for this size with uncertainty bands
                            u_range = np.linspace(
                                np.min(size_data["umis_per_cell"]), np.max(size_data["umis_per_cell"]), 100
                            )
                            u_bar = size_data["u_bar"].iloc[0]
                            I_max = size_data["I_max"].iloc[0]
                            y_info = size_data["y_info"].iloc[0]
                            y_info_lower = size_data["y_info_lower"].iloc[0]
                            fitted_curve = info_scaling(u_range, u_bar, I_max)
                            # Make sure y_info and y_info_lower are arrays of correct shape
                            if np.isscalar(y_info):
                                y_info = np.full_like(u_range, y_info)
                            if np.isscalar(y_info_lower):
                                y_info_lower = np.full_like(u_range, y_info_lower)

                            # Only plot uncertainty bands if y_info error is not too large
                            max_uncertainty = np.max(np.abs(y_info))
                            if max_uncertainty <= 0.5:
                                ax.fill_between(
                                    u_range,
                                    fitted_curve + y_info,
                                    fitted_curve + y_info_lower,
                                    color=size_color_map[size],
                                    alpha=0.1,
                                )
                            else:
                                print(
                                    f"Skipping uncertainty bands for {method} {metric} {dataset} size {size}: max uncertainty = {max_uncertainty:.3f} > 0.5"
                                )
                            ax.plot(
                                u_range, fitted_curve, color=size_color_map[size], linestyle=linestyle, lw=1, alpha=0.5
                            )
                ax.set_xlabel("UMIs per cell")
                ax.set_ylabel("MI value")
                ax.set_title(f"{method}\n{metric}")
                ax.set_xscale("log")
                ax.grid(True, alpha=0.3)
                # Only add legend if there are any handles with labels
                if i == 0 and j == 0 and legend_handles:
                    ax.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.05, 1), loc="upper left")
            else:
                ax.set_title(f"{method}\n{metric}\n(No data)")
    plt.tight_layout()

    # Save plot if requested
    if save_plots:
        if output_dir is None:
            output_dir = "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/intermediate_fits_noise_scaling"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create filename with hyperparameters (3 decimal places)
        filename = f"scaling_plots_u_bar_{initial_u_bar:.3f}_I_max_{initial_I_max:.3f}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {filepath}")

        # Save uncertainty data to CSV if requested
        if save_uncertainty:
            csv_filename = f"scaling_plots_u_bar_{initial_u_bar:.3f}_I_max_{initial_I_max:.3f}.csv"
            csv_filepath = os.path.join(output_dir, csv_filename)
            uncertainty_df.to_csv(csv_filepath, index=False)
            print(f"Uncertainty data saved to: {csv_filepath}")

    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory when not showing

    # Return uncertainty DataFrame if save_uncertainty is True
    if save_uncertainty:
        return uncertainty_df
    else:
        return None


def generate_hyperparameter_combinations(n_combinations=50, u_bar_range=(50, 2000), I_max_range=(0.5, 5.0)):
    """
    Generate random hyperparameter combinations for testing.

    Parameters:
        n_combinations: number of combinations to generate
        u_bar_range: tuple of (min, max) for u_bar values
        I_max_range: tuple of (min, max) for I_max values

    Returns:
        list of tuples (u_bar, I_max)
    """
    combinations = []
    for _ in range(n_combinations):
        u_bar = random.uniform(u_bar_range[0], u_bar_range[1])
        I_max = random.uniform(I_max_range[0], I_max_range[1])
        combinations.append((u_bar, I_max))
    return combinations


def plot_single_hyperparameter(args):
    """
    Plot function for multiprocessing. Takes a tuple of (df, u_bar, I_max, output_dir).

    Parameters:
        args: tuple containing (df, u_bar, I_max, output_dir)

    Returns:
        tuple: (u_bar, I_max, success_status, png_filepath, csv_filepath)
    """
    df, u_bar, I_max, output_dir = args

    try:
        # Call the plotting function with show_plot=False for multiprocessing
        uncertainty_df = plot_noise_scaling_fits(
            df,
            initial_u_bar=u_bar,
            initial_I_max=I_max,
            save_plots=True,
            output_dir=output_dir,
            show_plot=False,
            save_uncertainty=True,
        )

        png_filename = f"scaling_plots_u_bar_{u_bar:.3f}_I_max_{I_max:.3f}.png"
        png_filepath = os.path.join(output_dir, png_filename)

        csv_filename = f"scaling_plots_u_bar_{u_bar:.3f}_I_max_{I_max:.3f}.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)

        return (u_bar, I_max, True, png_filepath, csv_filepath)
    except Exception as e:
        print(f"Error with u_bar={u_bar:.3f}, I_max={I_max:.3f}: {str(e)}")
        return (u_bar, I_max, False, None, None)


def test_multiple_hyperparameters(df, hyperparameter_combinations, n_processes=None):
    """
    Test multiple hyperparameter combinations and save plots for each using multiprocessing.

    Parameters:
        df: DataFrame with the data
        hyperparameter_combinations: list of tuples (u_bar, I_max)
        n_processes: number of processes to use (default: number of CPU cores)
    """
    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(hyperparameter_combinations))

    output_dir = "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/intermediate_fits_noise_scaling"

    # Clear and recreate the output directory only once at initialization
    if os.path.exists(output_dir):
        import shutil

        print(f"Clearing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Prepare arguments for multiprocessing
    args_list = [(df, u_bar, I_max, output_dir) for u_bar, I_max in hyperparameter_combinations]

    print(f"Generating {len(hyperparameter_combinations)} plots using {n_processes} processes...")

    # Use multiprocessing with progress bar
    with Pool(processes=n_processes) as pool:
        results = list(
            tqdm(pool.imap(plot_single_hyperparameter, args_list), total=len(args_list), desc="Generating plots")
        )

    # Print summary
    successful = sum(1 for _, _, success, _, _ in results if success)
    failed = len(results) - successful

    print(f"\nPlot generation complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("Failed combinations:")
        for u_bar, I_max, success, _, _ in results:
            if not success:
                print(f"  u_bar={u_bar:.3f}, I_max={I_max:.3f}")

    # Print summary of saved files
    png_files = [png_path for _, _, success, png_path, _ in results if success and png_path]
    csv_files = [csv_path for _, _, success, _, csv_path in results if success and csv_path]
    print(f"PNG files saved: {len(png_files)}")
    print(f"CSV files saved: {len(csv_files)}")


def test_multiple_hyperparameters_sequential(df, hyperparameter_combinations):
    """
    Test multiple hyperparameter combinations sequentially (for comparison).

    Parameters:
        df: DataFrame with the data
        hyperparameter_combinations: list of tuples (u_bar, I_max)
    """
    for u_bar, I_max in tqdm(hyperparameter_combinations, desc="Generating plots"):
        print(f"Testing u_bar={u_bar:.3f}, I_max={I_max:.3f}")
        plot_noise_scaling_fits(df, initial_u_bar=u_bar, initial_I_max=I_max, save_plots=True)


# Load data and run the function
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Load the data
    df = pd.read_csv(
        "/home/igor/exploration/scaling_laws_paper/Scaling-up-measurement-noise-scaling-laws/collect_mi_results.csv"
    )

    # Rename columns for consistency
    df = df.rename(columns={"algorithm": "method", "signal": "metric"})

    # Generate 50 random hyperparameter combinations
    print("Generating 50 random hyperparameter combinations...")
    hyperparameter_combinations = generate_hyperparameter_combinations(
        n_combinations=50, u_bar_range=(50, 1_000), I_max_range=(0.5, 10)
    )

    print("Hyperparameter combinations:")
    for i, (u_bar, I_max) in enumerate(hyperparameter_combinations[:10]):  # Show first 10
        print(f"  {i+1:2d}: u_bar={u_bar:8.3f}, I_max={I_max:.3f}")
    print("  ...")
    print(
        f"  {len(hyperparameter_combinations)}: u_bar={hyperparameter_combinations[-1][0]:8.3f}, I_max={hyperparameter_combinations[-1][1]:.3f}"
    )

    # Test with multiprocessing
    print("\nTesting with multiprocessing...")
    test_multiple_hyperparameters(df, hyperparameter_combinations)
    print("All plots saved successfully!")
