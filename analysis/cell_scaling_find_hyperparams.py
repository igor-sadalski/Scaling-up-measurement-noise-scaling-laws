import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
import random
import multiprocessing as mp
import shutil
from multiprocessing import Pool
from lmfit import Model
from scipy.interpolate import interp1d

def cell_number_scaling(x, N0, s, I_inf):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(x > 0, I_inf - (x / N0) ** (-s), np.nan)
    return result


def _get_parameter_bounds(method, mi_values, initial_N0, initial_s, initial_I_inf):
    if method in ["PCA", "RandomProjection"]:
        return {
            "N0": dict(value=initial_N0, min=1e-6),
            "s": dict(value=initial_s, min=1e-3),
            "I_inf": dict(value=initial_I_inf, min=mi_values.max(), max=mi_values.max() * 1.5),
        }
    else:
        params = {"N0": initial_N0, "s": initial_s, "I_inf": initial_I_inf}
        bounds = {
            "N0": (10, 10**7),
            "s": (0.01, 10.0),
            "I_inf": (0.1, 10.0),
        }
        return {k: dict(value=v, min=bounds[k][0], max=bounds[k][1]) 
                for k, v in params.items()}


def fit_cell_number_scaling_model(cell_values, mi_values, method, initial_N0, initial_s, initial_I_inf):
    model = Model(cell_number_scaling)
    params = model.make_params(**_get_parameter_bounds(method, mi_values, initial_N0, initial_s, initial_I_inf))
    
    try:
        result = model.fit(mi_values, params, x=cell_values)
        
        fit_params = {k: result.params[k].value for k in ["N0", "s", "I_inf"]}
        fit_errors = {f"{k}_err": result.params[k].stderr for k in ["N0", "s", "I_inf"]}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                y_cell_upper = result.eval_uncertainty(params=result.params, x=cell_values, sigma=2)
            except Exception:
                y_cell_upper = np.zeros_like(cell_values)

        return {
            **fit_params,
            **fit_errors,
            "mean_residual": np.mean(np.abs(result.residual)),
            "fit_success": True,
            "result": result,
            "y_cell_upper": y_cell_upper,
            "y_cell_lower": -y_cell_upper,
        }
    except Exception as e:
        print(f"Fitting algorithm failed for method {method}: {str(e)}")
        return {
            "N0": np.nan, "s": np.nan, "I_inf": np.nan,
            "N0_err": np.nan, "s_err": np.nan, "I_inf_err": np.nan,
            "mean_residual": np.nan, "fit_success": False,
            "result": None, "y_cell_upper": np.nan, "y_cell_lower": np.nan,
        }


def _compute_uncertainty_bands(result_obj, x_vals, cell_range):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            y_upper_at_data = result_obj.eval_uncertainty(params=result_obj.params, x=x_vals, sigma=2)
            log_x = np.log10(x_vals + 1e-10)
            log_range = np.log10(cell_range + 1e-10)
            interp_func = interp1d(log_x, y_upper_at_data, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            y_upper = interp_func(log_range)
            return y_upper, y_upper_at_data
        except Exception:
            try:
                x_coarse = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_upper_coarse = result_obj.eval_uncertainty(params=result_obj.params, x=x_coarse, sigma=2)
                log_coarse = np.log10(x_coarse + 1e-10)
                log_range = np.log10(cell_range + 1e-10)
                interp_func = interp1d(log_coarse, y_upper_coarse, kind='linear',
                                      bounds_error=False, fill_value='extrapolate')
                y_upper = interp_func(log_range)
                return y_upper, y_upper_coarse
            except Exception:
                return np.zeros_like(cell_range), np.zeros_like(x_vals)


def _plot_quality_data(ax, df, quality_data, quality, quality_color_map):
    first_row = quality_data.iloc[0]
    
    mask = ((df["dataset"] == first_row["dataset"]) & 
            (df["method"] == first_row["method"]) &
            (df["metric"] == first_row["metric"]) &
            (df["quality"] == first_row["quality"]))
    original_data = df[mask]
    
    grouped = original_data.groupby("size")["mi_value"]
    x_vals = np.array(grouped.mean().index)
    y_means = grouped.mean().values
    y_stds = grouped.std().fillna(0).values
    
    color = quality_color_map[quality]
    if np.any(y_stds > 0):
        ax.errorbar(x_vals, y_means, yerr=y_stds, color=color, fmt="o", capsize=5, 
                   capthick=2, alpha=0.7, markersize=6, label=f"Quality {quality:.3f}")
    else:
        ax.scatter(x_vals, y_means, color=color, alpha=0.7, s=50, label=f"Quality {quality:.3f}")
    
    cell_range = np.linspace(original_data["size"].min(), original_data["size"].max(), 5000)
    fitted_curve = cell_number_scaling(cell_range, first_row["N0"], first_row["s"], first_row["I_inf"])
    ax.plot(cell_range, fitted_curve, color=color, linestyle="--", lw=2, alpha=0.8)
    
    if first_row["result"] is not None:
        y_upper, y_upper_at_data = _compute_uncertainty_bands(first_row["result"], x_vals, cell_range)
        max_uncertainty = np.max(np.abs(y_upper_at_data))
        if max_uncertainty <= 0.5:
            ax.fill_between(cell_range, fitted_curve + y_upper, fitted_curve - y_upper,
                           color=color, alpha=0.1)


def _create_filename(prefix, initial_N0, initial_s, initial_I_inf, ext):
    return f"{prefix}_N0_{initial_N0:.3f}_s_{initial_s:.3f}_I_inf_{initial_I_inf:.3f}.{ext}"


def plot_cell_scaling_fits(df, initial_N0=10**4, initial_s=1.0, initial_I_inf=2.5,
                          save_plots=True, output_dir=None, show_plot=True, save_uncertainty=True):
    if output_dir is None:
        output_dir = "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/final_results"
    os.makedirs(output_dir, exist_ok=True)

    unique_combinations = df.groupby(["dataset", "method", "metric", "quality"]).size().reset_index()
    results_list = []
    failed_fits = []

    for _, row in unique_combinations.iterrows():
        filtered_data = df[
            (df["dataset"] == row["dataset"]) & (df["method"] == row["method"]) &
            (df["metric"] == row["metric"]) & (df["quality"] == row["quality"])
        ]
        
        if len(filtered_data) < 3:
            continue

        fit_results = fit_cell_number_scaling_model(
            filtered_data["size"].values, filtered_data["mi_value"].values,
            row["method"], initial_N0, initial_s, initial_I_inf
        )
        
        results_list.append({
            "dataset": row["dataset"], "method": row["method"],
            "metric": row["metric"], "quality": row["quality"],
            "N0": fit_results["N0"], "s": fit_results["s"], "I_inf": fit_results["I_inf"],
            "N0_error": fit_results["N0_err"], "s_error": fit_results["s_err"],
            "I_inf_error": fit_results["I_inf_err"], "mean_residual": fit_results["mean_residual"],
            "result": fit_results["result"],
        })

        if not fit_results["fit_success"]:
            failed_fits.append(f"{row['dataset']}/{row['method']}/{row['metric']}/quality={row['quality']:.3f}")

    results_df = pd.DataFrame(results_list)

    if failed_fits:
        print(f"\nTotal failed fits: {len(failed_fits)}")
        for failed in failed_fits:
            print(f"  - {failed}")
    else:
        print("All fits successful!")

    successful_results = results_df[results_df["N0"].notna()].copy()
    
    if len(successful_results) > 0 and save_plots:
        metrics = successful_results["metric"].unique()
        methods = successful_results["method"].unique()
        fig, axes = plt.subplots(len(metrics), len(methods), 
                                figsize=(4 * len(methods), 4 * len(metrics)))
        
        if len(metrics) == 1 and len(methods) == 1:
            axes = np.array([[axes]])
        elif len(metrics) == 1:
            axes = axes.reshape(1, -1)
        elif len(methods) == 1:
            axes = axes.reshape(-1, 1)

        qualities = sorted(successful_results["quality"].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(qualities)))
        quality_color_map = dict(zip(qualities, colors))

        for i, metric in enumerate(metrics):
            for j, method in enumerate(methods):
                ax = axes[i, j]
                subset = successful_results[
                    (successful_results["metric"] == metric) & 
                    (successful_results["method"] == method)
                ]

                if len(subset) > 0:
                    for quality in qualities:
                        quality_data = subset[subset["quality"] == quality]
                        if len(quality_data) > 0:
                            _plot_quality_data(ax, df, quality_data, quality, quality_color_map)

                    ax.set_xlabel("Number of cells")
                    ax.set_ylabel("MI value")
                    ax.set_title(f"{method}\n{metric}")
                    ax.set_xscale("log")
                    ax.grid(True, alpha=0.3)
                    if i == 0 and j == 0:
                        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                else:
                    ax.set_xlabel("Number of cells")
                    ax.set_ylabel("MI value")
                    ax.set_title(f"{method}\n{metric}\n(No data)")
                    ax.set_xscale("log")
                    ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            filename = _create_filename("cell_scaling", initial_N0, initial_s, initial_I_inf, "png")
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {filepath}")

        plt.show() if show_plot else plt.close()

    if save_plots and save_uncertainty:
        csv_filename = _create_filename("cell_scaling", initial_N0, initial_s, initial_I_inf, "csv")
        csv_filepath = os.path.join(output_dir, csv_filename)
        numerical_cols = ["N0", "s", "I_inf", "N0_error", "s_error", "I_inf_error", "mean_residual"]
        results_df[numerical_cols] = results_df[numerical_cols].round(3)
        results_df.to_csv(csv_filepath, index=False)
        print(f"Results saved to: {csv_filepath}")

    return results_df if save_uncertainty else None


def generate_hyperparameter_combinations(n_combinations=50, N0_range=(1000, 50000),
                                         s_range=(0.1, 2.0), I_inf_range=(0.5, 5.0)):
    return [(random.uniform(N0_range[0], N0_range[1]),
             random.uniform(s_range[0], s_range[1]),
             random.uniform(I_inf_range[0], I_inf_range[1]))
            for _ in range(n_combinations)]


def plot_single_hyperparameter(args):
    df, N0, s, I_inf, output_dir = args
    try:
        plot_cell_scaling_fits(df, initial_N0=N0, initial_s=s, initial_I_inf=I_inf,
                              save_plots=True, output_dir=output_dir, show_plot=False,
                              save_uncertainty=True)
        return (N0, s, I_inf, True, 
                os.path.join(output_dir, _create_filename("cell_scaling", N0, s, I_inf, "png")),
                os.path.join(output_dir, _create_filename("cell_scaling", N0, s, I_inf, "csv")))
    except Exception as e:
        print(f"Error with N0={N0:.3f}, s={s:.3f}, I_inf={I_inf:.3f}: {str(e)}")
        return (N0, s, I_inf, False, None, None)


def test_multiple_cell_scaling_hyperparameters(df, hyperparameter_combinations, n_processes=None):
    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(hyperparameter_combinations))

    output_dir = "/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/intermediate_cell_scaling_fits"

    if os.path.exists(output_dir):
        print(f"Clearing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    args_list = [(df, N0, s, I_inf, output_dir) for N0, s, I_inf in hyperparameter_combinations]
    print(f"Generating {len(hyperparameter_combinations)} plots using {n_processes} processes...")

    with Pool(processes=n_processes) as pool:
        results = pool.map(plot_single_hyperparameter, args_list)

        print(f"Completed processing {len(results)} hyperparameter combinations")

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

    png_files = [png for _, _, _, success, png, _ in results if success and png]
    csv_files = [csv for _, _, _, success, _, csv in results if success and csv]
    print(f"PNG files saved: {len(png_files)}")
    print(f"CSV files saved: {len(csv_files)}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    df = pd.read_csv("/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/collect_mi_results.csv")
    df = df.rename(columns={"algorithm": "method", "signal": "metric"})

    print("Testing with default parameters...")
    results = plot_cell_scaling_fits(df, initial_N0=10**4, initial_s=1.0, initial_I_inf=2.5, save_uncertainty=True)
    print(f"Results shape: {results.shape}")

    print("\nGenerating 50 test hyperparameter combinations...")
    hyperparameter_combinations = generate_hyperparameter_combinations(
        n_combinations=100, N0_range=(1000, 10**7), s_range=(0.1, 10.0), I_inf_range=(0.5, 10.0)
    )

    print("Hyperparameter combinations:")
    for i, (N0, s, I_inf) in enumerate(hyperparameter_combinations[:10]):
        print(f"  {i+1:2d}: N0={N0:8.3f}, s={s:.3f}, I_inf={I_inf:.3f}")
    print("  ...")
    last = hyperparameter_combinations[-1]
    print(f"  {len(hyperparameter_combinations)}: N0={last[0]:8.3f}, s={last[1]:.3f}, I_inf={last[2]:.3f}")

    print("\nTesting with 50 different hyperparameter combinations...")
    test_multiple_cell_scaling_hyperparameters(df, hyperparameter_combinations)
    print("All plots saved successfully!")
