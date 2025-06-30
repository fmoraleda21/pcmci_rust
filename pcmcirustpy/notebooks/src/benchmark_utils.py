import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import pcmcirustpy


class CausalStructure:
    def __init__(self, edge_list, var_names):
        self.edge_list = edge_list
        self.var_names = var_names

    def as_dict(self):
        """
        Returns the causal structure as a dictionary mapping target variable indices
        to lists of (source index, lag, coefficient) tuples.
        """
        var_indices = {name: i for i, name in enumerate(self.var_names)}
        causal_structure_dict = {}

        for src, tgt, lag, coeff in self.edge_list:
            tgt_idx = var_indices[tgt]
            src_idx = var_indices[src]

            if tgt_idx not in causal_structure_dict:
                causal_structure_dict[tgt_idx] = []
            causal_structure_dict[tgt_idx].append((src_idx, lag, coeff))

        return causal_structure_dict

    def plot(self, title="", ax=None):
        """
        Plots a time-unrolled causal graph.
        If ax is given, draw on it; otherwise create a new figure.
        """

        var_names = self.var_names

        if var_names is None:
            var_set = {src for src, *_ in self.edge_list} | {
                tgt for _, tgt, *_ in self.edge_list
            }
            var_names = sorted(var_set)

        var_indices = {v: i for i, v in enumerate(var_names)}
        n_vars = len(var_names)

        max_lag = max(lag for _, _, lag, _ in self.edge_list)

        if ax is None:
            fig, ax = plt.subplots(figsize=(3 * max_lag + 2, 0.8 * n_vars + 2))

        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap("YlGn")

        # Draw variable nodes
        for var, y in var_indices.items():
            for lag in range(max_lag + 1):
                ax.plot(lag, y, "o", color="lightblue", markersize=8)
            ax.text(max_lag + 0.4, y, var, va="center", fontsize=10)

        # Draw arrows
        for src, tgt, lag, weight in self.edge_list:
            src_y = var_indices[src]
            tgt_y = var_indices[tgt]

            color = cmap(norm(abs(weight)))
            linewidth = 2.5 + 3.0 * norm(abs(weight))

            ax.annotate(
                "",
                xy=(0, tgt_y),
                xytext=(lag, src_y),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=linewidth,
                    color=color,
                    shrinkA=5,
                    shrinkB=5,
                ),
            )

        ax.set_yticks(range(n_vars))
        ax.set_yticklabels(var_names)
        ax.set_xticks(range(max_lag + 1))
        ax.set_xticklabels([f"lag {l}" for l in range(max_lag + 1)])
        ax.set_xlim(-0.5, max_lag + 1)
        ax.set_ylim(-1, n_vars)
        ax.invert_xaxis()
        ax.set_xlabel("Lag")
        ax.set_title(title, fontsize=13, pad=12)
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)

        # Colorbar only if standalone figure
        if ax is not None and hasattr(ax, "figure"):
            fig = ax.figure
        else:
            fig = plt.gcf()

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02)
        cbar.set_label("Causal 'strength' (abs(weight))", rotation=90)

        if ax is None:
            plt.tight_layout()
            plt.show()


class BenchmarkParam:
    def __init__(self, causal_structure, noise_level):
        """
        Args:
            causal_structure (CausalStructure): Causal structure for synthetic data.
            noise_level (float): Noise level in synthetic data.
        """
        self.causal_structure = causal_structure
        self.noise_level = noise_level

        self.param_name = None
        self.results = None

    def _generate_synthetic_data(
        self, n_time, n_vars, max_lag, causal_structure, noise_level=0.1, seed=42
    ):
        np.random.seed(seed)
        burn_in = 100
        data = np.zeros((n_time + burn_in, n_vars))
        data[:max_lag] = np.random.rand(max_lag, n_vars)

        for t in range(max_lag, n_time + burn_in):
            for target_var in range(n_vars):
                value = 0.0
                if target_var in causal_structure:
                    for source_var, lag, coeff in causal_structure[target_var]:
                        value += coeff * data[t - lag, source_var]
                value += np.random.normal(0, noise_level)
                data[t, target_var] = value

        return data[burn_in:]

    def _run_pcmci_tigramite(
        self, data, max_lag, alpha, max_condition_set_size, max_subsets
    ):
        df = DataFrame(data, var_names=[f"var{i}" for i in range(data.shape[1])])
        pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr())

        start = time.time()
        results = pcmci.run_pcmci(
            tau_max=max_lag,
            pc_alpha=alpha,
            max_conds_dim=max_condition_set_size,
            max_combinations=max_subsets,
        )
        end = time.time()

        return results, end - start

    def _run_pcmci_rust(
        self, data, max_lag, alpha, max_condition_set_size, max_subsets
    ):
        start = time.time()
        results = pcmcirustpy.run_pcmci(
            data, max_lag, alpha, max_condition_set_size, max_subsets
        )
        end = time.time()

        return results, end - start

    def _calculate_difference_metrics(self, matrix1, matrix2):
        # Mask valid values
        valid_mask = ~np.isnan(matrix1) & ~np.isnan(matrix2)
        x = matrix1[valid_mask].flatten()
        y = matrix2[valid_mask].flatten()

        if x.size == 0:
            return {
                "mean_rel_error": np.nan,
                "cosine_similarity": np.nan,
                "pearson_corr": np.nan,
            }

        # Mean Relative Error (MAPE-like, symmetric)
        mean_rel_error = np.mean(np.abs(x - y) / (np.abs(x) + np.abs(y) + 1e-8))

        # Cosine similarity (1 = perfect match)
        cosine_similarity = 1 - cosine(x, y)

        # Pearson correlation (1 = perfect linear match)
        pearson_corr, _ = pearsonr(x, y)

        return {
            "mean_rel_error": mean_rel_error,
            "cosine_similarity": cosine_similarity,
            "pearson_corr": pearson_corr,
        }

    def run(
        self,
        param_name,
        param_values,
        fixed_params,
        n_runs=5,
    ):
        """
        Benchmarks Tigramite and Rust PCMCI runtimes and differences over a range of a single parameter.
        Runs each configuration multiple times for stability.

        Args:
            param_name (str): Parameter to vary.
            param_values (list): Values to iterate over.
            fixed_params (dict): Other parameters held fixed.
            n_runs (int): Number of runs per parameter value.

        Returns:
            pd.DataFrame: Results with means over n_runs.
        """
        tig_times = []
        rust_times = []
        mean_rel_errors = []
        cosine_sims = []
        pearson_corrs = []

        causal_structure_dict = self.causal_structure.as_dict()

        for val in param_values:
            params = fixed_params.copy()
            params[param_name] = val

            tig_time_runs = []
            rust_time_runs = []
            rel_errors = []
            cos_sims = []
            pearsons = []

            for _ in range(n_runs):
                data = self._generate_synthetic_data(
                    params["n_time"],
                    params["n_vars"],
                    params["max_lag"],
                    causal_structure_dict,
                    self.noise_level,
                )

                result_tig, time_tig = self._run_pcmci_tigramite(
                    data,
                    params["max_lag"],
                    params["alpha"],
                    params["cond_size"],
                    params["subsets"],
                )

                result_rust, time_rust = self._run_pcmci_rust(
                    data,
                    params["max_lag"],
                    params["alpha"],
                    params["cond_size"],
                    params["subsets"],
                )

                metrics = self._calculate_difference_metrics(
                    result_tig["val_matrix"], result_rust["val_matrix"]
                )

                tig_time_runs.append(time_tig)
                rust_time_runs.append(time_rust)
                rel_errors.append(metrics["mean_rel_error"])
                cos_sims.append(metrics["cosine_similarity"])
                pearsons.append(metrics["pearson_corr"])

            # Store mean of n_runs
            tig_times.append(np.mean(tig_time_runs))
            rust_times.append(np.mean(rust_time_runs))
            mean_rel_errors.append(np.mean(rel_errors))
            cosine_sims.append(np.mean(cos_sims))
            pearson_corrs.append(np.mean(pearsons))

        df_results = pd.DataFrame(
            {
                param_name: param_values,
                "time_tigramite": tig_times,
                "time_rust": rust_times,
                "mean_rel_error": mean_rel_errors,
                "cosine_similarity": cosine_sims,
                "pearson_corr": pearson_corrs,
            }
        )

        self.param_name = param_name
        self.results = df_results

        return df_results

    def plot_exec_time(self):
        plt.figure(figsize=(8, 5))
        plt.plot(
            self.results[self.param_name],
            self.results["time_tigramite"],
            label="Tigramite",
            marker="o",
        )
        plt.plot(
            self.results[self.param_name],
            self.results["time_rust"],
            label="Rust",
            marker="x",
        )
        plt.title(f"Execution Time vs {self.param_name}")
        plt.xlabel(self.param_name)
        plt.ylabel("Time (s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_causal_structures(self, fixed_params):
        """
        Plots inferred causal structures side-by-side for Tigramite and Rust PCMCI.
        Uses the existing CausalStructure.plot() method for consistency.
        """
        # Generate fresh synthetic data with the same causal ground truth
        data = self._generate_synthetic_data(
            fixed_params["n_time"],
            fixed_params["n_vars"],
            fixed_params["max_lag"],
            self.causal_structure.as_dict(),
            self.noise_level,
        )

        # Run Tigramite
        result_tig, _ = self._run_pcmci_tigramite(
            data,
            fixed_params["max_lag"],
            fixed_params["alpha"],
            fixed_params["cond_size"],
            fixed_params["subsets"],
        )

        # Run Rust
        result_rust, _ = self._run_pcmci_rust(
            data,
            fixed_params["max_lag"],
            fixed_params["alpha"],
            fixed_params["cond_size"],
            fixed_params["subsets"],
        )

        # Extract inferred links (use val_matrix and some threshold)
        edges_tig = []
        val_matrix_tig = result_tig["val_matrix"]
        p_matrix_tig = result_tig["p_matrix"]

        for src in range(val_matrix_tig.shape[0]):
            for tgt in range(val_matrix_tig.shape[1]):
                for lag in range(1, val_matrix_tig.shape[2]):
                    effect = val_matrix_tig[src, tgt, lag]
                    if (
                        not np.isnan(effect)
                        and abs(effect) > 0.01
                        and p_matrix_tig[src, tgt, lag] < fixed_params["alpha"]
                    ):
                        # Only include edges with significant p-value and non-zero effect size
                        edges_tig.append((f"var{src}", f"var{tgt}", lag, effect))

        edges_rust = []
        val_matrix_rust = result_rust["val_matrix"]
        p_matrix_rust = result_rust["p_matrix"]

        for src in range(val_matrix_rust.shape[0]):
            for tgt in range(val_matrix_rust.shape[1]):
                for lag in range(1, val_matrix_rust.shape[2]):
                    effect = val_matrix_rust[src, tgt, lag]
                    if (
                        not np.isnan(effect)
                        and abs(effect) > 0.01
                        and p_matrix_rust[src, tgt, lag] < fixed_params["alpha"]
                    ):
                        # Only include edges with significant p-value and non-zero effect size
                        edges_rust.append((f"var{src}", f"var{tgt}", lag, effect))

        var_names = [f"var{i}" for i in range(fixed_params["n_vars"])]

        causal_tig = CausalStructure(edges_tig, var_names)
        causal_rust = CausalStructure(edges_rust, var_names)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        causal_tig.plot(title="Tigramite PCMCI", ax=axes[0])
        causal_rust.plot(title="Rust PCMCI", ax=axes[1])

        plt.tight_layout()
        plt.show()
