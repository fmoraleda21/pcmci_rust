use ndarray::{Array1, Array2, Array3, ArrayView2};
use ndarray_linalg::solve::Solve;
use ndarray_parallel::prelude::*;
use rayon::prelude::*;
use statrs::distribution::ContinuousCDF;
use std::collections::HashMap;

/// A candidate cause is given by the source variable index and lag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Candidate {
    pub var: usize,
    pub lag: usize,
}

/// An inferred link (without lag information).
#[derive(Clone, Copy, Debug)]
pub struct Link {
    pub source: usize,
    pub target: usize,
}

/// A causal link with detailed information about the effect.
#[derive(Clone, Debug)]
pub struct CausalLink {
    pub source: usize,
    pub target: usize,
    pub lag: usize,
    pub effect_size: f64,
    pub p_value: f64,
}

/// Efficiently stores and aligns time series data for different variables and lags
pub struct TimeSeriesCache {
    /// The aligned data for each (var, lag) combination
    pub cache: HashMap<(usize, usize), Array1<f64>>,
    /// Number of valid time points after alignment
    pub n_samples: usize,
}

/// PCMCI struct stores the raw data and results.
pub struct PCMCI<'a> {
    pub data: ArrayView2<'a, f64>,
    pub max_lag: usize,
    pub alpha: f64,
    /// The significance level to use for conditional independence tests.
    pub max_condition_set_size: usize,
    /// The maximum size of the conditioning set. This is a heuristic to avoid combinatorial explosion.
    pub max_subsets: usize,
    /// The maximum number of subsets to test for each candidate.
    pub parents: HashMap<usize, Vec<Candidate>>,
    pub causal_links: Vec<Link>,
    pub mci_links: Vec<CausalLink>,
}

impl<'a> PCMCI<'a> {
    /// Creates a new (empty) PCMCI instance.
    pub fn new() -> Self {
        PCMCI {
            data: ArrayView2::from_shape((0, 0), &[]).unwrap(),
            max_lag: 0,
            alpha: 0.05,
            max_condition_set_size: 3,
            max_subsets: 1,
            parents: HashMap::new(),
            causal_links: vec![],
            mci_links: vec![],
        }
    }

    /// Build a cache of all time series with different lags
    fn build_time_series_cache(&self) -> std::result::Result<TimeSeriesCache, String> {
        let n_time = self.data.dim().0;
        let n_vars = self.data.dim().1;
        let t0 = self.max_lag; // Alignment starting point

        let mut cache = HashMap::new();

        // Pre-compute all lagged time series
        for var in 0..n_vars {
            for lag in 0..=self.max_lag {
                let series = self.extract_series(&self.data, var, lag, t0)?;
                cache.insert((var, lag), series);
            }
        }

        Ok(TimeSeriesCache {
            cache,
            n_samples: n_time - t0,
        })
    }

    /// Extract a time series from the data for a given variable and lag
    fn extract_series(
        &self,
        data: &ArrayView2<f64>,
        var: usize,
        lag: usize,
        t0: usize,
    ) -> std::result::Result<Array1<f64>, String> {
        let n_time = data.dim().0;

        // Make sure that for every t from t0 to n_time-1, t - lag is a valid index
        if t0 < lag {
            return Err(format!("t0 ({}) must be >= lag ({})", t0, lag));
        }

        let n_samples = n_time - t0;

        // For large series, create in parallel using collect
        if n_samples > 1000 {
            // Build array in parallel and collect
            let result_vec: Vec<f64> = (0..n_samples)
                .into_par_iter()
                .map(|i| data[[i + t0 - lag, var]])
                .collect();

            // Convert Vec to Array1
            Ok(Array1::from_vec(result_vec))
        } else {
            // For smaller series, use regular sequential operations
            let mut result = Array1::<f64>::zeros(n_samples);
            for i in 0..n_samples {
                result[i] = data[[i + t0 - lag, var]];
            }
            Ok(result)
        }
    }

    /// Run the PC algorithm for a specific target variable
    fn run_pc_for_target(
        &self,
        target: usize,
        n_vars: usize,
        ts_cache: &TimeSeriesCache,
        n: usize,
    ) -> (usize, Vec<Candidate>) {
        // Build initial candidate set
        let mut candidates: Vec<Candidate> = Vec::new();
        for j in 0..n_vars {
            // if j == target {
            //     continue;
            // }
            for lag in 1..=self.max_lag {
                candidates.push(Candidate { var: j, lag });
            }
        }

        // Iterative removal procedure
        let mut removal_happened = true;
        while removal_happened {
            removal_happened = false;

            // Clone candidates for iteration
            let current_candidates = candidates.clone();
            'outer: for &cand in current_candidates.iter() {
                // Conditioning: all candidates except the current one
                let other_candidates: Vec<Candidate> =
                    candidates.iter().cloned().filter(|c| *c != cand).collect();
                let n_other = other_candidates.len();

                // Test with conditioning sets of increasing size
                for cond_set_size in 0..=std::cmp::min(n_other, self.max_condition_set_size) {
                    // Limit the number of subsets to test
                    let subsets = self.get_limited_subsets(&other_candidates, cond_set_size);

                    // Test with each conditioning set
                    for cond_set in subsets {
                        if let Ok((_, p_val)) = self.conditional_independence_test(
                            target,
                            cand,
                            &cond_set,
                            &ts_cache.cache,
                            n,
                        ) {
                            if p_val > self.alpha {
                                // Candidate deemed conditionally independent
                                candidates.retain(|&x| x != cand);
                                removal_happened = true;
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }

        (target, candidates)
    }

    /// Returns a limited number of subsets of items of a given size
    fn get_limited_subsets<T: Clone>(&self, items: &Vec<T>, size: usize) -> Vec<Vec<T>> {
        let n = items.len();

        if size > n {
            return vec![];
        }
        if size == 0 {
            return vec![vec![]];
        }

        // Allocate with capacity to avoid reallocations
        let mut result = Vec::with_capacity(self.max_subsets);
        let mut subset = Vec::with_capacity(size);

        // Use helper function for recursion with a limit
        self.generate_limited_subsets_helper(items, 0, size, &mut subset, &mut result);

        result
    }

    /// Helper function for generating subsets recursively with a limit
    fn generate_limited_subsets_helper<T: Clone>(
        &self,
        items: &Vec<T>,
        start: usize,
        size: usize,
        current: &mut Vec<T>,
        result: &mut Vec<Vec<T>>,
    ) {
        if current.len() == size {
            if result.len() < self.max_subsets {
                result.push(current.clone());
            }
            return;
        }

        // Optimization: early termination when we can't get enough items
        if items.len() - start < size - current.len() || result.len() >= self.max_subsets {
            return;
        }

        for i in start..items.len() {
            current.push(items[i].clone());
            self.generate_limited_subsets_helper(items, i + 1, size, current, result);
            current.pop();
        }
    }

    /// Partial correlation-based conditional independence test using Pearson correlation of residuals
    fn conditional_independence_test(
        &self,
        target: usize,
        candidate: Candidate,
        cond_set: &Vec<Candidate>,
        series_cache: &HashMap<(usize, usize), Array1<f64>>,
        n: usize,
    ) -> std::result::Result<(f64, f64), String> {
        // Get series
        let x_raw = series_cache
            .get(&(candidate.var, candidate.lag))
            .ok_or("Missing candidate series")?;
        let y_raw = series_cache
            .get(&(target, 0))
            .ok_or("Missing target series")?;

        // Standardize series
        let x = self.standardize(x_raw)?;
        let y = self.standardize(y_raw)?;

        // If no conditioning, compute direct Pearson correlation
        if cond_set.is_empty() {
            if n <= 2 {
                return Err(format!(
                    "Too few samples to compute t-statistic (n = {})",
                    n
                ));
            }

            let corr = self.pearson_correlation(&x, &y)?;

            if corr.abs() >= 1.0 {
                return Err(format!("Perfect correlation detected (corr = {})", corr));
            }

            let denom = (1.0 - corr * corr).sqrt();
            if denom == 0.0 {
                return Err("Denominator is zero in t-statistic calculation".to_string());
            }

            let t_stat = corr * (n as f64 - 2.0).sqrt() / denom;

            if !t_stat.is_finite() {
                return Err(format!("t_stat is not finite (t_stat = {})", t_stat));
            }

            let dist = statrs::distribution::StudentsT::new(0.0, 1.0, (n - 2) as f64)
                .map_err(|e| e.to_string())?;

            let p_val = 2.0 * (1.0 - dist.cdf(t_stat.abs()));
            return Ok((corr, p_val));
        }

        // Build design matrix Z for conditioning set
        let z_cols = cond_set.len();
        let mut z = Array2::<f64>::zeros((n, z_cols));
        for (i, cond) in cond_set.iter().enumerate() {
            let cond_series_raw = series_cache
                .get(&(cond.var, cond.lag))
                .ok_or("Missing conditioning series")?;
            let cond_series = self.standardize(cond_series_raw)?;
            z.column_mut(i).assign(&cond_series.view());
        }

        // Regress Z out of X and Y
        let r_x = self.regress_out(&z, &x)?;
        let r_y = self.regress_out(&z, &y)?;

        // Compute correlation between residuals
        let corr = self.pearson_correlation(&r_x, &r_y)?;

        // Effect ≈ partial correlation (as a proxy)
        let effect_size = corr;

        // Convert to p-value using Student's t distribution
        let dof = n as f64 - z_cols as f64 - 2.0;
        if dof <= 0.0 {
            return Err("Degrees of freedom <= 0".to_string());
        }

        let t_stat = corr * dof.sqrt() / (1.0 - corr * corr).sqrt();
        let dist =
            statrs::distribution::StudentsT::new(0.0, 1.0, dof).map_err(|e| e.to_string())?;
        let p_val = 2.0 * (1.0 - dist.cdf(t_stat.abs()));

        Ok((effect_size, p_val))
    }

    /// Standardizes a 1D array by subtracting the mean and dividing by the standard deviation.
    fn standardize(&self, series: &Array1<f64>) -> std::result::Result<Array1<f64>, String> {
        let mean = series.mean().ok_or("Cannot compute mean of empty array")?;
        let std = series.std(0.0);
        if std == 0.0 {
            return Err("Standard deviation is zero (constant series)".to_string());
        }
        Ok(series.mapv(|x| (x - mean) / std))
    }

    /// Computes the Pearson correlation coefficient between two series
    fn pearson_correlation(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
    ) -> std::result::Result<f64, String> {
        let n = x.len();
        if n != y.len() || n == 0 {
            return Err("Invalid input lengths.".to_string());
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_yy = 0.0;
        let mut sum_xy = 0.0;

        for i in 0..n {
            let xi = x[i];
            let yi = y[i];
            sum_x += xi;
            sum_y += yi;
            sum_xx += xi * xi;
            sum_yy += yi * yi;
            sum_xy += xi * yi;
        }

        let mean_x = sum_x / n as f64;
        let mean_y = sum_y / n as f64;

        let cov_xy = sum_xy / n as f64 - mean_x * mean_y;
        let var_x = sum_xx / n as f64 - mean_x.powi(2);
        let var_y = sum_yy / n as f64 - mean_y.powi(2);

        Ok(cov_xy / (var_x.sqrt() * var_y.sqrt()))
    }

    fn regress_out(
        &self,
        z: &Array2<f64>,
        v: &Array1<f64>,
    ) -> std::result::Result<Array1<f64>, String> {
        let xtx = z.t().dot(z);
        let xty = z.t().dot(v);
        let beta = xtx
            .solve(&xty)
            .map_err(|e| format!("Regression error: {:?}", e))?;
        let fitted = z.dot(&beta);
        Ok(v - &fitted)
    }

    /// Run the MCI phase of PCMCI
    fn run_mci_phase(
        &mut self,
        ts_cache: &TimeSeriesCache,
        n: usize,
    ) -> std::result::Result<(), String> {
        let n_vars = self.data.dim().1;

        // Calculate MCI for each target and each potential parent in parallel
        let mci_results: Vec<CausalLink> = (0..n_vars)
            .into_par_iter()
            .flat_map(|target| {
                let binding = vec![];
                let target_parents = self.parents.get(&target).unwrap_or(&binding);

                // For each target, test each parent
                target_parents
                    .iter()
                    .filter_map(|&parent_cand| {
                        // For MCI, we condition on parents of both source and target
                        let source_var = parent_cand.var;
                        let source_lag = parent_cand.lag;

                        // Get source parents, avoiding feedback loops
                        let source_parents = self
                            .parents
                            .get(&source_var)
                            .unwrap_or(&vec![])
                            .iter()
                            .filter(|&&p| p.var != target || p.lag > 0)
                            .cloned()
                            .collect::<Vec<_>>();

                        // Get other target parents
                        let other_target_parents = target_parents
                            .iter()
                            .filter(|&&p| p != parent_cand)
                            .cloned()
                            .collect::<Vec<_>>();

                        // Combine conditioning sets for MCI test
                        let mut cond_set = other_target_parents;

                        // Add time-shifted source parents
                        for source_parent in source_parents {
                            let adjusted_lag = source_parent.lag + source_lag;
                            if adjusted_lag <= self.max_lag {
                                cond_set.push(Candidate {
                                    var: source_parent.var,
                                    lag: adjusted_lag,
                                });
                            }
                        }

                        // Run MCI test
                        match self.conditional_independence_test(
                            target,
                            parent_cand,
                            &cond_set,
                            &ts_cache.cache,
                            n,
                        ) {
                            Ok((effect_size, p_value)) => Some(CausalLink {
                                source: source_var,
                                target,
                                lag: source_lag,
                                effect_size,
                                p_value,
                            }),
                            Err(_) => None,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        self.mci_links = mci_results;
        Ok(())
    }

    pub fn run_pcmci(&mut self) -> std::result::Result<(Array3<f64>, Array3<f64>), String> {
        // Run algorithm directly on the ndarray data
        let n_time = self.data.dim().0;
        let n_vars = self.data.dim().1;

        if n_time <= self.max_lag {
            return Err("Not enough time steps for the specified max_lag.".to_string());
        }

        let mut p_matrix = Array3::<f64>::from_elem((n_vars, n_vars, self.max_lag + 1), f64::NAN);
        let mut val_matrix = Array3::<f64>::from_elem((n_vars, n_vars, self.max_lag + 1), f64::NAN);

        // Create time series cache
        let ts_cache = self
            .build_time_series_cache()
            .expect("Failed to build cache");
        let n = ts_cache.n_samples;

        // Run PC phase in parallel
        let parents_per_target: Vec<(usize, Vec<Candidate>)> = (0..n_vars)
            .into_par_iter()
            .map(|target| self.run_pc_for_target(target, n_vars, &ts_cache, n))
            .collect();

        // Store parents for each target
        for (target, parents) in parents_per_target {
            self.parents.insert(target, parents);
        }

        // Convert parents to links for compatibility
        self.causal_links = self
            .parents
            .iter()
            .flat_map(|(&target, parents)| {
                parents.iter().map(move |cand| Link {
                    source: cand.var,
                    target,
                })
            })
            .collect();

        // Run the MCI phase
        self.run_mci_phase(&ts_cache, n).expect("MCI phase failed");

        for link in &self.mci_links {
            if link.lag <= self.max_lag {
                p_matrix[[link.source, link.target, link.lag]] = link.p_value;
                val_matrix[[link.source, link.target, link.lag]] = link.effect_size;
            }
        }

        return Ok((p_matrix, val_matrix));
    }
}
