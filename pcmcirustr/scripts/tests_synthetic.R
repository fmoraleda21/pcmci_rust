library(rextendr)

# Create R package from Rust code
rextendr::document("pcmcirustr")

# Function to generate synthetic data
generate_synthetic_data <- function(n_time, n_vars, max_lag, causal_structure, noise_level = 0.05, seed = 42) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  burn_in <- 100
  total_time <- n_time + burn_in

  data <- matrix(0, nrow = total_time, ncol = n_vars)

  data[1:max_lag, ] <- matrix(runif(max_lag * n_vars), nrow = max_lag, ncol = n_vars)

  for (t in max_lag:(total_time - 1)) {
    for (target_var in 0:(n_vars - 1)) {
      value <- 0
      if (!is.null(causal_structure[[as.character(target_var)]])) {
        for (causal in causal_structure[[as.character(target_var)]]) {
          source_var <- causal[[1]]
          lag <- causal[[2]]
          coeff <- causal[[3]]

          value <- value + coeff * data[t - lag + 1, source_var + 1]
        }
      }
      value <- value + rnorm(1, mean = 0, sd = noise_level)
      data[t + 1, target_var + 1] <- value
    }
  }

  data[(burn_in + 1):total_time, ]
}

# Define causal structure and generate synthetic data
causal_structure <- list(
  "0" = list(list(1, 1, 0.3), list(2, 1, 0.4), list(3, 2, 0.5)),
  "1" = list(list(2, 1, 0.2)),
  "2" = list(list(3, 1, 0.1))
)

max_lag <- 2
n_vars <- 20
alpha <- 0.05
max_condition_set_size <- 3
max_subsets <- 1

data <- generate_synthetic_data(
  n_time = 100000,
  n_vars = n_vars,
  max_lag = max_lag,
  causal_structure = causal_structure,
  noise_level = 0.05
)

# Load built library
devtools::load_all("pcmcirustr")
# devtools::install("pcmcirustr")
library(pcmcirustr)

# Run PCMCI
time_taken <- system.time({
  result <- run_pcmci(
    data_array = data,
    max_lag = max_lag,
    alpha = alpha,
    max_condition_set_size = max_condition_set_size,
    max_subsets = max_subsets
  )
})
print(time_taken) # user + system > elapsed -> multicore parallelization

# Check results
dim(result$p_matrix)

# var1(t-1) -> var0, due to Python (0-indexed) implementation:
result$val_matrix[1 + 1, 0 + 1, 1 + 1]

get_val_matrix <- function(result, i, j, k) {
  return(result$val_matrix[i + 1, j + 1, k + 1])
}

# var1(t-1) -> var0
get_val_matrix(result, 1, 0, 1)
# var2(t-1) -> var0
get_val_matrix(result, 2, 0, 1)
# var3(t-2) -> var0
get_val_matrix(result, 3, 0, 2)
# var2(t-1) -> var1
get_val_matrix(result, 2, 1, 1)
# var3(t-1) -> var2
get_val_matrix(result, 3, 2, 1)
