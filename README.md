# 🧩 PCMCI in Rust: High-performance Causal Discovery in Time Series Data

This repository provides an efficient and parallelized implementation of the PCMCI algorithm in **Rust**, with seamless bindings for **Python** and **R**. It aims to follow the behavior of the original implementation found in the [`tigramite`](https://github.com/jakobrunge/tigramite) library, but with improved performance powered by Rust.

---

## 📂 Repository Structure

```
pcmci-rust/
 ├── pcmcirust/               # Core Rust implementation (safe, parallel, zero-copy)
 ├── pcmcirustpy/             # Python bindings (via PyO3 and maturin)
 │    ├── notebooks/          # Jupyter notebooks with usage examples & benchmarks vs tigramite
 │    ├── requirements.txt    # Python dependencies for development & testing
 ├── pcmcirustr/              # R bindings (via rextendr)
 │    ├── scripts/            # R scripts with usage examples
```

---

## ⚡ Features

✅ **Safe Rust core**: Written in Rust with modern concurrency (Rayon).  

✅ **Zero-copy** bindings: Efficient memory sharing with `numpy` and R structures.

✅ **Parallelized**: Core phases run in parallel over variables and candidates.

✅ **Flexible**: Similar API parameters and outputs as `tigramite` for easy migration.

---

## 🐍 Python Installation & Usage

```bash
cd pcmcirustpy
```

1️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```

2️⃣ Build and install the Python package:  
```bash
maturin develop --release
```

3️⃣ Use in Python:  
```python
import pcmcirustpy

# Example usage:
result = pcmcirustpy.run_pcmci(data_array, max_lag, alpha, ...)
```

👉 Example notebooks: see `pcmcirustpy/notebooks/` for reproducible demos and benchmarks.

---

## 📊 R Installation & Usage

```bash
cd pcmcirustr
```

1️⃣ Generate bindings:  
```r
rextendr::document("pcmcirustr")
```

2️⃣ Load the package locally:  
```r
devtools::load_all("pcmcirustr")

# Example usage:
result <- pcmcirustr::run_pcmci(
  data_array = data_array,
  max_lag = max_lag,
  alpha = alpha,
  ...
)
```

👉 Example scripts: see `pcmcirustr/scripts/` for usage demos in R.

---

## 📬 About

Developed by Francisco Moraleda Moreno as part of the Final Thesis of the Master's Degree in Big Data Science at University of Navarra.

This effort is also part of the Mercury project at BBVA AI Factory.
