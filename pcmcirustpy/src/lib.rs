use numpy::{IntoPyArray, PyReadonlyArray2};
use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

extern crate pcmcirust;

#[pyfunction]
fn run_pcmci(
    py: Python<'_>,
    data_array: PyReadonlyArray2<'_, f64>,
    max_lag: usize,
    alpha: f64,
    max_condition_set_size: usize,
    max_subsets: usize,
) -> PyResult<Py<PyDict>> {
    // Create and initialize PCMCI instance
    let mut pcmci = pcmcirust::PCMCI {
        data: data_array.as_array(),
        max_lag,
        alpha,
        max_condition_set_size,
        max_subsets,
        parents: HashMap::new(),
        causal_links: vec![],
        mci_links: vec![],
    };

    match pcmci.run_pcmci() {
        Ok((p_matrix, val_matrix)) => {
            // handle the results
            let out = PyDict::new(py);
            out.set_item("p_matrix", p_matrix.into_pyarray(py))?;
            out.set_item("val_matrix", val_matrix.into_pyarray(py))?;
            // Copy results onto Python memory
            return Ok(out.into());
        }
        Err(err) => {
            // handle the error
            return Err(PyValueError::new_err(err));
        }
    }
}

/// Python module definition
#[pymodule]
fn pcmcirustpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_pcmci, m)?)?;
    Ok(())
}
