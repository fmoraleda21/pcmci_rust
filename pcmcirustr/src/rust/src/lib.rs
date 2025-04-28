use numpy::ndarray::ArrayView2;
use std::collections::HashMap;

use extendr_api::prelude::*;

extern crate pcmcirust;

#[extendr]
fn run_pcmci(
    data_array: ArrayView2<f64>,
    max_lag: usize,
    alpha: f64,
    max_condition_set_size: usize,
    max_subsets: usize,
) -> std::result::Result<Robj, String> {
    // Create and initialize PCMCI instance
    let mut pcmci = pcmcirust::PCMCI {
        data: data_array,
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
            // Copy results onto R memory
            let p_matrix_r: Robj = p_matrix.try_into().unwrap();
            let val_matrix_r: Robj = val_matrix.try_into().unwrap();
            return Ok(list!(p_matrix = p_matrix_r, val_matrix = val_matrix_r).into_robj());
        }
        Err(err) => {
            // handle the error
            return Err(err);
        }
    }
}

extendr_module! {
    mod pcmcirustr;
    fn run_pcmci;
}
