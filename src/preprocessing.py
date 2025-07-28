"""
Preprocessing utilities for single-cell RNA-seq data.
"""
import scanpy as sc
import anndata as ad
import mygene
import pandas as pd
import numpy as np
from collections import defaultdict

def check_sparsity(adata):
    """Print sparsity info for AnnData object."""
    print(f"Number of cells: {adata.shape[0]}")
    print(f"Number of genes: {adata.shape[1]}")
    print(f"Number of non-zero entries: {adata.X.nnz}")
    print(f"Sparsity: {1 - (adata.X.nnz / (adata.shape[0] * adata.shape[1])):.2%}")

def show_qc_plots(adata, violin_cols=None, scatter_x=None, scatter_y=None):
    """Show QC plots for AnnData object."""
    sc.pp.calculate_qc_metrics(
    adata,inplace=True, log1p=True)
    sc.pl.violin(adata,violin_cols,jitter=0.4,multi_panel=True)
    sc.pl.scatter(adata, scatter_x, scatter_y)

def digitize(x, bins, side="both"):
    """Digitize values into bins (with random tie-breaking)."""
    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits
    right_digits = np.digitize(x, bins, right=True)
    rands = np.random.rand(len(x))
    digits = rands * (right_digits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits

def bin_data(x, n_bins):
    """Bin RNA-seq data at patient level."""
    binned_rows = []
    for row in x:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
        binned_rows.append(binned_row)
    binned_data = np.stack(binned_rows)
    return binned_data

def cdf_data(x):
    """Get CDF values for each row."""
    cdf_rows = []
    for row in x:
        inp_row = row[row != 0].copy()
        cdf_dict = {0.:0.}
        values, counts = np.unique(inp_row, return_counts=True)
        cdf = np.cumsum(counts) / len(inp_row)
        cdf_dict.update({values[i]:cdf[i] for i in range(len(values))})
        cdf_rows.append(np.vectorize(cdf_dict.get)(row))
    cdf_data = np.stack(cdf_rows)
    return cdf_data

def rank_data(x):
    """Get rank of each entry in each row (random for ties)."""
    rank_rows = []
    for row in x:
        inp_row = row[row != 0].copy()
        ranks = inp_row.argsort().argsort() + 1
        rank_dict = dict()
        for val in np.unique(inp_row):
            val_indices = np.where(inp_row == val)[0]
            val_ranks = ranks[val_indices]
            np.random.shuffle(val_ranks)
            rank_dict[val] = val_ranks.tolist()
        rank_row = np.zeros_like(row, dtype=float)
        n = len(inp_row)
        for i, val in enumerate(row):
            rank_row[i] = 0.
        rank_rows.append(rank_row)
    rank_data = np.stack(rank_rows)
    return rank_data

def avg_rank_data(x):
    """Get average rank of each entry in each row."""
    avg_rank_rows = []
    for row in x:
        inp_row = row[row != 0].copy()
        ranks_dict = defaultdict(list)
        ranks_dict[0.].append(0.)
        ranks = inp_row.argsort().argsort() + 1
        for i in range(len(inp_row)):
            ranks_dict[inp_row[i]].append(ranks[i])
        n = len(inp_row)
        avg_rank_dict = {rank:np.mean(ranks_dict[rank]) / n for rank in ranks_dict.keys()}
        avg_rank_rows.append(np.vectorize(avg_rank_dict.get)(row))
    avg_rank_data = np.stack(avg_rank_rows)
    return avg_rank_data

def main(feature_selection="target", batch_correction=None, local_testing=False):  
    """Preprocess and save single-cell data for analysis."""
    # Load merged AnnData
    if local_testing:
        adata = ad.read_h5ad("output/local_testing_merged_gambardella_kinker_common_genes.h5ad")
    else:
        adata = ad.read_h5ad("output/merged_gambardella_kinker_common_genes.h5ad")
    adata.layers["raw_counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Feature selection
    if feature_selection == "HVG":
        adata.layers["pre_feature_selection"] = adata.X.copy()
        sc.pp.highly_variable_genes(adata, min_mean=0.1, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable]
    elif feature_selection == "target":
        adata.layers["pre_feature_selection"] = adata.X.copy()
        # Load TP53 target genes
        target_genes_df = pd.read_excel('data/TP53target/41388_2017_BFonc2016502_MOESM5_ESM_tab1.xlsx')
        target_genes = set(target_genes_df["Gene Symbol"].astype(str).str.upper())
        selected_genes = [gene for gene in adata.var_names if gene.upper() in target_genes]
        adata = adata[:, selected_genes]
    else:
        raise KeyError("feature_selection can only be values from this list ['HVG', 'target']")

    # Batch correction
    if batch_correction == "harmony":
        adata.layers["pre_harmony"] = adata.X.copy()
        if local_testing:
            sc.pp.pca(adata, n_comps=25)
        else:
            sc.pp.pca(adata, n_comps=400)
        sc.external.pp.harmony_integrate(adata, key="Cell_line")
    elif batch_correction == "combat":
        adata.layers["pre_combat"] = adata.X.copy()
        sc.pp.combat(adata, key="Cell_line")
    elif batch_correction is None:
        pass
    else:
        raise KeyError("batch_correction can only be values from this list [None, 'combat', 'harmony']")

    # Save to CSV
    final_df = ad.AnnData.to_df(adata)
    # Use TP53_status if available, else fallback to mutation_status
    if "TP53_status" in adata.obs.columns:
        final_df["mutation_status"] = adata.obs["TP53_status"].values
    elif "mutation_status" in adata.obs.columns:
        final_df["mutation_status"] = adata.obs["mutation_status"].values
    else:
        raise KeyError("No mutation status column found in AnnData.obs")
    suffix = f"{feature_selection}_{batch_correction}" if batch_correction else f"{feature_selection}"
    final_df.to_csv(f"output/final_preprocessed_data_{suffix}.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess single-cell RNA-seq data for multiclass TP53 analysis.")
    parser.add_argument('--feature_selection', type=str, choices=['HVG', 'target'], required=True,
                        help='Feature selection method: HVG or target')
    parser.add_argument('--batch_correction', type=str, choices=['combat', 'harmony', 'none'], default='none',
                        help='Batch correction method: combat, harmony, or none')
    parser.add_argument('--local_testing', type=bool, choices=[True, False], default=False,
                        help='Set true for local testing')
    args = parser.parse_args()

    batch_correction = None if args.batch_correction == 'none' else args.batch_correction
    main(feature_selection=args.feature_selection, batch_correction=batch_correction,local_testing=args.local_testing)