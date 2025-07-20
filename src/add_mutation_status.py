import pandas as pd
import argparse
import os
import anndata as ad

def update_anndata_with_tp53_status(adata, master_mutation_dict, cell_lines_with_multiple_mutations_global, cell_line_name="Cell_line"):
    """
    Updates the AnnData object with TP53 mutation status in .obs.
    Filters out cells from lines with multiple mutations or missing info.
    """
    initial_cell_count = adata.n_obs
    obs = adata.obs.copy()
    # Identify cells from lines with multiple mutations
    is_multi_mut_line = obs[cell_line_name].isin(cell_lines_with_multiple_mutations_global)
    count_removed_multiple_mutations = is_multi_mut_line.sum()
    # Filter out these cells first
    obs_filtered = obs[~is_multi_mut_line].copy()
    # Add TP53_status column
    obs_filtered['TP53_status'] = obs_filtered[cell_line_name].map(master_mutation_dict)
    is_missing_info = obs_filtered['TP53_status'].isna()
    count_removed_missing_info = is_missing_info.sum()
    obs_final = obs_filtered.dropna(subset=['TP53_status'])
    final_cell_count = obs_final.shape[0]
    print(f"Initial number of cells: {initial_cell_count}")
    print(f"Number of cells removed (multiple TP53 mutations): {count_removed_multiple_mutations}")
    print(f"Number of cells removed (missing TP53 info): {count_removed_missing_info}")
    print(f"Total number of cells kept: {final_cell_count}")
    # Subset AnnData to filtered cells
    adata_final = adata[obs_final.index].copy()
    adata_final.obs = obs_final
    return adata_final

def process_tp53_mutations(mutation_files):
    """
    Process TP53 mutation files and return:
    - master_mutation_dict: barcodes with exactly one TP53 mutation and a single Variant_Classification
    - cell_lines_with_multiple_mutations_global: barcodes with multiple distinct Variant_Classification values
    Uses only the short name (before the first underscore) for mapping.
    """
    raw_mutation_dfs = []

    # Specify dtypes for mutation files (cast after reading to avoid linter/type checker issues)
    for mut_file in mutation_files:
        try:
            mut_df = pd.read_csv(mut_file, sep='\t', comment='#', low_memory=False)
            # Cast columns to correct types if they exist
            if 'Hugo_Symbol' in mut_df.columns:
                mut_df['Hugo_Symbol'] = mut_df['Hugo_Symbol'].astype('category')
            if 'Tumor_Sample_Barcode' in mut_df.columns:
                mut_df['Tumor_Sample_Barcode'] = mut_df['Tumor_Sample_Barcode'].astype(str)
            if 'Variant_Classification' in mut_df.columns:
                mut_df['Variant_Classification'] = mut_df['Variant_Classification'].astype('category')
            required_cols = {'Hugo_Symbol', 'Tumor_Sample_Barcode', 'Variant_Classification'}
            if not required_cols.issubset(mut_df.columns):
                print(f"Warning: File {mut_file} is missing columns: {required_cols - set(mut_df.columns)}. Skipping.")
                continue
            raw_mutation_dfs.append(mut_df)
        except Exception as e:
            print(f"Error loading {mut_file}: {e}. Skipping.")

    if not raw_mutation_dfs:
        return {}, {}

    combined_mut_df = pd.concat(raw_mutation_dfs, ignore_index=True)
    tp53_all_files = combined_mut_df[combined_mut_df['Hugo_Symbol'] == 'TP53']

    if tp53_all_files.empty:
        return {}, {}

    master_mutation_dict = {}
    cell_lines_with_multiple_mutations_global = {}

    # Use only the short name for grouping
    tumor_barcodes = tp53_all_files['Tumor_Sample_Barcode'].astype(str)
    tp53_all_files['Short_Sample_Barcode'] = tumor_barcodes.str.split('_').str[0]
    grouped = tp53_all_files.groupby('Short_Sample_Barcode')

    for short_barcode, group in grouped:
        variant_list = list(group['Variant_Classification'])
        unique_classes = list(set(variant_list))
        if len(unique_classes) == 1:
            master_mutation_dict[short_barcode] = unique_classes[0]
        else:
            cell_lines_with_multiple_mutations_global[short_barcode] = list(unique_classes)

    return master_mutation_dict, cell_lines_with_multiple_mutations_global

def parse_arguments():
    parser = argparse.ArgumentParser(description='Add TP53 mutation status to AnnData (.h5ad) expression matrix')
    parser.add_argument('--dataset', type=str, required=True, choices=['gambardella', 'kinker'], help='Dataset name (gambardella or kinker)')
    parser.add_argument('--expression-h5ad', type=str, required=True, help='Path to input AnnData .h5ad file')
    parser.add_argument('--mutation-files', type=str, nargs='+', required=True, help='List of mutation files')
    parser.add_argument('--output-h5ad', type=str, required=True, help='Output AnnData .h5ad file path')
    parser.add_argument('--cell-line-column', type=str, required=True, help='Name of the cell line column in AnnData .obs')
    return parser.parse_args()


def main():
    args = parse_arguments()
    expression_h5ad_file = args.expression_h5ad
    mutation_files = args.mutation_files
    output_h5ad_file = args.output_h5ad
    cell_line_column = args.cell_line_column

    print(f"Dataset: {args.dataset}")
    print(f"Expression AnnData file: {expression_h5ad_file}")
    print(f"Mutation files: {mutation_files}")
    print(f"Output AnnData file: {output_h5ad_file}")
    print(f"Cell line column: {cell_line_column}")
    print("-" * 50)

    print("Processing mutation files to extract TP53 mutation status...")
    master_mutation_dict, cell_lines_with_multiple_mutations_global = process_tp53_mutations(mutation_files)

    print("Example keys from master_mutation_dict:")
    print(list(master_mutation_dict.keys())[:10])

    print(f"\nProcessed all mutation files:")
    print(f"  Master TP53 mutation dictionary (unique classification): {len(master_mutation_dict)} entries")
    multi_mut_list_preview = list(cell_lines_with_multiple_mutations_global.keys())
    print(f"  Cell lines with multiple TP53 variant types: {len(cell_lines_with_multiple_mutations_global)} {multi_mut_list_preview[:5]}{'...' if len(multi_mut_list_preview) > 5 else ''}")

    print(f"\nLoading AnnData expression matrix: {expression_h5ad_file}")
    adata = ad.read_h5ad(expression_h5ad_file)
    print(f"AnnData loaded. Shape: {adata.shape}")

    # Normalize Cell_line to short name (before first underscore)
    adata.obs[cell_line_column] = adata.obs[cell_line_column].astype(str).str.split('_').str[0]

    print(f"Updating AnnData with TP53 mutation status...")
    adata_final = update_anndata_with_tp53_status(
        adata,
        master_mutation_dict,
        cell_lines_with_multiple_mutations_global,
        cell_line_column
    )
    print(f"Saving updated AnnData to: {output_h5ad_file}")
    adata_final.write(output_h5ad_file)
    print("Processing complete.")

if __name__ == '__main__':
    main()
