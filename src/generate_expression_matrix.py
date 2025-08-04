"""
This script generates an AnnData (.h5ad) expression matrix from raw single-cell gene expression data and associated metadata.
It loads metadata (with cell line information), processes feature files to ensure compatibility with Scanpy, and constructs an AnnData object.
The resulting matrix can be used for downstream single-cell analysis, including integration with mutation status and other annotations.
"""

import pandas as pd
import argparse
import os
import scanpy as sc
import sys
import anndata as ad
import tempfile
import glob
import numpy as np
from scipy import sparse

def load_metadata(filepath):
    """
    Loads the metadata file.
    - Assumes tab-delimited format.
    - Uses the 'NAME' column for cell identifiers.
    """
    try:
        print(f"Loading metadata from: {filepath}")
        meta_df = pd.read_csv(filepath, sep='\\t')

        if 'NAME' not in meta_df.columns:
            print("ERROR: 'NAME' column not found in metadata. Check file structure.")
            return None

        # Check for and skip the 'TYPE' row if it's read as data and is the second row
        if len(meta_df) > 1 and meta_df.iloc[0,0] == 'TYPE':
             meta_df = pd.read_csv(filepath, sep='\\t', header=0, skiprows=[1])

        meta_df = meta_df[['NAME', 'Cell_line']].copy()
        meta_df.set_index('NAME', inplace=True)
        print(f"Metadata loaded. Shape: {meta_df.shape}")
        return meta_df
    except FileNotFoundError:
        print(f"ERROR: Metadata file not found at {filepath}")
        return None
    except KeyError:
        print("ERROR: 'NAME' or 'Cell_line' column not found in metadata. Please check column names.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the metadata: {e}")
        return None



def one_to_three_columns_features_file(file_path):
    """
    scanpy.read_10x_mtx requires the features.tsv file to have 3 columns. 
    This function creates two new useless columns to ensure the function does not return an error.
    """
    features = pd.read_csv(file_path, header=None, sep="\t")

    assert features.shape[1] == 1

    features[1] = features[0] 
    features[2] = "Gene Expression"

    features.to_csv(file_path, header=False, index=False, sep="\t")


def load_expression_data(file_path, verbosity=False): 
    """
    Loads expression data from a 10X Genomics file into an AnnData object.
    If verbosity is True it prints some info about the dataframe encaptured in the AnnData object.
    The expected format is a folder containing 3 files: matrix.mtx, barcodes.tsv, and features.tsv.
    IMPORTANT: THE FILES MUST BE COMPRESSED WITH GZIP, OTHERWISE scanpy.read_10x_mtx() WILL NOT WORK.
    """
    features_path = os.path.join(file_path, "features.tsv.gz")
    #print("DEBUG: Reading features from:", features_path)
    features = pd.read_csv(features_path, header=None, sep="\t", compression="gzip")
    #print("DEBUG: Features loaded, shape:", features.shape)
    assert features.shape[1] == 3, f"features.tsv must have 3 columns, but has {features.shape[1]} columns"
    #print("DEBUG: Passed features shape assertion")
   
    adata = sc.read_10x_mtx(file_path,
    var_names="gene_ids",
    cache=True)
    df_expression=ad.AnnData.to_df(adata)

    if verbosity:
        print(f"df_expression shape: {df_expression.shape}")
        print(f"df_expression columns: {df_expression.columns}")
        print(f"df_expression head: {df_expression.head()}")

    return adata


def split_kinker_to_blocks(input_file, output_dir, block_size=1000):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file) as f:
        header = f.readline().strip().split('\t')
    all_cells = header[1:]
    with open(input_file) as f:
        f.readline()
        cell_line_row = f.readline().strip().split('\t')
    cell_line_per_cell = cell_line_row[1:]
    cell_id_to_line = dict(zip(all_cells, cell_line_per_cell))
    for i in range(0, len(all_cells), block_size):
        block_cells = all_cells[i:i+block_size]
        usecols = [0] + [j+1 for j in range(i, min(i+block_size, len(all_cells)))]
        out_path = f"{output_dir}/block_{i//block_size:04d}.csv"
        if os.path.exists(out_path):
            print(f"[SKIP] {out_path} already exists. Skipping block {i//block_size:04d}.")
            continue
        df = pd.read_csv(input_file, sep='\t', usecols=usecols, index_col=0, skiprows=[1,2,3])
        non_gene_rows = ['Cell_line', 'Pool_ID', 'TYPE', 'group']
        df = df[~df.index.isin(non_gene_rows)]
        df = df.apply(pd.to_numeric, errors='coerce').dropna(how='any', axis=0)
        df_t = df.T
        df_t['Cell_line'] = [cell_id_to_line.get(cell, "UNKNOWN") for cell in df_t.index]
        cols = ['Cell_line'] + [col for col in df_t.columns if col != 'Cell_line']
        df_t = df_t[cols]
        df_t.to_csv(out_path)
        print(f"Saved {out_path}: {df_t.shape[0]} cells x {df_t.shape[1]-1} genes (+Cell_line)")


def merge_kinker_blocks_to_h5ad(block_dir, output_h5ad):
    block_files = sorted(glob.glob(os.path.join(block_dir, "block_*.csv")))
    if not block_files:
        raise RuntimeError(f"No block files found in {block_dir}")
    adatas = []
    for i, block_file in enumerate(block_files):
        print(f"Loading {block_file} ({i+1}/{len(block_files)})...")
        df = pd.read_csv(block_file, index_col=0)
        obs = pd.DataFrame(index=df.index)
        obs['Cell_line'] = df['Cell_line']
        X = df.drop(columns=['Cell_line'])
        X_sparse = sparse.csr_matrix(X.values)
        adata = ad.AnnData(X_sparse, obs=obs, var=pd.DataFrame(index=X.columns))
        adatas.append(adata)
    print(f"Concatenating {len(adatas)} blocks...")
    adata_merged = ad.concat(adatas, axis=0, join='outer', merge='same')
    print(f"Saving merged AnnData to {output_h5ad} ...")
    adata_merged.write(output_h5ad)
    print("Done!")
    return adata_merged


def main_kinker():
    import argparse
    parser = argparse.ArgumentParser(description="Process Kinker dataset (block-wise AnnData version)")
    parser.add_argument("--raw_file", type=str, default="data/Kinker/UMIcount_data.txt", help="Path to raw UMI count data file")
    parser.add_argument("--block_dir", type=str, default="output/blocks", help="Directory for block CSVs")
    parser.add_argument("--block_size", type=int, default=1000, help="Number of cells per block")
    parser.add_argument("--output_h5ad", type=str, default="output/expression_matrix_kinker_blocks_merged.h5ad", help="Output AnnData file name")
    parser.add_argument("--output_csv", type=str, default=None, help="(Optional) Output CSV file name for merged matrix")
    args = parser.parse_args()


    print("Splitting raw file into blocks (if needed)...")
    split_kinker_to_blocks(args.raw_file, args.block_dir, block_size=args.block_size)

    print("Merging blocks into AnnData .h5ad file...")
    adata_merged = merge_kinker_blocks_to_h5ad(args.block_dir, args.output_h5ad)


def main_gambardella():
    parser = argparse.ArgumentParser(description="Process 10x Genomics dataset using Scanpy.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz")
    parser.add_argument("--output_file", type=str, default=None, help="(Optional) Filename for the output CSV (e.g. matrix.csv)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory where the output file will be saved")
    parser.add_argument("--output_h5ad", type=str, default="output/expression_matrix_gambardella.h5ad", help="Output AnnData file name")
    parser.add_argument("--verbosity", action="store_true", help="Print expression matrix info")
    args = parser.parse_args()

    print("Eseguendo main_gambardella()")

    adata = load_expression_data(args.input_dir, verbosity=args.verbosity)
    df_expression = adata.to_df()
    df_expression['Cell_line'] = df_expression.index.str.split('_').str[0]

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_file:
        output_path = os.path.join(args.output_dir, args.output_file)
        # print(f"Saving output to: {output_path}")
        # df_expression.to_csv(output_path)

    output_h5ad_path = args.output_h5ad
    print(f"Saving AnnData to: {output_h5ad_path}")
    adata.obs['Cell_line'] = df_expression['Cell_line']
    adata.write(output_h5ad_path)
    print("Processo completato con successo!")


if __name__ == "__main__":
    
    if "--data" not in sys.argv:
        raise ValueError("You must specify --data <kinker|gambardella>")

    idx = sys.argv.index("--data")
    data_type = sys.argv[idx + 1]

    new_argv = sys.argv[:idx] + sys.argv[idx+2:]
    sys.argv = new_argv

    if data_type == "kinker":
        main_kinker()
    elif data_type == "gambardella":
        main_gambardella()
    else:
        raise ValueError(f"Unknown dataset type: {data_type}")