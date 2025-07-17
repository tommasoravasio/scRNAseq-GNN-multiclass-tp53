import pandas as pd
import argparse
import os
import scanpy as sc
import sys
import anndata as ad
import tempfile
import glob

#print("DEBUG: pandas version:", pd.__version__)

# def load_expression_data(filepath, chunksize_genes=500):
#     """
#     Loads the UMI count data using chunking to handle large files.
#     - Assumes tab-delimited format.
#     - Skips initial rows that are not part of the gene expression matrix
#       (e.g., 'Cell_line', 'Pool_ID' rows in the provided snippet).
#     - Sets the first column (gene names) as index.
#     """
#     chunk_list = []

#     try:
#         print(f"Starting to read expression data: {filepath} (chunksize={chunksize_genes} genes per chunk)...")
#         # The first row is the header (cell IDs), the first column is the index (gene IDs)
#         reader = pd.read_csv(filepath, sep='\\t', header=0, index_col=0, chunksize=chunksize_genes)

#         processed_chunks_count = 0
#         total_genes_processed_estimate = 0

#         for i, chunk in enumerate(reader):
#             # Filter out known non-gene rows based on the snippet
#             non_gene_rows = ['Cell_line', 'Pool_ID']
#             chunk = chunk[~chunk.index.isin(non_gene_rows)]

#             # Ensure data is numeric, coercing errors to NaN
#             chunk = chunk.apply(pd.to_numeric, errors='coerce')

#             # Drop rows that might have become all NaN after coercion
#             chunk.dropna(how='all', axis=0, inplace=True)

#             if not chunk.empty:
#                 chunk_list.append(chunk)
#                 processed_chunks_count += 1
#                 total_genes_processed_estimate += len(chunk)
#                 if processed_chunks_count % 5 == 0: # Print progress every 5 processed chunks
#                     print(f"  Processed chunk {i+1}, approx. {total_genes_processed_estimate} valid genes so far...")
#             else:
#                 print(f"  Chunk {i+1} was empty after filtering non-gene rows and NaNs.")

#         if not chunk_list:
#             print("ERROR: No data left after filtering non-gene rows from all chunks.")
#             return None

#         print(f"Finished reading file. Concatenating {len(chunk_list)} processed chunks.")
#         # Concatenate along rows (axis=0) as chunks are gene-row based
#         full_gene_df = pd.concat(chunk_list, axis=0)
#         print(f"Concatenation complete. Shape of gene matrix (genes x cells): {full_gene_df.shape}")

#         return full_gene_df

#     except FileNotFoundError:
#         print(f"ERROR: Expression data file not found at {filepath}")
#         return None
#     except Exception as e:
#         print(f"An error occurred while loading the expression data: {e}")
#         return None

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



def main_kinker():
    import tempfile
    import glob
    parser = argparse.ArgumentParser(description="Process single-cell RNA-seq data to create a cell-by-gene matrix with cell line annotations.")
    parser.add_argument("--expression_file", type=str, required=True, help="Path to the UMI count data file (genes x cells).")
    parser.add_argument("--metadata_file", type=str, required=True, help="Path to the metadata file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the final processed DataFrame (e.g., output.csv).")
    parser.add_argument("--chunksize_genes", type=int, default=1000, help="Number of genes to process per chunk for the expression file.")

    args = parser.parse_args()

    print("Starting data processing script...")
    print(f"  Expression file: {args.expression_file}")
    print(f"  Metadata file: {args.metadata_file}")
    print(f"  Output file: {args.output_file}")
    print(f"  Chunksize (genes): {args.chunksize_genes}")

    import pandas as pd  # Ensure pandas is imported before use
    try:
        if args.expression_file.endswith('.gz'):
            reader = pd.read_csv(args.expression_file, sep='\t', header=0, index_col=0, compression='gzip', chunksize=args.chunksize_genes)
        else:
            reader = pd.read_csv(args.expression_file, sep='\t', header=0, index_col=0, chunksize=args.chunksize_genes)

        meta_df = load_metadata(args.metadata_file)
        if meta_df is None:
            print("Failed to load metadata. Exiting.")
            return
        print("First 5 rows of loaded metadata (index='NAME', column='Cell_line'):")
        print(meta_df.head())

        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        first_chunk = True
        processed_chunks_count = 0
        total_genes_processed_estimate = 0
        for i, chunk in enumerate(reader):
            non_gene_rows = ['Cell_line', 'Pool_ID']
            chunk = chunk[~chunk.index.isin(non_gene_rows)]
            chunk = chunk.apply(pd.to_numeric, errors='coerce')
            chunk.dropna(how='all', axis=0, inplace=True)
            if not chunk.empty:
                chunk_transposed = chunk.transpose()
                # Merge with metadata
                expr_df = chunk_transposed
                cell_line_df = meta_df[['Cell_line']] if isinstance(meta_df, pd.DataFrame) else pd.DataFrame(meta_df['Cell_line'])
                final_df = expr_df.merge(cell_line_df, left_index=True, right_index=True, how='left')
                # Write to CSV (append mode, header only for first chunk)
                final_df.to_csv(args.output_file, mode='w' if first_chunk else 'a', header=first_chunk, index_label="Cell_ID")
                if first_chunk:
                    first_chunk = False
                processed_chunks_count += 1
                total_genes_processed_estimate += chunk.shape[0]
                if processed_chunks_count % 5 == 0:
                    print(f"  Processed chunk {i+1}, approx. {total_genes_processed_estimate} valid genes so far...")
            else:
                print(f"  Chunk {i+1} was empty after filtering non-gene rows and NaNs.")
        print(f"\nSuccessfully processed and saved all chunks to {args.output_file}.")
    except Exception as e:
        print(f"Failed to process expression data: {e}. Exiting.")
        return


def main_gambardella():
    parser = argparse.ArgumentParser(description="Process 10x Genomics dataset using Scanpy.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz")
    parser.add_argument("--output_file", type=str, required=True, help="Filename for the output CSV (e.g. matrix.csv)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory where the output file will be saved")
    parser.add_argument("--verbosity", action="store_true", help="Print expression matrix info")
    args = parser.parse_args()

    print("Eseguendo main_gambardella()")

    # Carica i dati con controllo automatico su features.tsv.gz
    #print("DEBUG: Calling load_expression_data")
    adata = load_expression_data(args.input_dir, verbosity=args.verbosity)
    #print("DEBUG: load_expression_data returned")

    # Converte in DataFrame
    df_expression = adata.to_df()

    # Estrai la cell line dal barcode (indice)
    df_expression['Cell_line'] = df_expression.index.str.split('_').str[0]

    # Costruisci path completo e crea la directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    # Salva CSV
    print(f"Saving output to: {output_path}")
    df_expression.to_csv(output_path)


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