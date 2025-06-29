import pandas as pd
import argparse
import os

def load_expression_data(filepath, chunksize_genes=500):
    """
    Loads the UMI count data using chunking to handle large files.
    - Assumes tab-delimited format.
    - Skips initial rows that are not part of the gene expression matrix
      (e.g., 'Cell_line', 'Pool_ID' rows in the provided snippet).
    - Sets the first column (gene names) as index.
    """
    chunk_list = []

    try:
        print(f"Starting to read expression data: {filepath} (chunksize={chunksize_genes} genes per chunk)...")
        # The first row is the header (cell IDs), the first column is the index (gene IDs)
        reader = pd.read_csv(filepath, sep='\\t', header=0, index_col=0, chunksize=chunksize_genes)

        processed_chunks_count = 0
        total_genes_processed_estimate = 0

        for i, chunk in enumerate(reader):
            # Filter out known non-gene rows based on the snippet
            non_gene_rows = ['Cell_line', 'Pool_ID']
            chunk = chunk[~chunk.index.isin(non_gene_rows)]

            # Ensure data is numeric, coercing errors to NaN
            chunk = chunk.apply(pd.to_numeric, errors='coerce')

            # Drop rows that might have become all NaN after coercion
            chunk.dropna(how='all', axis=0, inplace=True)

            if not chunk.empty:
                chunk_list.append(chunk)
                processed_chunks_count += 1
                total_genes_processed_estimate += len(chunk)
                if processed_chunks_count % 5 == 0: # Print progress every 5 processed chunks
                    print(f"  Processed chunk {i+1}, approx. {total_genes_processed_estimate} valid genes so far...")
            else:
                print(f"  Chunk {i+1} was empty after filtering non-gene rows and NaNs.")

        if not chunk_list:
            print("ERROR: No data left after filtering non-gene rows from all chunks.")
            return None

        print(f"Finished reading file. Concatenating {len(chunk_list)} processed chunks.")
        # Concatenate along rows (axis=0) as chunks are gene-row based
        full_gene_df = pd.concat(chunk_list, axis=0)
        print(f"Concatenation complete. Shape of gene matrix (genes x cells): {full_gene_df.shape}")

        return full_gene_df

    except FileNotFoundError:
        print(f"ERROR: Expression data file not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the expression data: {e}")
        return None

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

def main():
    parser = argparse.ArgumentParser(description="Process single-cell RNA-seq data to create a cell-by-gene matrix with cell line annotations.")
    parser.add_argument("--expression_file", type=str, required=True, help="Path to the UMI count data file (genes x cells).")
    parser.add_argument("--metadata_file", type=str, required=True, help="Path to the metadata file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the final processed DataFrame (e.g., output.csv or output.parquet).")
    parser.add_argument("--chunksize_genes", type=int, default=1000, help="Number of genes to process per chunk for the expression file.")

    args = parser.parse_args()

    print("Starting data processing script...")
    print(f"  Expression file: {args.expression_file}")
    print(f"  Metadata file: {args.metadata_file}")
    print(f"  Output file: {args.output_file}")
    print(f"  Chunksize (genes): {args.chunksize_genes}")

    umi_df = load_expression_data(args.expression_file, args.chunksize_genes)

    if umi_df is None:
        print("Failed to load expression data. Exiting.")
        return

    print(f"Successfully loaded expression data. Original shape (genes x cells): {umi_df.shape}")

    # Transpose UMI data: cells as rows, genes as columns
    print("Transposing expression data...")
    umi_df_transposed = umi_df.transpose()
    print(f"Transposed expression data shape (cells x genes): {umi_df_transposed.shape}")

    meta_df = load_metadata(args.metadata_file)

    if meta_df is None:
        print("Failed to load metadata. Exiting.")
        return

    print("First 5 rows of loaded metadata (index='NAME', column='Cell_line'):")
    print(meta_df.head())

    # Merge data
    print("\nMerging transposed expression data with metadata (Cell_line)...")
    final_df = umi_df_transposed.merge(meta_df[['Cell_line']],
                                       left_index=True,
                                       right_index=True,
                                       how='left') # Use left merge to keep all cells from expression data

    print("\nMerge complete.")
    print(f"Shape of the final DataFrame (cells x (genes + Cell_line)): {final_df.shape}")

    # Verification: Check if 'Cell_line' column has NAs
    if final_df['Cell_line'].isnull().any():
        print(f"\nWARNING: {final_df['Cell_line'].isnull().sum()} cells did not have corresponding 'Cell_line' information in the metadata.")
    else:
        print("\nSuccessfully added 'Cell_line' information for all cells found in expression data.")

    # Save the final DataFrame
    try:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir): # Create output directory if it doesn't exist
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        print(f"\nSaving final DataFrame to: {args.output_file}")
        if args.output_file.endswith(".csv"):
            final_df.to_csv(args.output_file, index_label="Cell_ID")
        elif args.output_file.endswith(".parquet"):
            final_df.to_parquet(args.output_file, index=True)
        else:
            print("WARNING: Output file format not recognized (use .csv or .parquet). Saving as CSV by default.")
            # Ensure filename ends with .csv if not specified as parquet
            output_csv_path = args.output_file if args.output_file.endswith(".csv") else args.output_file + ".csv"
            final_df.to_csv(output_csv_path, index_label="Cell_ID")
            if not args.output_file.endswith(".csv"):
                 print(f"Saved to {output_csv_path}")

        print("Successfully saved the final DataFrame.")
    except Exception as e:
        print(f"ERROR: Could not save the final DataFrame to {args.output_file}: {e}")

    print("\nScript finished.")

if __name__ == "__main__":
    main()
