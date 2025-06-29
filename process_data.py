import pandas as pd

def load_expression_data(filepath="data/GSE157220/UMIcount_data.txt"):
    """
    Loads the UMI count data.
    - Assumes tab-delimited format.
    - Skips initial rows that are not part of the gene expression matrix.
      (e.g., 'Cell_line', 'Pool_ID' rows in the provided snippet).
    - Sets the first column (gene names) as index.
    """
    # Based on the snippet, the actual gene data starts from the row 'A1BG'
    # The first row of the file contains cell IDs and should be the header.
    # We need to find which row contains the first gene.
    # A robust way is to read a few lines to identify the start of the matrix.
    
    # For now, let's assume 'A1BG' is representative and is the 4th row in the file,
    # meaning we skip the first 3 rows of data (header is row 0, then 'Cell_line', 'Pool_ID').
    # However, pandas skiprows works on data lines, not including header.
    # If the file looks like:
    # Header_row (cell IDs)
    # Cell_line_info_row
    # Pool_ID_info_row
    # Gene1_data_row
    # Gene2_data_row
    # ...
    # We want to use the first row as header and then skip the next two rows.
    # A common way to handle this is to read the header separately,
    # then read the rest of the data skipping appropriate lines.

    try:
        # Read the header (cell IDs)
        header = pd.read_csv(filepath, sep='\\t', nrows=0).columns
        
        # Read the data, skipping the metadata rows embedded in the UMI file
        # The snippet showed 'Cell_line' and 'Pool_ID' as rows before genes.
        # So, we expect the gene data to start at index 2 if we consider the data part
        # (after the header row).
        # Let's try to identify the first gene row by checking the first column.
        # This is a bit heuristic without seeing the full file.
        # A simpler approach for the user, assuming a fixed number of header/info rows:
        # If 'A1BG' is the first gene, and it's the 4th line in the file overall,
        # then it's the 3rd data line after the header.
        # So we would use header=0, and then figure out how many rows to skip.
        
        # Let's try reading with header on the first row, and first column as index.
        # The challenge is the 'Cell_line' and 'Pool_ID' rows which are not genes.
        df = pd.read_csv(filepath, sep='\\t', header=0, index_col=0)
        
        # Filter out known non-gene rows based on the snippet
        # This assumes 'Cell_line' and 'Pool_ID' are actual values in the index
        # that need to be removed.
        non_gene_rows = ['Cell_line', 'Pool_ID']
        df = df[~df.index.isin(non_gene_rows)]
        
        # Ensure data is numeric, coercing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        
        return df
    except FileNotFoundError:
        print(f"ERROR: Expression data file not found at {filepath}")
        print("Please ensure the file path is correct and the script is run from a location where it can access this path.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the expression data: {e}")
        return None

def load_metadata(filepath="data/GSE157220/Metadata.txt"):
    """
    Loads the metadata file.
    - Assumes tab-delimited format.
    - Uses the 'NAME' column for cell identifiers.
    """
    try:
        meta_df = pd.read_csv(filepath, sep='\\t')
        # The snippet shows 'TYPE' as a row that might need skipping if it's read as data.
        # However, standard CSV reading with header=0 should handle this if 'TYPE' is just a descriptor for columns.
        # If 'NAME' is not the first column, this needs adjustment or use index_col='NAME'.
        # Let's assume the first row is headers.
        if 'NAME' not in meta_df.columns:
            print("ERROR: 'NAME' column not found in metadata. Check file structure.")
            return None
        # Check for and skip the 'TYPE' row if it's read as data
        if meta_df.iloc[0,0] == 'TYPE': # Assuming 'TYPE' would be in the first cell of that row
            meta_df = pd.read_csv(filepath, sep='\\t', header=0, skiprows=[1])

        meta_df = meta_df[['NAME', 'Cell_line']].copy()
        meta_df.set_index('NAME', inplace=True)
        return meta_df
    except FileNotFoundError:
        print(f"ERROR: Metadata file not found at {filepath}")
        print("Please ensure the file path is correct and the script is run from a location where it can access this path.")
        return None
    except KeyError:
        print("ERROR: 'NAME' or 'Cell_line' column not found in metadata. Please check column names.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the metadata: {e}")
        return None

if __name__ == "__main__":
    print("Starting data processing...")

    # Define file paths (can be made configurable)
    expression_file = "data/GSE157220/UMIcount_data.txt"
    metadata_file = "data/GSE157220/Metadata.txt"

    # Load data
    print(f"Loading expression data from: {expression_file}")
    umi_df = load_expression_data(expression_file)
    
    if umi_df is not None:
        print(f"Loaded expression data. Shape: {umi_df.shape}")
        # Transpose UMI data: cells as rows, genes as columns
        umi_df_transposed = umi_df.transpose()
        print(f"Transposed expression data. Shape: {umi_df_transposed.shape}")

        print(f"\nLoading metadata from: {metadata_file}")
        meta_df = load_metadata(metadata_file)

        if meta_df is not None:
            print(f"Loaded metadata. Shape: {meta_df.shape}")
            print("First 5 rows of metadata (index and Cell_line):")
            print(meta_df.head())

            # Merge data
            print("\nMerging expression data with metadata (Cell_line)...")
            # Ensure indices are aligned for merging. The cell IDs should be the index.
            final_df = umi_df_transposed.merge(meta_df[['Cell_line']], 
                                               left_index=True, 
                                               right_index=True, 
                                               how='left')
            
            print("\nMerge complete.")
            print("Shape of the final DataFrame:", final_df.shape)
            print("\nFirst 5 rows of the final DataFrame:")
            print(final_df.head())

            # Verification: Check if 'Cell_line' column has NAs
            if final_df['Cell_line'].isnull().any():
                print("\nWARNING: Some cells did not have corresponding 'Cell_line' information in the metadata.")
                print("Number of cells with missing Cell_line:", final_df['Cell_line'].isnull().sum())
            else:
                print("\nSuccessfully added 'Cell_line' information for all cells found in expression data.")
        else:
            print("Could not proceed with merge due to metadata loading issues.")
    else:
        print("Could not proceed with merge due to expression data loading issues.")
    
    print("\nScript finished.")
