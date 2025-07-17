import pandas as pd
import argparse
import os

def update_expression_matrix(expression_df, master_mutation_dict, cell_lines_with_multiple_mutations_global, cell_line_name="Cell_line"):
    """
    Updates the expression matrix with TP53 mutation status.

    Args:
        expression_df (pd.DataFrame): The gene expression DataFrame.
        master_mutation_dict (dict): Dictionary mapping cell lines to their TP53 mutation status (single mutation).
        cell_lines_with_multiple_mutations_global (set): Set of cell lines to exclude due to multiple TP53 mutations.

    Returns:
        pd.DataFrame: Updated expression DataFrame with 'TP53_status' column and filtered rows.
    """
    initial_cell_count = len(expression_df)

    # Identify cells from lines with multiple mutations
    is_multi_mut_line = expression_df[cell_line_name].isin(cell_lines_with_multiple_mutations_global)
    count_removed_multiple_mutations = is_multi_mut_line.sum()

    # Filter out these cells first
    expression_df_filtered = expression_df[~is_multi_mut_line].copy()

    # Add TP53_status column to the filtered DataFrame
    # Ensure 'TP53_status' column exists before trying to fill it, or handle if it might not.
    # Using .loc for assignment to avoid SettingWithCopyWarning
    expression_df_filtered.loc[:, 'TP53_status'] = expression_df_filtered[cell_line_name].map(master_mutation_dict)

    is_missing_info = expression_df_filtered['TP53_status'].isna()
    count_removed_missing_info = is_missing_info.sum()

    expression_df_final = expression_df_filtered.dropna(subset=['TP53_status'])

    final_cell_count = len(expression_df_final)

    print(f"Initial number of cells in chunk: {initial_cell_count}")
    print(f"Number of cells removed (due to multiple TP53 mutations for the cell line): {count_removed_multiple_mutations}")
    print(f"Number of cells removed (cell line not in mutation data or no single TP53 entry after filtering multi-mut): {count_removed_missing_info}")
    print(f"Total number of cells kept in chunk: {final_cell_count}")

    return expression_df_final

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
    """
    Parse command line arguments for different datasets (Gambardella, Kinker, etc.)
    """
    parser = argparse.ArgumentParser(description='Add TP53 mutation status to expression matrix')
    
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['gambardella', 'kinker'],
                       help='Dataset name (gambardella or kinker)')
    
    parser.add_argument('--expression-matrix', type=str, required=True,
                       help='Path to expression matrix file')
    
    parser.add_argument('--mutation-files', type=str, nargs='+', required=True,
                       help='List of mutation files')
    
    parser.add_argument('--output-file', type=str, required=True,
                       help='Output file path')
    
    parser.add_argument('--cell-line-column', type=str, required=True,
                       help='Name of the cell line column in expression matrix')
    
    return parser.parse_args()



def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # All arguments are now required, so we use them directly
    expression_matrix_file = args.expression_matrix
    mutation_files = args.mutation_files
    output_expression_file = args.output_file
    cell_line_column = args.cell_line_column

    print(f"Dataset: {args.dataset}")
    print(f"Expression matrix file: {expression_matrix_file}")
    print(f"Mutation files: {mutation_files}")
    print(f"Output file: {output_expression_file}")
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

    print(f"\nLoading expression matrix in chunks: {expression_matrix_file}")
    chunk_size = 100000  # Adjust as needed for your memory
    first_chunk = True
    try:
        for chunk in pd.read_csv(expression_matrix_file, chunksize=chunk_size):
            # Ensure the cell_line_column is string type
            if cell_line_column in chunk.columns:
                chunk[cell_line_column] = chunk[cell_line_column].astype(str)
                print("Example cell lines from expression matrix chunk (original):")
                print(chunk[cell_line_column].unique()[:10])
                # Transform to short barcode format (before first underscore)
                chunk[cell_line_column] = chunk[cell_line_column].str.split('_').str[0]
                print("Example cell lines from expression matrix chunk (short barcode):")
                print(chunk[cell_line_column].unique()[:10])
            else:
                print(f"Error: Column '{cell_line_column}' not found in chunk. Available columns: {list(chunk.columns)}")
                continue
            print(f"\nProcessing new chunk...")
            final_expression_df = update_expression_matrix(
                chunk,
                master_mutation_dict,
                cell_lines_with_multiple_mutations_global,
                cell_line_column
            )
            if not final_expression_df.empty:
                final_expression_df.to_csv(
                    output_expression_file,
                    mode='w' if first_chunk else 'a',
                    index=False,
                    header=first_chunk
                )
                first_chunk = False
    except FileNotFoundError:
        print(f"Error: Expression matrix file not found at {expression_matrix_file}")
        return
    except Exception as e:
        print(f"Error loading or processing expression matrix {expression_matrix_file}: {e}")
        return
    print("Processing complete.")

if __name__ == '__main__':
    main()
