import pandas as pd

def update_expression_matrix(expression_df, master_mutation_dict, cell_lines_with_multiple_mutations_global,cell_line_name="Cell_line"):
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

    print(f"Initial number of cells: {initial_cell_count}")
    print(f"Number of cells removed (due to multiple TP53 mutations for the cell line): {count_removed_multiple_mutations}")
    print(f"Number of cells removed (cell line not in mutation data or no single TP53 entry after filtering multi-mut): {count_removed_missing_info}")
    print(f"Total number of cells kept: {final_cell_count}")

    return expression_df_final

def process_tp53_mutations(mutation_files):
    """
    Process TP53 mutation files and return:
    - master_mutation_dict: barcodes with exactly one TP53 mutation and a single Variant_Classification
    - cell_lines_with_multiple_mutations_global: barcodes with multiple distinct Variant_Classification values
    """
    raw_mutation_dfs = []

    for mut_file in mutation_files:
        try:
            mut_df = pd.read_csv(mut_file, sep='\t', comment='#')
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

    grouped = tp53_all_files.groupby('Tumor_Sample_Barcode')

    for barcode, group in grouped:
        # Convert to list and use set for unique values to avoid type checker issues
        variant_list = list(group['Variant_Classification'])
        unique_classes = list(set(variant_list))
    
        if len(unique_classes) == 1:
            master_mutation_dict[barcode] = unique_classes[0]
        else:
            cell_lines_with_multiple_mutations_global[barcode] = list(unique_classes)

    return master_mutation_dict, cell_lines_with_multiple_mutations_global

def main():
    # --- Configuration ---
    expression_matrix_file = 'output/expression_matrix.csv'
    mutation_files = [
        'data/ccle_broad_2019/data_mutations.txt',
        'data/cellline_ccle_broad/data_mutations.txt'
    ]
    output_expression_file = 'output/expression_matrix_with_tp53_status.csv'
    # --- End Configuration ---

    print("Processing mutation files to extract TP53 mutation status...")
    master_mutation_dict, cell_lines_with_multiple_mutations_global = process_tp53_mutations(mutation_files)

    print(f"\nProcessed all mutation files:")
    print(f"  Master TP53 mutation dictionary (unique classification): {len(master_mutation_dict)} entries")
    multi_mut_list_preview = list(cell_lines_with_multiple_mutations_global.keys())
    print(f"  Cell lines with multiple TP53 variant types: {len(cell_lines_with_multiple_mutations_global)} {multi_mut_list_preview[:5]}{'...' if len(multi_mut_list_preview) > 5 else ''}")

    print(f"\nLoading expression matrix: {expression_matrix_file}")
    try:
        expression_df = pd.read_csv(expression_matrix_file)
    except FileNotFoundError:
        print(f"Error: Expression matrix file not found at {expression_matrix_file}")
        return
    except Exception as e:
        print(f"Error loading expression matrix {expression_matrix_file}: {e}")
        return

    cell_line_name = "Cell_line"  # Adjust this if needed
    if cell_line_name not in expression_df.columns:
        print(f"Error: Column '{cell_line_name}' not found in expression matrix. Mapping requires this column.")
        return

    print("Updating expression matrix with TP53 status...")
    final_expression_df = update_expression_matrix(
        expression_df,
        master_mutation_dict,
        cell_lines_with_multiple_mutations_global
    )

    print(f"\nSaving updated expression matrix to: {output_expression_file}")
    try:
        final_expression_df.to_csv(output_expression_file, index=False)
        print("Processing complete.")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == '__main__':
    main()
