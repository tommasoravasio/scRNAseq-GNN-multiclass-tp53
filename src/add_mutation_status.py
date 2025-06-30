import pandas as pd

# load_and_filter_mutations is not directly used by the new main but might be useful if run per file for other reasons.
# For this specific problem, the combined approach in main is better.
# We can leave it here or remove it if it's confirmed to be unused.
# For now, let's keep it but ensure main() does not call it.

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
    expression_df_filtered.loc[:, 'TP53_status'] = expression_df_filtered['cell_line_name'].map(master_mutation_dict)

    is_missing_info = expression_df_filtered['TP53_status'].isna()
    count_removed_missing_info = is_missing_info.sum()

    expression_df_final = expression_df_filtered.dropna(subset=['TP53_status'])

    final_cell_count = len(expression_df_final)

    print(f"Initial number of cells: {initial_cell_count}")
    print(f"Number of cells removed (due to multiple TP53 mutations for the cell line): {count_removed_multiple_mutations}")
    print(f"Number of cells removed (cell line not in mutation data or no single TP53 entry after filtering multi-mut): {count_removed_missing_info}")
    print(f"Total number of cells kept: {final_cell_count}")

    return expression_df_final

def main():
    # --- Configuration ---
    expression_matrix_file = 'output/expression_matrix.csv'
    mutation_files = [
        'data/ccle_broad_2019/data_mutations.txt',
        'data/cellline_ccle_broad/data_mutations.txt'
    ]
    output_expression_file = 'output/expression_matrix_with_tp53_status.csv'
    # --- End Configuration ---

    raw_mutation_dfs = []
    print("Loading and combining mutation files...")
    for mut_file in mutation_files:
        try:
            mut_df = pd.read_csv(mut_file, sep='\t', header=0)
            # Check for essential columns
            required_cols = {'Hugo_Symbol', 'Tumor_Sample_Barcode', 'Variant_Classification'}
            if not required_cols.issubset(mut_df.columns):
                print(f"Warning: File {mut_file} is missing one or more required columns: {required_cols - set(mut_df.columns)}. Skipping this file.")
                continue
            raw_mutation_dfs.append(mut_df)
            print(f"  Successfully loaded {mut_file}")
        except FileNotFoundError:
            print(f"Error: Mutation file not found at {mut_file}. Skipping.")
        except Exception as e:
            print(f"Error loading mutation file {mut_file}: {e}. Skipping.")

    if not raw_mutation_dfs:
        print("No valid mutation data loaded. Exiting.")
        # Create empty outputs or handle as error
        master_mutation_dict = {}
        cell_lines_with_multiple_mutations_global = set()
    else:
        combined_mut_df = pd.concat(raw_mutation_dfs, ignore_index=True)
        tp53_all_files = combined_mut_df[combined_mut_df['Hugo_Symbol'] == 'TP53']

        if tp53_all_files.empty:
            print("No TP53 mutations found in any of the provided files.")
            master_mutation_dict = {}
            cell_lines_with_multiple_mutations_global = set()
        else:
            mutation_counts = tp53_all_files['Tumor_Sample_Barcode'].value_counts()

            cell_lines_with_multiple_mutations_global = set(mutation_counts[mutation_counts > 1].index)
            single_mutation_lines_barcodes = mutation_counts[mutation_counts == 1].index

            master_mutation_dict = {}
            for barcode in single_mutation_lines_barcodes:
                # Get the variant classification for this barcode from the combined TP53 table
                # Since count is 1, there's only one row for this barcode in tp53_all_files
                variant_class = tp53_all_files[
                    tp53_all_files['Tumor_Sample_Barcode'] == barcode
                ]['Variant_Classification'].iloc[0]
                master_mutation_dict[barcode] = variant_class

    print(f"\nProcessed all mutation files:")
    print(f"  Master TP53 mutation dictionary (single mutation lines) size: {len(master_mutation_dict)}")
    multi_mut_list_preview = list(cell_lines_with_multiple_mutations_global)
    print(f"  Cell lines with multiple TP53 mutations: {len(cell_lines_with_multiple_mutations_global)} {multi_mut_list_preview[:5]}{'...' if len(multi_mut_list_preview) > 5 else ''}")

    print(f"\nLoading expression matrix: {expression_matrix_file}")
    try:
        expression_df = pd.read_csv(expression_matrix_file)
    except FileNotFoundError:
        print(f"Error: Expression matrix file not found at {expression_matrix_file}")
        return
    except Exception as e:
        print(f"Error loading expression matrix {expression_matrix_file}: {e}")
        return

    if 'cell_line_name' not in expression_df.columns:
        print(f"Error: 'cell_line_name' column not found in {expression_matrix_file}. This column is required for mapping.")
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
