import anndata as ad
import argparse
import sys

def count_classes_generic(input_file, cell_line_column="Cell_line"):
    """
    Generic function to count classes in a TP53 status dataset from .h5ad.
    """
    print(f"Loading data from: {input_file}")
    adata = ad.read_h5ad(input_file)
    df = adata.obs[[cell_line_column, "TP53_status"]].copy()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    # Count single cell observations per class
    counts = df["TP53_status"].value_counts()
    print("\nNumber of single cell observations per class:")
    print(counts)
    # Count unique cell lines per TP53 status
    result = df.groupby("TP53_status")[cell_line_column].nunique()
    print(f"\nNumber of unique {cell_line_column} per TP53_status:")
    for status, count in result.items():
        print(f"Class {status}: {count} unique {cell_line_column}")
    return counts, result

def main_gambardella():
    """Main function for Gambardella dataset"""
    parser = argparse.ArgumentParser(description="Count TP53 mutation classes for Gambardella dataset (.h5ad)")
    parser.add_argument("--input-file", type=str, 
                       default="output/expression_matrix_gambardella_with_tp53_status.h5ad",
                       help="Path to the input .h5ad file with TP53 status")
    parser.add_argument("--cell-line-column", type=str, default="Cell_line",
                       help="Name of the cell line column")
    args = parser.parse_args()
    print("=" * 60)
    print("COUNTING CLASSES - GAMBARDELLA DATASET")
    print("=" * 60)
    count_classes_generic(args.input_file, args.cell_line_column)
    print("=" * 60)
    print("Analysis complete for Gambardella dataset")
    print("=" * 60)

def main_kinker():
    """Main function for Kinker dataset"""
    parser = argparse.ArgumentParser(description="Count TP53 mutation classes for Kinker dataset (.h5ad)")
    parser.add_argument("--input-file", type=str, 
                       default="output/expression_matrix_kinker_with_tp53_status.h5ad",
                       help="Path to the input .h5ad file with TP53 status")
    parser.add_argument("--cell-line-column", type=str, default="Cell_line",
                       help="Name of the cell line column")
    args = parser.parse_args()
    print("=" * 60)
    print("COUNTING CLASSES - KINKER DATASET")
    print("=" * 60)
    count_classes_generic(args.input_file, args.cell_line_column)
    print("=" * 60)
    print("Analysis complete for Kinker dataset")
    print("=" * 60)

def main():
    """Legacy main function for backward compatibility"""
    print("=" * 60)
    print("COUNTING CLASSES - LEGACY MODE")
    print("=" * 60)
    print("Using default file: output/expression_matrix_with_tp53_status.h5ad")
    count_classes_generic("output/expression_matrix_with_tp53_status.h5ad", "Cell_line")
    print("=" * 60)
    print("Analysis complete")
    print("=" * 60)

if __name__ == '__main__':
    # Check if --data argument is provided
    if "--data" in sys.argv:
        idx = sys.argv.index("--data")
        data_type = sys.argv[idx + 1]
        # Remove --data and dataset type from sys.argv
        new_argv = sys.argv[:idx] + sys.argv[idx+2:]
        sys.argv = new_argv
        if data_type == "kinker":
            main_kinker()
        elif data_type == "gambardella":
            main_gambardella()
        else:
            raise ValueError(f"Unknown dataset type: {data_type}")
    else:
        main()


