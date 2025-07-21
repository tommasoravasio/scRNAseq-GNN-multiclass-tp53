import anndata as ad
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Merge Gambardella and Kinker AnnData .h5ad files, keeping only common genes and all cells.")
    parser.add_argument('--gambardella', type=str, required=True, help='Path to Gambardella .h5ad file')
    parser.add_argument('--kinker', type=str, required=True, help='Path to Kinker .h5ad file')
    parser.add_argument('--output', type=str, required=True, help='Path to output merged .h5ad file')
    args = parser.parse_args()

    print(f"Loading Gambardella AnnData from {args.gambardella}")
    adata_gambardella = ad.read_h5ad(args.gambardella)
    adata_gambardella.var_names_make_unique()

    print(f"Loading Kinker AnnData from {args.kinker}")
    adata_kinker = ad.read_h5ad(args.kinker)
    adata_kinker.var_names_make_unique()

    print("Finding common genes...")
    common_genes = adata_gambardella.var_names.intersection(adata_kinker.var_names)
    print(f"Number of common genes: {len(common_genes)}")
    if len(common_genes) == 0:
        raise ValueError("No common genes found between the two datasets.")

    print("Subsetting AnnData objects to common genes...")
    adata_gambardella = adata_gambardella[:, common_genes].copy()
    adata_kinker = adata_kinker[:, common_genes].copy()

    print("Adding dataset origin column to .obs...")
    adata_gambardella.obs['dataset'] = 'gambardella'
    adata_kinker.obs['dataset'] = 'kinker'

    print("Concatenating AnnData objects along cells (rows)...")
    adata_merged = ad.concat([adata_gambardella, adata_kinker], axis=0, join='inner', merge='same')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Saving merged AnnData to {args.output}")
    adata_merged.write(args.output)
    print("Done!")

if __name__ == "__main__":
    main() 