#!/bin/bash
cd "$(dirname "$0")"
# Job script to merge Gambardella and Kinker .h5ad files, keeping only common genes

# Set paths 
GAMBARDELLA_H5AD="../output/expression_matrix_gambardella_with_tp53_status_and_gene_symbols.h5ad"
KINKER_H5AD="../output/expression_matrix_kinker_with_tp53_status.h5ad"
OUTPUT_H5AD="../output/merged_gambardella_kinker_common_genes.h5ad"


python ../src/merge_gambardella_kinker.py \
  --gambardella "$GAMBARDELLA_H5AD" \
  --kinker "$KINKER_H5AD" \
  --output "$OUTPUT_H5AD" 