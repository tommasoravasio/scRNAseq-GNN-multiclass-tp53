"""
This script checks the uniqueness of "short names" (the prefix before the first underscore)
in the 'Tumor_Sample_Barcode' column across multiple mutation data files. It reports if any
short name maps to more than one full barcode, which would indicate ambiguity in sample naming.
"""

import pandas as pd

mutation_files = [
    "data/ccle_broad_2019/data_mutations.txt",
    "data/cellline_ccle_broad/data_mutations.txt"
]

all_short_to_full = {}

for mut_file in mutation_files:
    print(f"Processing {mut_file} ...")
    try:
        for chunk in pd.read_csv(mut_file, sep='\t', comment='#', usecols=['Tumor_Sample_Barcode'], chunksize=100000):
            for full in chunk['Tumor_Sample_Barcode'].dropna().unique():
                short = full.split('_')[0]
                if short not in all_short_to_full:
                    all_short_to_full[short] = set()
                all_short_to_full[short].add(full)
    except Exception as e:
        print(f"Error processing {mut_file}: {e}")

ambiguous = {k: v for k, v in all_short_to_full.items() if len(v) > 1}

if ambiguous:
    print("WARNING! The following short names are NOT unique:")
    for short, fulls in ambiguous.items():
        print(f"{short}: {sorted(fulls)}")
else:
    print("All short names are unique!")