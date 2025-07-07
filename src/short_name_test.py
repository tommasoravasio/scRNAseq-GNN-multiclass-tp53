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
    print("ATTENZIONE! I seguenti short name NON sono univoci:")
    for short, fulls in ambiguous.items():
        print(f"{short}: {sorted(fulls)}")
else:
    print("Tutti gli short name sono univoci!")