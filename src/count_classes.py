import pandas as pd



def main():
    df = pd.read_csv("output/expression_matrix_with_tp53_status.csv", usecols=["Cell_line","TP53_status"])
    counts = df["TP53_status"].value_counts()
    print("Number of single cell observations per class:")
    print(counts)

    result = df.groupby("TP53_status")["Cell_line"].nunique()
    print("Number of unique Cell_line per TP53_status:\n")
    for status, count in result.items():
        print(f"Class {status}: {count} unique Cell_line")

if __name__ == "__main__":
    main()


