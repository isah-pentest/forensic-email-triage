"""Table I: subsets, splits and weak-label positives. Source: splits/*/split_meta.yaml."""
import pandas as pd, yaml
from analysis import common as C

def make(show=False):
    rows, latex = [], []
    for s in C.SCALES:
        m = yaml.safe_load(open(C.paths()["splits"] / s / "split_meta.yaml"))
        n = sum(m["sizes"].values()); prev = 100 * m["label_distribution_full"][2] / n
        rows.append({"scale": s.split("_")[1][:-3] + "k", "train": m["sizes"]["train"], "val": m["sizes"]["val"],
                     "test": m["sizes"]["test"], "test_pos": m["label_distribution_test"][2], "prevalence_pct": round(prev, 2)})
        r = rows[-1]
        latex.append(f"{r['scale']} & {r['train']:,} & {r['val']:,} & {r['test']:,} & {r['test_pos']} & {prev:.2f}\\,\\% \\\\")
    df = pd.DataFrame(rows)
    C.save_table(df, "table1_splits", latex, ["splits/*/split_meta.yaml"])
    if show: display_df(df)
    return df

def display_df(df):
    try:
        from IPython.display import display; display(df)
    except ImportError:
        print(df.to_string(index=False))

if __name__ == "__main__":
    C.cli(__doc__); print(make().to_string(index=False))
