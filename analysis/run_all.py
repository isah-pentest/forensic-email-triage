"""Rebuild all figures and tables: python -m analysis.run_all [--v2 PATH] [--out PATH]"""
from analysis import common as C
from analysis import fig1_learning_curve, fig2_review_budget, table1_splits, table2_main_results, table3_late_fusion, table4_review_budget, table5_trec07, derived_statistics

if __name__ == "__main__":
    C.cli(__doc__)
    print("reading:", {k: str(v) for k, v in C.paths().items()})
    for mod in [fig1_learning_curve, fig2_review_budget]:
        print("wrote", mod.make())
    for mod in [table1_splits, table2_main_results, table3_late_fusion, table4_review_budget, table5_trec07, derived_statistics]:
        mod.make(); print("wrote", mod.__name__.split(".")[-1])
    print("wrote", C.write_manifest(), "(verify with: python -m analysis.verify_manifest)")
