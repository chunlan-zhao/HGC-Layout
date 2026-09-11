"""Turn the evaluation records into the reported tables.

Usage:
    python scripts/make_tables.py --run-dir outputs/demo
"""
import argparse
import logging
from pathlib import Path

from hgc_layout.metrics.evaluate import METRIC_NAMES
from hgc_layout.tables import aggregate, markdown_table, save_table, significance_table
from hgc_layout.utils.io import read_json

ABLATION_COLUMNS = ["SPR", "ARE", "COL", "BUF", "CLIPsim", "SP"]
OPTIMIZER_COLUMNS = ["COL", "BUF", "GAC", "BND", "SP", "Time"]
DATA_COLUMNS = ["SPR", "CAT", "RCN", "BUF", "CLIPsim", "SP"]
ROBUSTNESS_COLUMNS = ["SPR", "COL", "RCN", "SP"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="outputs/main")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_dir = Path(args.run_dir)
    results = read_json(run_dir / "results.json")
    written = []

    if "comparison" in results:
        records = aggregate(results["comparison"], with_std=True)
        save_table(records, run_dir, "table1_main_results", list(METRIC_NAMES),
                   title="Main results on the test split", with_std=True)
        written.append("table1_main_results")
    if "significance" in results:
        rows = significance_table(results["significance"])
        (run_dir / "tableS1_significance.md").write_text(
            markdown_table(rows, ["W", "p_raw", "p_holm", "n", "significant"],
                           key_columns=("metric", "baseline"),
                           title="Paired Wilcoxon signed-rank tests with Holm correction",
                           bold_best=False), encoding="utf-8")
        written.append("tableS1_significance")
    if "core_ablation" in results:
        save_table(aggregate(results["core_ablation"]), run_dir, "table2_core_ablation",
                   ABLATION_COLUMNS, title="Ablation of the core components")
        written.append("table2_core_ablation")
    if "module_ablation" in results:
        save_table(aggregate(results["module_ablation"]), run_dir, "tableS3_module_ablation",
                   ABLATION_COLUMNS + ["CNT"], title="Ablation of the individual modules")
        written.append("tableS3_module_ablation")
    if "optimizer_ablation" in results:
        save_table(aggregate(results["optimizer_ablation"]), run_dir, "table4_optimizer_ablation",
                   OPTIMIZER_COLUMNS, title="Ablation of the adaptive hierarchical optimization")
        written.append("table4_optimizer_ablation")
    if "data_source" in results:
        save_table(aggregate(results["data_source"]), run_dir, "table5_data_source",
                   DATA_COLUMNS, title="Contribution of the individual data sources")
        written.append("table5_data_source")
    if "robustness" in results:
        records = aggregate(results["robustness"], extra_keys=("dropout",))
        records.sort(key=lambda r: (r["method"], r["dropout"]))
        save_table(records, run_dir, "table3_robustness", ROBUSTNESS_COLUMNS,
                   key_columns=("method", "dropout"),
                   title="Robustness to dropout of POI records and road segments")
        written.append("table3_robustness")
    if "editing" in results:
        rows = results["editing"]
        groups = {}
        for r in rows:
            groups.setdefault((r["operation"], r["complexity"]), []).append(float(r["time"]))
        records = [{"operation": k[0], "complexity": k[1], "Time": sum(v) / len(v)}
                   for k, v in sorted(groups.items())]
        save_table(records, run_dir, "tableS2_editing", ["Time"],
                   key_columns=("operation", "complexity"),
                   title="Editing completion time per operation")
        written.append("tableS2_editing")
    if "reliability" in results:
        rel = results["reliability"]
        rows = [{"quantity": k, "value": (f"{v:.3f}" if isinstance(v, float) else str(v))}
                for k, v in rel.items() if k != "spearman"]
        rows += [{"quantity": f"Spearman pair {i + 1}", "value": f"{v:.3f}"}
                 for i, v in enumerate(rel.get("spearman", []))]
        (run_dir / "tableS6_sp_reliability.md").write_text(
            markdown_table(rows, ["value"], key_columns=("quantity",),
                           title="Reliability of the semantic-plausibility score",
                           bold_best=False), encoding="utf-8")
        written.append("tableS6_sp_reliability")

    parts = [(run_dir / f"{n}.md").read_text(encoding="utf-8") for n in written]
    (run_dir / "all_tables.md").write_text("\n\n".join(parts), encoding="utf-8")
    logging.info("wrote %d tables to %s", len(written), run_dir)


if __name__ == "__main__":
    main()
