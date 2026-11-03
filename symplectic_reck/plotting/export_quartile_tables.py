"""Export saved results as Markdown and LaTeX tables of medians and quartiles."""

import csv

from ..io_utils import ROOT

METHODS = [("measured_closed_form", "Measured closed-form"), ("finite_diff", "Finite difference"),
           ("dither_lockin", "Dithering"), ("genetic_algorithm", "Genetic algorithm")]
BUDGET = ("All feedback methods use 2000 loss evaluations per gate. Lock-in dithers all gate parameters "
          "at once on separate tones (20-sample windows for two-mode gates, 8 for single-mode gates) "
          "with amplitude 0.05; FD uses step 0.08. ")


def export_table(root, source, stem, index, caption):
    with (root / source).open() as stream:
        rows = list(csv.DictReader(stream))
    selected = [r for r in rows if r["method"] in dict(METHODS)]
    keys = sorted({r[index] for r in selected}, key=lambda x: float(x) if x else float("inf"))
    headings = ["q" if index == "shots" else "N"] + [label for _, label in METHODS]
    table = []
    for key in keys:
        cells = [key or "infinity"]
        for method, _ in METHODS:
            row = next(r for r in selected if r[index] == key and r["method"] == method)
            med, lo, hi = (float(row[k]) for k in ("median", "q25", "q75"))
            assert lo <= med <= hi
            cells.append(f"{med:.3g} [{lo:.3g}, {hi:.3g}]")
        table.append(cells)
    markdown = caption + "\n\n| " + " | ".join(headings) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headings)) + " |\n"
    markdown += "\n".join("| " + " | ".join(row) + " |" for row in table) + "\n"
    (root / f"{stem}.md").write_text(markdown)
    latex = "\\begin{table}[ht]\n\\centering\n\\caption{" + caption + "}\n"
    latex += "\\begin{tabular}{l" + "r" * len(METHODS) + "}\n\\hline\n" + " & ".join(headings) + " \\\\\n\\hline\n"
    for row in table:
        latex += " & ".join(row).replace("infinity", "$\\infty$") + " \\\\\n"
    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    (root / f"{stem}.tex").write_text(latex)
    print(f"Wrote {stem}.md and {stem}.tex")


def main():
    root = ROOT / "results/finite_shots"
    export_table(root, "finite_shots_local_nulling_losses.csv", "table1_median_quartiles", "shots",
                 "Local nulling loss at N=3, reported as median [Q25, Q75]. "
                 "Quartiles pool the three two-mode steps across 30 trials (90 losses per entry). "
                 "These describe variation in pooled step losses, not confidence intervals. "
                 "Closed-form losses are evaluated on the actual post-gate block; feedback methods use the saved "
                 "final optimizer loss. " + BUDGET + "The closed-form shot allocation is budget matched.")
    export_table(root, "finite_shots_full_mesh_residuals.csv", "table2_median_quartiles", "N",
                 "Normalized final residual, reported as median [Q25, Q75] across 30 trials at q=1000. "
                 + BUDGET + "The closed-form shot allocation is budget matched. "
                 "Quartiles describe variation across trials, not confidence intervals.")


if __name__ == "__main__":
    main()
