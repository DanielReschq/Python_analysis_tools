"""Generate LaTeX figure+table blocks from Binder single-fit result .txt files.

Reads every ``*.txt`` fit-result file under
``/home/daniel/Master_thesis/Latex/plots/Binder/Single_Fits/`` (searched
recursively, since the files live in ``ZeroTemp``/``FiniteTemp``
subdirectories) and turns each one into a LaTeX figure block: the matching
PDF plot on the left, a table of the fitted parameters on the right.
"""
import argparse
import re
from pathlib import Path

PLOTS_ROOT = Path("/home/daniel/Master_thesis/Latex/plots")
DEFAULT_SEARCH_DIR = PLOTS_ROOT / "Binder" / "Single_Fits"

LINE_RE = re.compile(r"^\s*([^\s=]+)\s*=\s*(.+?)\s*$")

# preferred left-to-right column order; any keys not listed here (other than
# a0, a1, ... which are handled separately) are appended at the end.
ORDER = ["chi2/d.o.f", "gc", "nu", "z", "twoBetaNuInv", "omega","c"]

SYMBOLS = {
    "chi2/d.o.f": r"\chi^2",
    "gc": "g_c",
    "nu": r"\nu",
    "omega": r"\omega",
    "twoBetaNuInv": r"2\beta/\nu",
    "z": "z",
    "c": "c",
}


def _symbol(key: str, gc_label: str = "g_c") -> str:
    if key == "gc":
        return gc_label
    if key in SYMBOLS:
        return SYMBOLS[key]
    match = re.fullmatch(r"a(\d+)", key)
    if match:
        return f"a_{match.group(1)}"
    return key


def _gc_label(txt_path: Path) -> str:
    parts = txt_path.parts
    if "ZeroTemp" in parts:
        return r"J^{\perp}_c"
    if "FiniteTemp" in parts:
        return "T_c"
    return "g_c"

def _strip_uncertainty(value: str) -> str:
    return value.split("(")[0]


def _alpha_str(txt_path: Path) -> str:
    match = re.match(r"alpha(\d+\.\d+)", txt_path.stem)
    alpha = match.group(1).rstrip("0")
    if alpha.endswith("."):
        alpha += "0"
    return alpha

def _caption(txt_path: Path) -> str:
    parts = txt_path.parts
    alpha = _alpha_str(txt_path)
    if "ZeroTemp" in parts:
        return f"Spin-stiffness for $\\alpha = {alpha}$ at zero temperature."
    if "FiniteTemp" in parts:
        return f"Spin-stiffness for $\\alpha = {alpha}$ at finite temperature."
    return f"Spin-stiffness for $\\alpha = {alpha}$."


def parse_fit_file(path: Path) -> dict:
    """Parse 'key = value' lines of a fit-result file into a dict."""
    results = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        match = LINE_RE.match(line)
        if not match:
            continue
        key, value = match.groups()
        results[key] = value
    return results


def ordered_keys(results: dict) -> list:
    keys = [k for k in ORDER if k in results]
    a_keys = sorted(
        (k for k in results if re.fullmatch(r"a\d+", k)),
        key=lambda k: int(k[1:]),
    )
    keys += a_keys
    keys += [k for k in results if k not in keys]
    return keys


def make_figure_block(txt_path: Path) -> str:
    results = parse_fit_file(txt_path)
    keys = ordered_keys(results)
    gc_label = _gc_label(txt_path)
    caption = _caption(txt_path)

    pdf_rel = txt_path.relative_to(PLOTS_ROOT).with_suffix(".pdf")
    table_keys = [k for k in keys if k not in ("gc", "nu")]
    rows = "\n".join(
        f"\t\t\t${_symbol(k, gc_label)}$ & ${results[k]}$ \\\\ \n\t\t\t\\hline" for k in table_keys
    )

    return (
        "\\begin{figure}[!htb]\n"
        "\t\\centering\n"
        "\t\\begin{minipage}{0.74\\textwidth}\n"
        "\t\t\\centering\n"
        f"\t\t\\includegraphics[width=\\linewidth, keepaspectratio]{{{pdf_rel}}}\n"
        "\t\\end{minipage}\n"
        "\t\\hfill\n"
        "\t\\begin{minipage}{0.24\\textwidth}\n"
        "\t\t\\centering\n"
        "\t\t\\begin{tabular}{|c|c|}\n"
        "\t\t\t\\hline\n"
        f"{rows}\n"
        "\t\t\\end{tabular}\n"
        "\t\\end{minipage}\n"
        f"\t\\caption{{{caption}}}\n"
        f"\t\\label{{fig:{txt_path.stem}}}\n"
        "\\end{figure}\n"
    )


def build_tables(search_dir: Path) -> str:
    txt_files = sorted(
        p for p in search_dir.rglob("*.txt") if p.stem.endswith("fixed")
    )
    return "\n".join(make_figure_block(p) for p in txt_files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="write result to this file instead of stdout",
    )
    args = parser.parse_args()

    result = build_tables(args.search_dir)
    if args.output:
        args.output.write_text(result)
    else:
        print(result)


if __name__ == "__main__":
    main()
