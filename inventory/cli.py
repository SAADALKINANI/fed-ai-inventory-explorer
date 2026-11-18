"""``fedai`` command line."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fedai", description="Explore the Federal AI Use Case Inventory")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary", help="headline numbers")
    d = sub.add_parser("dashboard", help="build the static HTML dashboard")
    d.add_argument("--out", default="docs/index.html")
    s = sub.add_parser("similar", help="find existing federal use cases similar to an idea")
    s.add_argument("idea")
    s.add_argument("--top", type=int, default=8)
    c = sub.add_parser("compliance", help="check a proposal against a solicitation")
    c.add_argument("solicitation")
    c.add_argument("proposal")
    c.add_argument("--csv")
    c.add_argument("--html")
    a = p.parse_args(argv)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 70)

    if a.cmd == "compliance":
        from .proposal import check_compliance, matrix_to_html

        m = check_compliance(Path(a.solicitation).read_text(encoding="utf-8"),
                             Path(a.proposal).read_text(encoding="utf-8"))
        print(m[["req_id", "section", "status", "score", "requirement"]].to_string(index=False))
        print("\n" + m.status.value_counts().to_string())
        if a.csv:
            m.to_csv(a.csv, index=False)
        if a.html:
            Path(a.html).write_text(matrix_to_html(m, f"Compliance: {Path(a.proposal).stem}"), encoding="utf-8")
        return

    from .analysis import headline
    from .data import load_inventory

    df = load_inventory()
    if a.cmd == "summary":
        print(headline(df).to_string())
    elif a.cmd == "dashboard":
        from .dashboard import build_dashboard

        print(f"dashboard → {build_dashboard(df, a.out)}")
    elif a.cmd == "similar":
        from .textmodel import UseCaseSearch

        print(UseCaseSearch(df).query(a.idea, a.top).to_string(index=False))


if __name__ == "__main__":
    main()
