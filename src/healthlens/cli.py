"""Command-line interface.

    healthlens run export.zip        # import + report + dashboard, opens the report
    healthlens demo                  # same, with a synthetic person (no real data needed)
    healthlens import export.zip     # parse only
    healthlens report | dashboard    # rebuild from the last import
    healthlens info                  # what's in the imported data
    healthlens export-csv            # daily / nights / workouts tables as CSV
    healthlens clear --yes           # delete the imported data and reports
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import webbrowser
from datetime import date
from pathlib import Path
from typing import Optional

from . import __version__
from .config import Workspace


def _progress(stage: str, frac: float, counts: dict) -> None:
    if stage == "retry":
        print("\n  The export has malformed characters; re-reading it in repair mode…", flush=True)
        return
    bar = int(frac * 30)
    sys.stdout.write(f"\r  Reading export  [{'#' * bar}{'.' * (30 - bar)}] {frac * 100:5.1f}%  "
                     f"{counts.get('records', 0):,} records")
    sys.stdout.flush()


def _open(path: Path, enabled: bool) -> None:
    if enabled:
        try:
            webbrowser.open(path.resolve().as_uri())
        except Exception:
            pass


def do_import(ws: Workspace, export: str, force: bool = False) -> dict:
    from .parser import import_export

    print(f"Importing {export}")
    t = time.monotonic()
    meta = import_export(export, ws.cache, progress=_progress, force=force)
    if meta.get("skipped"):
        print("  Already imported (same file) — using the existing data. Pass --force to re-read it.")
    else:
        print(f"\n  Done in {time.monotonic() - t:.0f} s: {meta['n_records']:,} records, "
              f"{meta['n_workouts']:,} workouts, {len(meta['types'])} data types.")
    return meta


def _analysis(ws: Workspace, units: Optional[str]):
    from .analysis import analyse
    from .metrics import load_or_compute
    from .store import HealthStore

    store = HealthStore(ws.cache)
    print("Computing daily metrics, sleep and workouts…")
    m = load_or_compute(store, progress=lambda s: None)
    print("Finding patterns…")
    return analyse(m, None if units in (None, "auto") else units)


def do_report(ws: Workspace, units: Optional[str], out: Optional[str], demo: bool, open_it: bool,
              analysis=None) -> Path:
    from .report import write_report

    a = analysis or _analysis(ws, units)
    ws.ensure()
    path = Path(out) if out else ws.reports / "report.html"
    write_report(a, path, demo=demo)
    print(f"Report:    {path}")
    if a.errors:
        print(f"  Note: {len(a.errors)} part(s) of the analysis failed; details at the end of the report.")
    _open(path, open_it)
    return path


def do_dashboard(ws: Workspace, units: Optional[str], out: Optional[str], demo: bool, open_it: bool,
                 analysis=None) -> Path:
    from .dashboard import write_dashboard

    a = analysis or _analysis(ws, units)
    ws.ensure()
    path = Path(out) if out else ws.reports / "dashboard.html"
    write_dashboard(a, path, demo=demo)
    print(f"Dashboard: {path}")
    _open(path, open_it)
    return path


def cmd_run(args) -> int:
    ws = Workspace.at(args.home)
    do_import(ws, args.export, force=args.force)
    a = _analysis(ws, args.units)
    do_report(ws, args.units, None, False, not args.no_open, analysis=a)
    do_dashboard(ws, args.units, None, False, False, analysis=a)
    _privacy_note(ws)
    return 0


def cmd_demo(args) -> int:
    from .synthetic import SyntheticConfig, generate_export

    base = Workspace.at(args.home)
    ws = Workspace(base.home / "demo")
    export = ws.home / "demo-export.zip"
    end = date.today().replace(day=1)
    start = date(end.year - args.years, end.month, 1)
    if not export.exists() or args.force:
        print(f"Generating a synthetic Apple Health export ({args.years} years)…")
        cfg = SyntheticConfig(start=start, end=end, watch_start=date(start.year + 1, 10, 6) if args.years > 2
                              else start, seed=args.seed)
        generate_export(export, cfg)
    do_import(ws, str(export), force=args.force)
    a = _analysis(ws, args.units)
    do_report(ws, args.units, None, True, not args.no_open, analysis=a)
    do_dashboard(ws, args.units, None, True, False, analysis=a)
    print("\nThis is synthetic demo data. Run `healthlens run path/to/export.zip` for your own.")
    return 0


def cmd_import(args) -> int:
    do_import(Workspace.at(args.home), args.export, force=args.force)
    return 0


def cmd_report(args) -> int:
    do_report(Workspace.at(args.home), args.units, args.out, False, not args.no_open)
    return 0


def cmd_dashboard(args) -> int:
    do_dashboard(Workspace.at(args.home), args.units, args.out, False, not args.no_open)
    return 0


def cmd_info(args) -> int:
    from .store import HealthStore
    import pandas as pd

    ws = Workspace.at(args.home)
    store = HealthStore(ws.cache)
    meta = store.meta
    print(f"Workspace: {ws.home}")
    print(f"Export:    {meta.get('source_file')} (exported {meta.get('export_date')})")
    print(f"Imported:  {meta.get('imported_at')} in {meta.get('parse_seconds')} s"
          f"{' (repair mode)' if meta.get('repair_mode') else ''}")
    print(f"Records:   {meta.get('n_records', 0):,} · workouts {meta.get('n_workouts', 0):,} · "
          f"activity summaries {meta.get('n_activity_summaries', 0):,}")
    print(f"Units:     {meta.get('unit_system')}")
    print("\nSources:")
    for _, r in store.sources.sort_values("records", ascending=False).iterrows():
        if int(r.get("records", 0) or 0) == 0:
            continue
        first = pd.Timestamp(int(r["first"]), unit="s").date() if pd.notna(r["first"]) else "?"
        last = pd.Timestamp(int(r["last"]), unit="s").date() if pd.notna(r["last"]) else "?"
        print(f"  {r['name'][:34]:34s} {r['kind']:6s} {r['friendly'][:34]:34s} {first} → {last} "
              f"{int(r['records']):>10,}")
    print("\nData types:")
    for t, info in sorted(meta.get("types", {}).items(), key=lambda kv: -kv[1]["count"]):
        first = pd.Timestamp(info["first"], unit="s").date() if info.get("first") else "?"
        last = pd.Timestamp(info["last"], unit="s").date() if info.get("last") else "?"
        print(f"  {info['label'][:40]:40s} {info['count']:>10,}  {first} → {last}")
    return 0


def cmd_export_csv(args) -> int:
    from .metrics import load_or_compute
    from .store import HealthStore

    ws = Workspace.at(args.home)
    out = Path(args.out) if args.out else ws.home / "csv"
    out.mkdir(parents=True, exist_ok=True)
    m = load_or_compute(HealthStore(ws.cache))
    m.daily.to_csv(out / "daily.csv", index_label="date")
    m.nights.to_csv(out / "nights.csv", index=False)
    m.workouts.to_csv(out / "workouts.csv", index=False)
    print(f"Wrote daily.csv, nights.csv and workouts.csv to {out}")
    print("These files contain your health data; keep them private.")
    return 0


def cmd_clear(args) -> int:
    ws = Workspace.at(args.home)
    if not args.yes:
        print(f"This deletes the imported data and reports in {ws.home}. Re-run with --yes to confirm.")
        return 1
    for p in (ws.cache, ws.reports, ws.home / "csv", ws.home / "demo"):
        if p.exists():
            shutil.rmtree(p)
            print(f"Deleted {p}")
    return 0


def _privacy_note(ws: Workspace) -> None:
    print(f"\nEverything stays in {ws.home} on this computer. The report files contain your health data;")
    print("share them only with people you trust.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="healthlens", description="Private, offline analysis of your Apple Health export.")
    p.add_argument("--version", action="version", version=f"healthlens {__version__}")
    p.add_argument("--home", help="Workspace folder for data and reports (default: ~/HealthLens or $HEALTHLENS_HOME)")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, export=False, units=True, open_flag=True):
        if export:
            sp.add_argument("export", help="export.zip from the Health app (or the unzipped folder / export.xml)")
        if units:
            sp.add_argument("--units", choices=("auto", "metric", "imperial"), default="auto",
                            help="Display units (default: whatever your iPhone used)")
        if open_flag:
            sp.add_argument("--no-open", action="store_true", help="Don't open the result in the browser")

    sp = sub.add_parser("run", help="Import an export and build the report and dashboard")
    common(sp, export=True)
    sp.add_argument("--force", action="store_true", help="Re-read the export even if it was imported already")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("demo", help="Try HealthLens on a synthetic person")
    common(sp)
    sp.add_argument("--years", type=int, default=8)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--force", action="store_true", help="Regenerate the demo data")
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("import", help="Only import (parse) an export")
    common(sp, export=True, units=False, open_flag=False)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_import)

    sp = sub.add_parser("report", help="Rebuild the HTML report from the last import")
    common(sp)
    sp.add_argument("--out", help="Where to write the report (default: <home>/reports/report.html)")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("dashboard", help="Rebuild the interactive dashboard from the last import")
    common(sp)
    sp.add_argument("--out", help="Where to write the dashboard (default: <home>/reports/dashboard.html)")
    sp.set_defaults(func=cmd_dashboard)

    sp = sub.add_parser("info", help="Show what is in the imported data")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("export-csv", help="Write daily, nights and workouts tables as CSV")
    sp.add_argument("--out", help="Folder (default: <home>/csv)")
    sp.set_defaults(func=cmd_export_csv)

    sp = sub.add_parser("clear", help="Delete imported data and reports")
    sp.add_argument("--yes", action="store_true", help="Confirm deletion")
    sp.set_defaults(func=cmd_clear)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
