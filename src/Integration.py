from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from common.artifacts import read_json, write_json
from common.config import get_paths

STAGES = ["collect", "preprocess", "embed", "encrypt", "evaluate", "security"]

REPORT_FILE = "integration_report.json"

# Files the laptop needs from the Colab run (the Backend reads exactly these).
BUNDLE_FILES = ["eval_results.json", "security_results.json", "splits.csv", "embeddings.npy",
                "embeddings_index.csv", "pipeline_config.json"]


def stage_command(stage: str, args: argparse.Namespace) -> List[str]:
    """The command line for one stage. Options are only passed when the user set them."""
    py = [sys.executable, "-m"]
    if stage == "collect":
        cmd = py + ["src.Data_Collection"]
        if args.zip:
            cmd += ["--zip", str(args.zip)]
        if args.dir:
            cmd += ["--dir", str(args.dir)]
        return cmd
    if stage == "preprocess":
        return py + ["src.Data_Preprocessing"] + (["--subset", str(args.subset)] if args.subset else [])
    if stage == "embed":
        return py + ["src.Embedding_Generation"] + (["--fp16"] if args.fp16 else [])
    if stage == "encrypt":
        return py + ["src.Embedding_Encryption"]
    if stage == "evaluate":
        return py + ["src.Testing_Evaluation", "evaluate", "--max-probes", str(args.max_probes)]
    if stage == "security":
        return py + ["src.Security_Simulation", "--victims", str(args.victims)]
    raise ValueError(f"unknown stage {stage!r}")


def run_stage(stage: str, args: argparse.Namespace) -> Dict:
    cmd = stage_command(stage, args)
    print(f"\n=== {stage}: {' '.join(cmd[2:])}", flush=True)
    t0 = time.perf_counter()
    code = subprocess.call(cmd)
    return {"stage": stage, "command": " ".join(cmd[1:]), "exit_code": code,
            "seconds": round(time.perf_counter() - t0, 1)}


def run_pipeline(args: argparse.Namespace) -> int:
    first, last = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
    if first > last:
        raise SystemExit("--from-stage must not come after --to-stage")
    paths = get_paths().ensure()
    report: Dict = {"subset": args.subset, "stages": []}
    for stage in STAGES[first:last + 1]:
        result = run_stage(stage, args)
        report["stages"].append(result)
        write_json(paths.file(REPORT_FILE), report)
        if result["exit_code"] != 0:
            print(f"\nStage '{stage}' failed (exit {result['exit_code']}); stopping.")
            return result["exit_code"]
    print(f"\n{'stage':<12}{'seconds':>10}")
    for r in report["stages"]:
        print(f"{r['stage']:<12}{r['seconds']:>10}")
    print(f"Report: {paths.file(REPORT_FILE)}")
    return 0

def make_bundle(out: Optional[Path] = None) -> Path:
    """Copy only what the laptop demo needs. Keys and the template database are NOT included:
    the demo builds its own gallery and key on the laptop."""
    paths = get_paths()
    out = out or paths.file("laptop_bundle")
    out.mkdir(parents=True, exist_ok=True)
    missing = [n for n in BUNDLE_FILES if not paths.file(n).exists()]
    if missing:
        raise SystemExit(f"Missing from {paths.artifacts}: {', '.join(missing)}. Run the pipeline first.")
    for name in BUNDLE_FILES:
        shutil.copy2(paths.file(name), out / name)
    print(f"Bundle written to {out}\nCopy these files into the laptop's work\\artifacts folder.")
    return out


def launch_demo(args: argparse.Namespace) -> int:
    base = [sys.executable, "-m", "src.Backend"]
    build = base + ["build-demo"] + (["--celeba-dir", str(args.celeba_dir)] if args.celeba_dir else [])
    if not args.skip_build:
        code = subprocess.call(build)
        if code != 0:
            return code
    return subprocess.call(base + ["serve"])


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Run and package the whole pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run pipeline stages in order")
    r.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    r.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    r.add_argument("--subset", type=int, default=None, help="number of identities (SUBSET switch)")
    r.add_argument("--zip", type=Path)
    r.add_argument("--dir", type=Path)
    r.add_argument("--fp16", action="store_true")
    r.add_argument("--max-probes", type=int, default=20000)
    r.add_argument("--victims", type=int, default=200)
    b = sub.add_parser("bundle", help="collect the files the laptop demo needs")
    b.add_argument("--out", type=Path)
    d = sub.add_parser("demo", help="build the demo gallery and start the dashboard")
    d.add_argument("--celeba-dir", type=Path)
    d.add_argument("--skip-build", action="store_true")
    args = p.parse_args(argv)
    if args.command == "run":
        return run_pipeline(args)
    if args.command == "bundle":
        make_bundle(args.out)
        return 0
    return launch_demo(args)


if __name__ == "__main__":
    sys.exit(main())
