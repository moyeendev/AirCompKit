"""Command-line interface (FEATURES C7): ``list``, ``run``, ``train``, ``reproduce``, ``bench``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from aircompkit import __version__
from aircompkit.core.registry import KINDS, available


def _cmd_list(args: argparse.Namespace) -> int:
    kinds = [args.kind] if args.kind else list(KINDS)
    for kind in kinds:
        names = available(kind)
        print(f"{kind}: {', '.join(names) if names else '(none yet)'}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from aircompkit.core.scenario import Scenario, load_yaml

    cfg = load_yaml(args.config)
    run_opts = dict(cfg.get("run", {}) or {})
    for key in ("trials", "dim", "batch"):
        if getattr(args, key) is not None:
            run_opts[key] = getattr(args, key)
    res = Scenario.from_dict(cfg).run(**run_opts)
    for name, (m, ci) in res.summary().items():
        print(f"{name:>14}: {m:.6g} +/- {ci:.2g}  (95% CI)")
    if args.out:
        npz, js = res.save(args.out)
        print(f"saved {npz} and {js}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from aircompkit.core.scenario import Scenario, load_yaml
    from aircompkit.learning import FederatedConfig, train

    cfg = load_yaml(args.config)
    if "learning" not in cfg:
        raise SystemExit(f"{args.config}: needs a 'learning:' section")
    learning = dict(cfg["learning"])
    if args.rounds is not None:
        learning["rounds"] = args.rounds
    res = train(Scenario.from_dict(cfg), FederatedConfig(**learning),
                data_root=args.data_root, verbose=True)
    print(f"final test accuracy {res['test_acc'][-1]:.4f}, "
          f"mean aggregation NMSE {res['agg_nmse'].mean():.3g}")
    if args.out:
        npz, js = res.save(args.out)
        print(f"saved {npz} and {js}")
    return 0


def _cmd_reproduce(args: argparse.Namespace) -> int:
    import inspect
    from pathlib import Path

    from aircompkit.core.registry import get
    from aircompkit.reproduce import CHECKS, report

    if args.all:
        names = available("reproduction")
    elif args.name:
        names = [args.name]
    else:
        print("available reproductions: " + ", ".join(available("reproduction")))
        return 0
    failed, skipped = [], []
    for name in names:
        fn = get("reproduction", name)
        out = None if args.out is None else (Path(args.out) / name if args.all else args.out)
        kw = {"trials": args.trials, "quick": args.quick, "out": out}
        if args.data_root is not None:
            if "data_root" in inspect.signature(fn).parameters:
                kw["data_root"] = args.data_root
            elif not args.all:
                raise SystemExit(f"{name} uses no dataset; drop --data-root")
        if args.all:
            print(f"\n=== {name} ===", flush=True)
        try:
            res = fn(**kw)
        except FileNotFoundError as e:
            if not args.all:
                raise
            skipped.append(name)
            print(f"skipped {name}: {str(e).splitlines()[0]}")
            continue
        if args.check:
            if name not in CHECKS:
                raise SystemExit(f"no checks defined for {name}")
            if not report(name, res):
                failed.append(name)
    if args.all:
        print(f"\nreproductions: {len(names) - len(skipped)} run, {len(skipped)} skipped"
              + (f" ({', '.join(skipped)}: dataset not found)" if skipped else "")
              + (f", checks failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from aircompkit import bench

    bench.run(groups=args.groups or tuple(bench.GROUPS), quick=args.quick, repeat=args.repeat,
              out=args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aircompkit",
        description="Security-aware over-the-air computation simulator.",
    )
    p.add_argument("--version", action="version", version=f"aircompkit {__version__}")
    sub = p.add_subparsers(dest="command")
    pl = sub.add_parser("list", help="list registered components")
    pl.add_argument("kind", nargs="?", choices=KINDS, help="only this component kind")
    pl.set_defaults(func=_cmd_list)
    pr = sub.add_parser("run", help="run a scenario from a YAML config")
    pr.add_argument("config", help="YAML scenario file")
    pr.add_argument("--trials", type=int, help="override run.trials")
    pr.add_argument("--dim", type=int, help="override run.dim (0 = analytical MSE only)")
    pr.add_argument("--batch", type=int, help="override run.batch")
    pr.add_argument("--out", help="save results to OUT.npz / OUT.json")
    pr.set_defaults(func=_cmd_run)
    pt = sub.add_parser("train", help="run OTA-FL from a YAML config with a 'learning:' section")
    pt.add_argument("config")
    pt.add_argument("--rounds", type=int, help="override learning.rounds")
    pt.add_argument("--data-root", help="folder with mnist.npz etc. (else $AIRCOMPKIT_DATA)")
    pt.add_argument("--out", help="save results to OUT.npz / OUT.json")
    pt.set_defaults(func=_cmd_train)
    pp = sub.add_parser("reproduce", help="reproduce a published result (no name: list)")
    pp.add_argument("name", nargs="?")
    pp.add_argument("--all", action="store_true", help="run every reproduction (OUT/NAME/ each)")
    pp.add_argument("--quick", action="store_true", help="few trials (smoke test)")
    pp.add_argument("--trials", type=int)
    pp.add_argument("--out", help="directory for JSON results and figure")
    pp.add_argument("--data-root", help="folder with mnist.npz (else $AIRCOMPKIT_DATA)")
    pp.add_argument("--check", action="store_true",
                    help="compare with the published values; exit 1 if a tolerance fails")
    pp.set_defaults(func=_cmd_reproduce)
    pb = sub.add_parser("bench", help="performance benchmarks")
    pb.add_argument("groups", nargs="*", choices=["mse", "receiver", "fl"],
                    help="benchmark groups (default: all)")
    pb.add_argument("--quick", action="store_true", help="small sizes (smoke test)")
    pb.add_argument("--repeat", type=int, default=3, help="best of REPEAT runs")
    pb.add_argument("--out", help="directory for bench.json and bench.md")
    pb.set_defaults(func=_cmd_bench)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
