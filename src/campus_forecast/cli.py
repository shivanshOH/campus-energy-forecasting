"""Command-line entry points."""

from __future__ import annotations

import argparse

from campus_forecast.data.audit import audit_directory
from campus_forecast.demo import run_demo
from campus_forecast.schema import (
    EnergyMeasurementConfig,
    IrregularIntervalPolicy,
    MeasurementSemantics,
    MeasurementUnit,
)
from campus_forecast.utils import json_dump


def audit(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit an I-BLEND-like source directory")
    parser.add_argument("--root", required=True, help="Dataset directory")
    parser.add_argument("--output", default="artifacts/reports/data-availability.json")
    parser.add_argument("--source-unit", choices=[unit.value for unit in MeasurementUnit])
    parser.add_argument("--semantics", choices=[item.value for item in MeasurementSemantics])
    parser.add_argument(
        "--target-unit", choices=[MeasurementUnit.WH.value, MeasurementUnit.KWH.value], default="Wh"
    )
    parser.add_argument("--source-column")
    parser.add_argument("--expected-interval-minutes", type=float)
    args = parser.parse_args(argv)
    config = None
    if args.source_unit or args.semantics:
        if not args.source_unit or not args.semantics:
            parser.error("--source-unit and --semantics must be supplied together")
        config = EnergyMeasurementConfig(
            source_unit=MeasurementUnit(args.source_unit),
            semantics=MeasurementSemantics(args.semantics),
            target_unit=MeasurementUnit(args.target_unit),
            source_column=args.source_column,
            expected_interval_minutes=args.expected_interval_minutes,
            irregular_interval_policy=IrregularIntervalPolicy.USE_ACTUAL_INTERVAL,
        )
    result = audit_directory(args.root, energy_config=config)
    path = json_dump(result, args.output)
    print(f"Audit written to {path}")


def demo(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic software fixture")
    parser.add_argument("--output", default="artifacts/reports/demo-results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    result = run_demo(seed=args.seed)
    path = json_dump(result, args.output)
    print(f"DEMO DATA written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m campus_forecast.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--root", required=True)
    audit_parser.add_argument("--output", default="artifacts/reports/data-availability.json")
    audit_parser.add_argument("--source-unit", choices=[unit.value for unit in MeasurementUnit])
    audit_parser.add_argument("--semantics", choices=[item.value for item in MeasurementSemantics])
    audit_parser.add_argument(
        "--target-unit", choices=[MeasurementUnit.WH.value, MeasurementUnit.KWH.value], default="Wh"
    )
    audit_parser.add_argument("--source-column")
    audit_parser.add_argument("--expected-interval-minutes", type=float)
    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("--output", default="artifacts/reports/demo-results.json")
    demo_parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "audit":
        audit_args = ["--root", args.root, "--output", args.output]
        for name in ("source_unit", "semantics", "source_column", "expected_interval_minutes"):
            value = getattr(args, name)
            if value is not None:
                audit_args.extend([f"--{name.replace('_', '-')}", str(value)])
        if args.target_unit:
            audit_args.extend(["--target-unit", args.target_unit])
        audit(audit_args)
    else:
        demo_args = ["--output", args.output, "--seed", str(args.seed)]
        demo(demo_args)


if __name__ == "__main__":
    main()
