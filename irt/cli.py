"""
cli.py — the command-line entry point for the Hybrid IRT engine.

Responsibility
--------------
This module contains NO pipeline logic of its own. Its only job is to:

  1. Parse command-line arguments (argparse) into a choice of data
     source (`--csv`, `--postgres`, or `--demo`) plus a couple of
     run-scoping options (`--student`, `--verbose`).
  2. Construct the matching `irt.repository.IRTRepository` implementation
     (`CSVRepository` or `PostgresRepository`) — this module is the only
     place that turns a shell flag into a repository choice; it never
     talks to a CSV file or a database connection itself.
  3. Call `irt.service.run_pipeline()` — the one function that already
     runs the complete pipeline (Feature Builder -> Clustering ->
     Segregation -> Question Parameters -> Theta -> Mastery Initializer)
     end-to-end — and print its result in a readable report.
  4. Translate the exceptions `repository.py` and `service.py` already
     define (`RepositoryError`, `ItemBankBuildError`) into a clear
     stderr message and a non-zero process exit code, rather than a raw
     Python traceback.

If a bug ever produces a wrong theta, a wrong mastery value, or a wrong
repository query, the fix belongs in `theta.py`, `mastery_initializer.py`,
or `repository.py` — not here. This module can only be wrong about
argument parsing, wiring, or reporting.

Why three data-source flags instead of one `--source` option
---------------------------------------------------------------------
`--csv DATA_DIR`, `--postgres [DATABASE_URL]`, and `--demo` are mutually
exclusive (argparse enforces this directly) because exactly one data
source is meaningful per run — there is no scenario where a caller wants
both a CSV folder and a live Postgres connection scored in the same
invocation. `--demo` is kept separate from `--csv` (even though
`--demo` is implemented as `CSVRepository.from_default_sample_data()`,
i.e. a CSV run under the hood) specifically so a brand-new user, a
supervisor, or a viva examiner can run `python -m irt --demo` with zero
setup and zero knowledge of where `sample_data/` lives on disk.

How this fits into the Hybrid IRT architecture
---------------------------------------------------------------------
    shell
        -> python -m irt ...                  (irt/__main__.py)
            -> cli.main(argv)
                -> cli.build_parser().parse_args(argv)
                -> cli._build_repository(args)
                    -> repository.CSVRepository / repository.PostgresRepository
                -> service.run_pipeline(repo, student_ids=args.student)
                    -> CohortPipelineResult
                -> cli._print_report(result)
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from .repository import CSVRepository, IRTRepository, PostgresRepository, RepositoryError
from .service import CohortPipelineResult, ItemBankBuildError, run_pipeline

PROG = "python -m irt"

# Process exit codes. Kept as named constants (not bare 0/1/2 scattered
# through the function bodies below) so a test — or a future caller
# shelling out to this CLI — can assert on *why* a run failed, not just
# that it failed.
EXIT_OK = 0
EXIT_DATA_SOURCE_ERROR = 1  # the chosen repository couldn't be reached/read at all
EXIT_PIPELINE_ERROR = 2  # the repository was reachable, but the item bank couldn't be built


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Kept as its own function (rather than
    inlined into main()) so tests can inspect/parse against it directly
    without going through main()'s side effects (repository construction,
    printing, process exit codes).
    """
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Run the complete Synapse Hybrid IRT pipeline end-to-end: "
            "Feature Builder -> Normalization -> KMeans Clustering -> "
            "Segregation -> Question IRT Parameters -> Theta Estimation "
            "-> Mastery Initialization."
        ),
        epilog=(
            "examples:\n"
            "  %(prog)s --demo\n"
            "  %(prog)s --demo --student S1 --student S2\n"
            "  %(prog)s --csv sample_data\n"
            "  %(prog)s --csv sample_data --verbose\n"
            "  %(prog)s --postgres\n"
            "  %(prog)s --postgres postgresql://user:pass@host:5432/synapse\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--csv",
        metavar="DATA_DIR",
        help=(
            "Run the pipeline against CSV files in DATA_DIR (expects "
            "students.csv, questions.csv, responses.csv — see "
            "irt.repository.CSVRepository). Use this to run against a "
            "different dataset than the bundled sample_data/."
        ),
    )
    source.add_argument(
        "--postgres",
        nargs="?",
        const="",
        metavar="DATABASE_URL",
        help=(
            "Run the pipeline against a Postgres database, via "
            "irt.repository.PostgresRepository. DATABASE_URL is optional: "
            "if omitted, it is resolved from backend/.env or the "
            "DATABASE_URL environment variable "
            "(irt.config.load_database_url)."
        ),
    )
    source.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run the pipeline against this repository's own bundled "
            "sample data (sample_data/), via "
            "CSVRepository.from_default_sample_data(). The quickest way "
            "to see the whole pipeline run end-to-end with no setup."
        ),
    )

    parser.add_argument(
        "--student",
        "-s",
        metavar="STUDENT_ID",
        action="append",
        default=None,
        help=(
            "Restrict scoring to this student_id. May be repeated "
            "(--student S1 --student S2) to score more than one specific "
            "student. If omitted, every student the data source knows "
            "about is scored."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help=(
            "Print the full item-bank report (strong/weak cluster "
            "membership, flagged discriminators) and every non-fatal "
            "warning collected during the run, in addition to the "
            "per-student summary."
        ),
    )

    return parser


def _build_repository(args: argparse.Namespace) -> IRTRepository:
    """Turn the parsed data-source flag into an IRTRepository instance.
    Exactly one of args.demo / args.csv / args.postgres is set, per the
    mutually-exclusive-and-required argparse group in build_parser() —
    see that function's docstring for why `--postgres` used `nargs="?"`
    with `const=""` rather than `action="store_true"`.
    """
    if args.demo:
        return CSVRepository.from_default_sample_data()
    if args.csv is not None:
        return CSVRepository(args.csv)
    # args.postgres is not None here (it's "" if bare --postgres was
    # given, or the DATABASE_URL string if one was); "" is falsy, so
    # `or None` lets PostgresRepository fall back to
    # config.load_database_url()'s own .env / environment-variable
    # resolution exactly as if no override had been passed at all.
    database_url = args.postgres or None
    return PostgresRepository(database_url=database_url)


def _format_theta(theta_result) -> str:
    se = (
        f"{theta_result.standard_error:.3f}"
        if theta_result.standard_error is not None
        else "n/a"
    )
    return (
        f"theta={theta_result.theta:+.3f} "
        f"(converged={theta_result.converged}, SE={se}, "
        f"n_responses={theta_result.n_responses})"
    )


def _print_report(result: CohortPipelineResult, *, verbose: bool) -> None:
    bank = result.item_bank

    print("=" * 60)
    print("Synapse Hybrid IRT Engine — pipeline run")
    print("=" * 60)
    print(f"Students scored:               {len(result.scored_student_ids())}")
    print(f"Students skipped:              {len(result.skipped_students)}")
    print(f"Questions with IRT parameters: {len(bank.parameters)}")
    print()

    if verbose:
        print("Item bank")
        print("-" * 60)
        print(f"Strong cluster: {bank.cluster_result.strong_student_ids()}")
        print(f"Weak cluster:   {bank.cluster_result.weak_student_ids()}")
        flagged = bank.segregation_batch.flagged()
        if flagged:
            flagged_ids = [r.question_id for r in flagged]
            print(f"Flagged (poor/negative) discriminators: {flagged_ids}")
        print("-" * 60)
        print()

    print("Per-student results")
    print("-" * 60)
    for sid in result.scored_student_ids():
        r = result.result_for(sid)
        s = r.mastery_result.summary
        print(f"{sid}  [{r.cluster_label}]")
        print(f"  {_format_theta(r.theta_result)}")
        print(
            f"  mastery: {s.n_concepts} concept(s), "
            f"avg={s.average_initial_mastery:.3f}, "
            f"lowest={s.lowest_mastery_concept_id}({s.lowest_mastery_value:.3f}), "
            f"highest={s.highest_mastery_concept_id}({s.highest_mastery_value:.3f})"
        )
    print("-" * 60)

    if result.skipped_students:
        print()
        print("Skipped students")
        print("-" * 60)
        for skipped in result.skipped_students:
            print(f"  {skipped.student_id}: {skipped.reason}")
        print("-" * 60)

    if verbose:
        warnings = result.warnings()
        if warnings:
            print()
            print("Warnings")
            print("-" * 60)
            for w in warnings:
                print(f"  {w}")
            print("-" * 60)
    elif result.warnings():
        print()
        print(f"({len(result.warnings())} warning(s) — re-run with --verbose to see them.)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code rather than calling
    sys.exit() itself, so tests (and irt/__main__.py) can decide how to
    surface it. `argv` defaults to `sys.argv[1:]` (argparse's own
    default) when omitted — pass an explicit list in tests.

    Exit codes
    ----------
    0  success (EXIT_OK) — the pipeline ran; individual skipped students
       are reported, not treated as a failure (mirrors
       service.run_pipeline()'s never-abort-the-cohort design).
    1  (EXIT_DATA_SOURCE_ERROR) the chosen data source could not be
       constructed or reached at all — e.g. a missing --csv directory, a
       missing psycopg2 install, or an unreachable Postgres server.
    2  (EXIT_PIPELINE_ERROR) the data source was reachable, but the
       cohort-level pipeline stages could not produce a usable item bank
       — e.g. fewer than 2 students in the data source.

    Argument-parsing errors (missing/conflicting flags, `--help`) are
    handled by argparse itself, which prints its own usage message and
    raises SystemExit — that exception is intentionally left to
    propagate rather than caught here, matching standard CLI behavior.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        repo = _build_repository(args)
    except RepositoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DATA_SOURCE_ERROR

    with repo:
        try:
            result = run_pipeline(repo, student_ids=args.student)
        except ItemBankBuildError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_PIPELINE_ERROR
        except RepositoryError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_DATA_SOURCE_ERROR

        _print_report(result, verbose=args.verbose)

    return EXIT_OK


__all__ = [
    "PROG",
    "EXIT_OK",
    "EXIT_DATA_SOURCE_ERROR",
    "EXIT_PIPELINE_ERROR",
    "build_parser",
    "main",
]
