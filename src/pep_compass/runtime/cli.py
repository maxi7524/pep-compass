"""Command-line interface for PepCompass runtime execution."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from pep_compass.runtime.backends import (
    execute_subprocess_plan,
    write_slurm_array_script,
)
from pep_compass.runtime.configuration import load_runtime_configuration
from pep_compass.runtime.output import ResultWriter
from pep_compass.runtime.planning.plan import materialize_execution_plan
from pep_compass.runtime.planning.validation import validate_execution_plan
from pep_compass.runtime.runner import RuntimeRunner
from pep_compass.runtime.workflows import ComposableWorkflow


def _parser() -> argparse.ArgumentParser:
    """Build the PepCompass command-line parser."""
    # main parser
    parser = argparse.ArgumentParser(prog="pep-compass")
    ## Subparser: runner
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Execute a PepCompass configuration.")
    run.add_argument("configuration", type=Path)
    run.add_argument("--working-directory", type=Path, default=Path.cwd())
    run.add_argument("--backend", choices=("local", "subprocess", "slurm"))
    run.add_argument("--max-workers", type=int)
    run.add_argument("--run-index", type=int, action="append")
    run.add_argument("--resume", action="store_true", default=None)
    run.add_argument("--continue-on-error", action="store_true", default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--slurm-script", type=Path)
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    #TODO  Dodać subparser do PoGS'a 
    ## Subparser - PoGS  
    return parser


def main() -> None:
    """Load configuration, materialize a plan and dispatch its backend."""
    # Load configuration
    ## Read from parser
    arguments = _parser().parse_args()
    if arguments.command != "run":
        raise ValueError(f"Unsupported command: {arguments.command}")
    working_directory = arguments.working_directory.resolve()
    configuration_path = arguments.configuration.resolve()
    ## Load configuration
    ## REMARK: it reads files properly, and validate config  
    configuration = load_runtime_configuration(configuration_path)
    execution = configuration.execution
    ## Replace with parsed arguments 
    execution = replace(
        execution,
        backend=arguments.backend or execution.backend,
        max_workers=arguments.max_workers or execution.max_workers,
        resume=execution.resume if arguments.resume is None else arguments.resume,
        continue_on_error=(
            execution.continue_on_error
            if arguments.continue_on_error is None
            else arguments.continue_on_error
        ),
    )
    configuration = replace(configuration, execution=execution)
    ## Create execution plan (split files etc.) 
    plan = materialize_execution_plan(
        configuration,
        working_directory=working_directory,
    ).select(set(arguments.run_index) if arguments.run_index else None)
    ## Final validation 
    validate_execution_plan(plan)

    # Execution type
    ## Case: dry-run
    if arguments.dry_run:
        for entry in plan.entries:
            print(
                f"{entry.index:05d} {entry.run_id} {entry.variant.variant_id} "
                f"{entry.task.task_id} seed={entry.seed}"
            )
        return
    ## Case: subprocess
    if execution.backend == "subprocess" and not arguments.worker:
        execute_subprocess_plan(
            plan,
            configuration_path,
            working_directory=working_directory,
            max_workers=execution.max_workers,
        )
        return
    ## Case: slurm 
    if execution.backend == "slurm" and not arguments.worker:
        if arguments.slurm_script is None:
            raise ValueError("--slurm-script is required for the Slurm backend.")
        write_slurm_array_script(
            plan,
            configuration_path,
            arguments.slurm_script,
            working_directory=working_directory,
            settings=execution.slurm,
        )
        return

    # Create | Load output directory 
    output = configuration.experiment.output_directory
    writer = None
    if output is not None:
        if not output.is_absolute():
            output = working_directory / output
        writer = ResultWriter(output)

    # Final workflow
    ## workflow creation
    workflow = ComposableWorkflow(configuration.autoencoder)
    ## workflow excecitno 
    executions = RuntimeRunner(configuration, workflow, writer).run(plan)
    if any(execution.status == "failed" for execution in executions):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
