"""Command entry points for the research pipeline."""

import argparse
from importlib import import_module

COMMANDS = {
    "prepare": "leaf_hierarchy.data.plantclef2015",
    "train": "leaf_hierarchy.training",
    "evaluate": "leaf_hierarchy.evaluation",
    "predict": "leaf_hierarchy.prediction",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Leaf classification research pipeline.")
    parser.add_argument("command", choices=COMMANDS, help="Pipeline stage; use COMMAND --help for options.")
    # Parse only the command so each stage owns its complete option validation.
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments[:1])
    return import_module(COMMANDS[args.command]).main(arguments[1:])
