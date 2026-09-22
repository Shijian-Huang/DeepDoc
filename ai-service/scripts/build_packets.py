#!/usr/bin/env python3
"""Build evaluator evidence packets from one document or a directory.

Example configuration file::

    {
      "input": "papers",
      "output": "evaluation/default_packets.json",
      "summary_mode": "standard"
    }

Paths in a configuration file are resolved relative to that file.
Command-line values override their configuration-file equivalents.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


AI_SERVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_DIR))

from parser.document_parser import SUPPORTED_EXTENSIONS, parse_document_pages  # noqa: E402
from utils.section_extractor import build_summary_input_from_pages  # noqa: E402


SUMMARY_MODES = ("paragraph", "standard", "one_page")


class PacketBuildError(RuntimeError):
    """Raised when evidence packets cannot be built from the supplied input."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build evaluator evidence packets with "
            "build_summary_input_from_pages."
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="A supported document or a directory containing documents.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "JSON object containing input and optional output/summary_mode fields. "
            "Relative paths are resolved from the config file's directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write packets to this JSON file instead of stdout.",
    )
    parser.add_argument(
        "--summary-mode",
        choices=SUMMARY_MODES,
        help="Excerpt size preset (default: standard).",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise PacketBuildError(f"Could not read config file {config_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise PacketBuildError(f"Config file is not valid JSON: {config_path}") from error

    if not isinstance(config, dict):
        raise PacketBuildError("The config file must contain a JSON object.")

    allowed_fields = {"input", "output", "summary_mode"}
    unknown_fields = sorted(set(config) - allowed_fields)
    if unknown_fields:
        raise PacketBuildError(
            f"Unknown config field(s): {', '.join(unknown_fields)}"
        )

    for field in ("input", "output", "summary_mode"):
        if field in config and not isinstance(config[field], str):
            raise PacketBuildError(f"Config field '{field}' must be a string.")

    config["_base_dir"] = config_path.parent
    return config


def _configured_path(config: dict[str, Any], field: str) -> Path | None:
    value = config.get(field)
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config["_base_dir"] / path
    return path.resolve()


def resolve_options(args: argparse.Namespace) -> tuple[Path, Path | None, str]:
    config = load_config(args.config) if args.config else {}

    input_path = (
        args.input.expanduser().resolve()
        if args.input
        else _configured_path(config, "input")
    )
    if input_path is None:
        raise PacketBuildError("An input file or directory is required.")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else _configured_path(config, "output")
    )
    summary_mode = args.summary_mode or config.get("summary_mode", "standard")
    if summary_mode not in SUMMARY_MODES:
        raise PacketBuildError(
            f"Invalid summary_mode '{summary_mode}'; choose from {', '.join(SUMMARY_MODES)}."
        )
    return input_path, output_path, summary_mode


def discover_documents(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise PacketBuildError(
                f"Unsupported input format '{input_path.suffix}' for {input_path}."
            )
        return [input_path]

    if input_path.is_dir():
        documents = sorted(
            (
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=lambda path: path.name.casefold(),
        )
        if not documents:
            extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise PacketBuildError(
                f"No supported documents found in {input_path} ({extensions})."
            )
        return documents

    raise PacketBuildError(f"Input path does not exist: {input_path}")


def build_packets(
    input_path: Path,
    summary_mode: str = "standard",
) -> list[dict[str, str]]:
    if summary_mode not in SUMMARY_MODES:
        raise PacketBuildError(
            f"Invalid summary_mode '{summary_mode}'; choose from {', '.join(SUMMARY_MODES)}."
        )

    packets: list[dict[str, str]] = []
    for document_path in discover_documents(input_path):
        try:
            pages = parse_document_pages(str(document_path))
        except Exception as error:
            raise PacketBuildError(f"Could not parse {document_path}: {error}") from error
        if not pages:
            raise PacketBuildError(f"No text could be extracted from {document_path}.")

        packet_text, _selected_sections, _evidence_sources = (
            build_summary_input_from_pages(pages, summary_mode=summary_mode)
        )
        packets.append({"id": document_path.name, "text": packet_text})
    return packets


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        input_path, output_path, summary_mode = resolve_options(args)
        packets = build_packets(input_path, summary_mode)
        payload = json.dumps(packets, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload, encoding="utf-8")
            print(f"Evidence packets: {output_path}")
        else:
            print(payload, end="")
    except (PacketBuildError, OSError) as error:
        print(f"Packet build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
