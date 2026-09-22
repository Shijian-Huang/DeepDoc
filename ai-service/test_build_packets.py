import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.build_packets import (
    PacketBuildError,
    build_packets,
    main,
)


class BuildDefaultPacketsTests(unittest.TestCase):
    def test_single_file_uses_filename_as_packet_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            document = Path(temp_dir) / "paper.txt"
            document.write_text(
                "Abstract\nA useful finding.\n\nIntroduction\nThe research problem.",
                encoding="utf-8",
            )

            packets = build_packets(document)

            self.assertEqual(len(packets), 1)
            self.assertEqual(packets[0]["id"], "paper.txt")
            self.assertIn("[SOURCE_ID: abstract_p1_01]", packets[0]["text"])
            self.assertIn("Section: abstract", packets[0]["text"])

    def test_directory_is_sorted_and_ignores_unsupported_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "zeta.txt").write_text(
                "Some useful paper text.", encoding="utf-8"
            )
            (directory / "Alpha.md").write_text(
                "# Paper\n\nSome markdown text.", encoding="utf-8"
            )
            (directory / "notes.csv").write_text("ignored", encoding="utf-8")

            packets = build_packets(directory)

            self.assertEqual(
                [packet["id"] for packet in packets],
                ["Alpha.md", "zeta.txt"],
            )

    def test_config_paths_are_relative_to_config_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "paper.txt").write_text("Paper content.", encoding="utf-8")
            config = directory / "packets.json"
            config.write_text(
                json.dumps({
                    "input": "paper.txt",
                    "output": "output/default_packets.json",
                    "summary_mode": "paragraph",
                }),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                result = main(["--config", str(config)])

            output = directory / "output" / "default_packets.json"
            self.assertEqual(result, 0)
            self.assertTrue(output.is_file())
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))[0]["id"],
                "paper.txt",
            )

    def test_directory_without_documents_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(PacketBuildError):
                build_packets(Path(temp_dir))


if __name__ == "__main__":
    unittest.main()
