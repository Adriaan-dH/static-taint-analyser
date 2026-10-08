"""Test the command-line interface and build script."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tests" / "testcases"


class CliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["./main", *arguments], cwd=ROOT, capture_output=True,
            text=True, timeout=5,
        )

    def assert_success(self, result: subprocess.CompletedProcess, expected: bool) -> None:
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        output = json.loads(result.stdout)
        self.assertIsInstance(output, dict)
        self.assertEqual(set(output), {"reaches"})
        self.assertIs(type(output["reaches"]), bool)
        self.assertIs(output["reaches"], expected)
        self.assertEqual(result.stdout, json.dumps(output) + "\n")

    def assert_failure(self, result: subprocess.CompletedProcess, diagnostic: str) -> None:
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn(diagnostic, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def copy_inputs(self, directory: Path) -> None:
        for filename in ("graph.json", "test_case.json"):
            shutil.copyfile(CASES / "test1" / filename, directory / filename)

    def test_build(self) -> None:
        result = subprocess.run(
            ["./build.sh"], cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertTrue((ROOT / "main").stat().st_mode & 0o111)

    def test_all_real_fixtures(self) -> None:
        cases = sorted(path for path in CASES.iterdir() if path.is_dir())
        self.assertEqual(len(cases), 63)
        for case in cases:
            with self.subTest(case=case.name):
                expected = json.loads((case / "test_case.json").read_text())["reaches"]
                self.assertIs(type(expected), bool)
                start = time.perf_counter()
                result = self.run_cli("-i", str(case))
                elapsed = time.perf_counter() - start
                self.assert_success(result, expected)
                self.assertLess(elapsed, 5)

    def test_only_json_inputs_are_needed_and_expected_result_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.copy_inputs(directory)
            metadata_path = directory / "test_case.json"
            metadata = json.loads(metadata_path.read_text())
            for expectation in (False, "not a boolean", None):
                with self.subTest(expectation=expectation):
                    metadata["reaches"] = expectation
                    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                    self.assert_success(self.run_cli("-i", str(directory)), True)
            del metadata["reaches"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assert_success(self.run_cli("-i", str(directory)), True)

    def test_missing_argument(self) -> None:
        result = self.run_cli()
        self.assert_failure(result, "-i")
        self.assertEqual(result.returncode, 2)

    def test_invalid_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assert_failure(self.run_cli("-i", str(Path(temporary) / "missing")), "directory")
            file_path = Path(temporary) / "file"
            file_path.write_text("", encoding="utf-8")
            self.assert_failure(self.run_cli("-i", str(file_path)), "directory")

    def test_missing_input_files(self) -> None:
        for filename in ("graph.json", "test_case.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.copy_inputs(directory)
                (directory / filename).unlink()
                self.assert_failure(self.run_cli("-i", str(directory)), filename)

    def test_malformed_json(self) -> None:
        for filename in ("graph.json", "test_case.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.copy_inputs(directory)
                (directory / filename).write_text("{broken JSON", encoding="utf-8")
                self.assert_failure(self.run_cli("-i", str(directory)), "Expecting")

    def test_invalid_graph_and_metadata(self) -> None:
        for filename in ("graph.json", "test_case.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.copy_inputs(directory)
                (directory / filename).write_text("{}", encoding="utf-8")
                self.assert_failure(self.run_cli("-i", str(directory)), "missing required field")

    def test_analysis_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.copy_inputs(directory)
            path = directory / "test_case.json"
            metadata = json.loads(path.read_text())
            metadata["sourceNode"] = -1
            path.write_text(json.dumps(metadata), encoding="utf-8")
            result = self.run_cli("-i", str(directory))
            self.assert_failure(result, "Unknown node ID -1")
            self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()
