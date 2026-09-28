"""Focused tests for JSON input parsing. No analysis is performed."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from src.graph import (
    GraphEdge,
    GraphNode,
    ProgramGraph,
    TestMetadata,
    load_graph,
    load_test_case,
    load_test_metadata,
)


class GraphParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        self.graph_path = self.directory / "graph.json"
        self.metadata_path = self.directory / "test_case.json"
        # Larger than a floating-point integer can represent exactly.
        self.source_id = 2**60 + 1
        self.sink_id = self.source_id + 1
        self.graph_data = {
            "nodes": [
                {"id": self.source_id, "kind": "PARAMETER", "value": "x", "astOrder": 7},
                {"id": self.sink_id, "kind": "IDENTIFIER", "value": "x", "astOrder": 2},
            ],
            "edges": [
                {"source": self.source_id, "destination": self.sink_id, "kind": "AST"},
                {"source": self.source_id, "destination": self.sink_id, "kind": "CFG"},
            ],
        }
        self.metadata_data = {
            "sourceNode": self.source_id,
            "sinkNode": self.sink_id,
            "reaches": True,
        }
        self.write_json(self.graph_path, self.graph_data)
        self.write_json(self.metadata_path, self.metadata_data)

    def write_json(self, path: Path, data: object) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_graph_node_fields_and_order_are_preserved(self) -> None:
        graph = load_graph(self.graph_path)
        self.assertIsInstance(graph, ProgramGraph)
        self.assertEqual(
            graph.nodes,
            [
                GraphNode(self.source_id, "PARAMETER", "x", 7),
                GraphNode(self.sink_id, "IDENTIFIER", "x", 2),
            ],
        )

    def test_edge_fields_and_kinds_are_preserved(self) -> None:
        graph = load_graph(self.graph_path)
        self.assertEqual(
            graph.edges,
            [
                GraphEdge(self.source_id, self.sink_id, "AST"),
                GraphEdge(self.source_id, self.sink_id, "CFG"),
            ],
        )

    def test_metadata_exposes_only_source_and_sink(self) -> None:
        metadata = load_test_metadata(self.metadata_path)
        self.assertEqual(metadata, TestMetadata(self.source_id, self.sink_id))
        self.assertEqual(
            asdict(metadata),
            {"source_node": self.source_id, "sink_node": self.sink_id},
        )
        self.assertFalse(hasattr(metadata, "reaches"))

    def test_reaches_is_ignored_regardless_of_its_value(self) -> None:
        for expected_result in (True, False, None, "unused", {"unexpected": [1]}):
            with self.subTest(reaches=expected_result):
                self.metadata_data["reaches"] = expected_result
                self.write_json(self.metadata_path, self.metadata_data)
                self.assertEqual(
                    load_test_metadata(self.metadata_path),
                    TestMetadata(self.source_id, self.sink_id),
                )

    def test_reaches_is_not_required(self) -> None:
        del self.metadata_data["reaches"]
        self.write_json(self.metadata_path, self.metadata_data)
        self.assertEqual(
            load_test_metadata(self.metadata_path),
            TestMetadata(self.source_id, self.sink_id),
        )

    def test_directory_loader_needs_only_json_files(self) -> None:
        graph, metadata = load_test_case(str(self.directory))
        self.assertEqual(graph, load_graph(self.graph_path))
        self.assertEqual(metadata, load_test_metadata(self.metadata_path))

    def test_empty_graph_is_preserved(self) -> None:
        self.write_json(self.graph_path, {"nodes": [], "edges": []})
        self.assertEqual(load_graph(self.graph_path), ProgramGraph([], []))

    def test_missing_graph_collections_fail_clearly(self) -> None:
        for field in ("nodes", "edges"):
            with self.subTest(field=field):
                data = dict(self.graph_data)
                del data[field]
                self.write_json(self.graph_path, data)
                with self.assertRaisesRegex(ValueError, f"missing required field '{field}'"):
                    load_graph(self.graph_path)

    def test_missing_node_and_edge_fields_fail_clearly(self) -> None:
        for collection in ("nodes", "edges"):
            for field in self.graph_data[collection][0]:
                with self.subTest(collection=collection, field=field):
                    data = json.loads(json.dumps(self.graph_data))
                    del data[collection][0][field]
                    self.write_json(self.graph_path, data)
                    with self.assertRaisesRegex(ValueError, f"missing required field '{field}'"):
                        load_graph(self.graph_path)

    def test_invalid_graph_shapes_fail_clearly(self) -> None:
        for data in (
            [],
            {"nodes": {}, "edges": []},
            {"nodes": [], "edges": None},
            {"nodes": [None], "edges": []},
            {"nodes": [], "edges": [1]},
        ):
            with self.subTest(data=data):
                self.write_json(self.graph_path, data)
                with self.assertRaisesRegex(ValueError, "expected"):
                    load_graph(self.graph_path)

    def test_wrong_node_and_edge_field_types_are_rejected(self) -> None:
        invalid_fields = (
            ("nodes", "id", True),
            ("nodes", "id", str(self.source_id)),
            ("nodes", "kind", 1),
            ("nodes", "value", None),
            ("nodes", "astOrder", 1.5),
            ("nodes", "astOrder", False),
            ("edges", "source", 1.0),
            ("edges", "destination", "2"),
            ("edges", "kind", None),
        )
        for collection, field, value in invalid_fields:
            with self.subTest(collection=collection, field=field, value=value):
                data = json.loads(json.dumps(self.graph_data))
                data[collection][0][field] = value
                self.write_json(self.graph_path, data)
                with self.assertRaisesRegex(ValueError, f"{field}: expected"):
                    load_graph(self.graph_path)

    def test_missing_metadata_fields_fail_clearly(self) -> None:
        for field in ("sourceNode", "sinkNode"):
            with self.subTest(field=field):
                data = dict(self.metadata_data)
                del data[field]
                self.write_json(self.metadata_path, data)
                with self.assertRaisesRegex(ValueError, f"missing required field '{field}'"):
                    load_test_metadata(self.metadata_path)

    def test_wrong_metadata_types_are_rejected(self) -> None:
        for field in ("sourceNode", "sinkNode"):
            for value in (True, "123", 1.5, None):
                with self.subTest(field=field, value=value):
                    data = dict(self.metadata_data)
                    data[field] = value
                    self.write_json(self.metadata_path, data)
                    with self.assertRaisesRegex(ValueError, f"{field}: expected int"):
                        load_test_metadata(self.metadata_path)
        self.write_json(self.metadata_path, [])
        with self.assertRaisesRegex(ValueError, "expected a JSON object"):
            load_test_metadata(self.metadata_path)

    def test_invalid_json_is_rejected(self) -> None:
        for loader, path in (
            (load_graph, self.graph_path),
            (load_test_metadata, self.metadata_path),
        ):
            with self.subTest(path=path.name):
                path.write_text("{broken JSON", encoding="utf-8")
                with self.assertRaises(json.JSONDecodeError):
                    loader(path)

    def test_missing_input_files_are_rejected(self) -> None:
        for filename in ("graph.json", "test_case.json"):
            with self.subTest(filename=filename):
                path = self.directory / filename
                original = path.read_text(encoding="utf-8")
                path.unlink()
                with self.assertRaises(FileNotFoundError) as raised:
                    load_test_case(self.directory)
                self.assertEqual(raised.exception.filename, str(path))
                path.write_text(original, encoding="utf-8")

    def test_all_supplied_cases_can_be_loaded(self) -> None:
        cases_directory = Path(__file__).resolve().parents[1] / "testcases"
        cases = sorted(cases_directory.iterdir())
        self.assertTrue(cases)
        for case in cases:
            with self.subTest(case=case.name):
                graph, metadata = load_test_case(case)
                raw_graph = json.loads((case / "graph.json").read_text(encoding="utf-8"))
                self.assertEqual(len(graph.nodes), len(raw_graph["nodes"]))
                self.assertEqual(len(graph.edges), len(raw_graph["edges"]))
                for node, record in zip(graph.nodes, raw_graph["nodes"]):
                    self.assertEqual(
                        node,
                        GraphNode(record["id"], record["kind"], record["value"], record["astOrder"]),
                    )
                for edge, record in zip(graph.edges, raw_graph["edges"]):
                    self.assertEqual(
                        edge, GraphEdge(record["source"], record["destination"], record["kind"])
                    )
                raw_metadata = json.loads(
                    (case / "test_case.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    metadata,
                    TestMetadata(raw_metadata["sourceNode"], raw_metadata["sinkNode"]),
                )


if __name__ == "__main__":
    unittest.main()
