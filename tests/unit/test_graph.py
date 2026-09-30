"""Focused tests for graph parsing and structural indexes. No analysis."""

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


class GraphIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = ProgramGraph(
            nodes=[
                GraphNode(100, "METHOD", "outer", 1),
                GraphNode(102, "PARAMETER", "second", 2),
                GraphNode(190, "EXIT", "EXIT", 2),
                GraphNode(110, "BLOCK", "[ ... ]", 1),
                GraphNode(101, "PARAMETER", "first", 1),
                GraphNode(120, "CALL", "sink", 2),
                GraphNode(121, "IDENTIFIER", "first", 1),
                GraphNode(200, "METHOD", "other", 2),
                GraphNode(201, "PARAMETER", "x", 1),
                GraphNode(290, "EXIT", "EXIT", 2),
                GraphNode(300, "METHOD", "other", 1),
                GraphNode(310, "BLOCK", "[ ... ]", 1),
                GraphNode(301, "PARAMETER", "nested", 1),
                GraphNode(320, "RETURN", "", 1),
                GraphNode(390, "EXIT", "EXIT", 2),
                GraphNode(400, "MODULE", "", 0),
                GraphNode(401, "IDENTIFIER", "outside", 3),
            ],
            edges=[
                GraphEdge(100, 102, "AST"),
                GraphEdge(100, 190, "AST"),
                GraphEdge(100, 110, "AST"),
                GraphEdge(100, 101, "AST"),
                GraphEdge(110, 120, "AST"),
                GraphEdge(110, 300, "AST"),
                GraphEdge(120, 121, "AST"),
                GraphEdge(200, 201, "AST"),
                GraphEdge(200, 290, "AST"),
                GraphEdge(300, 310, "AST"),
                GraphEdge(300, 301, "AST"),
                GraphEdge(300, 390, "AST"),
                GraphEdge(310, 320, "AST"),
                GraphEdge(400, 100, "AST"),
                GraphEdge(400, 200, "AST"),
                GraphEdge(400, 401, "AST"),
                GraphEdge(100, 120, "CFG"),
                GraphEdge(120, 190, "CFG"),
                GraphEdge(100, 190, "CFG"),
                GraphEdge(200, 290, "CFG"),
                GraphEdge(300, 320, "CFG"),
                GraphEdge(320, 390, "CFG"),
            ],
        )

    def test_node_lookup_returns_original_node(self) -> None:
        self.assertIs(self.graph.node(102), self.graph.nodes[1])

    def test_unknown_ids_fail_for_all_node_queries(self) -> None:
        for query in (
            self.graph.node, self.graph.ast_children, self.graph.cfg_successors,
            self.graph.cfg_predecessors, self.graph.containing_method,
            self.graph.method_parameters, self.graph.method_entry, self.graph.method_exit,
        ):
            with self.subTest(query=query.__name__):
                with self.assertRaisesRegex(KeyError, "Unknown node ID 999"):
                    query(999)

    def test_ast_children_use_ast_order_and_stable_ties(self) -> None:
        self.assertEqual(
            self.graph.ast_children(100),
            tuple(self.graph.node(i) for i in (110, 101, 102, 190)),
        )
        self.assertEqual(
            self.graph.ast_children(110),
            (self.graph.node(300), self.graph.node(120)),
        )
        self.assertEqual(self.graph.ast_children(190), ())

    def test_cfg_successors_exclude_ast_edges(self) -> None:
        self.assertEqual(
            self.graph.cfg_successors(100),
            (self.graph.node(120), self.graph.node(190)),
        )
        self.assertEqual(self.graph.cfg_successors(120), (self.graph.node(190),))
        self.assertEqual(self.graph.cfg_successors(101), ())
        self.assertEqual(self.graph.cfg_successors(190), ())

    def test_cfg_predecessors_exclude_ast_edges(self) -> None:
        self.assertEqual(
            self.graph.cfg_predecessors(190),
            (self.graph.node(120), self.graph.node(100)),
        )
        self.assertEqual(self.graph.cfg_predecessors(120), (self.graph.node(100),))
        self.assertEqual(self.graph.cfg_predecessors(121), ())
        self.assertEqual(self.graph.cfg_predecessors(100), ())

    def test_methods_and_parameters_are_separate_and_ordered(self) -> None:
        self.assertEqual(self.graph.methods, tuple(self.graph.node(i) for i in (100, 200, 300)))
        self.assertEqual(
            self.graph.method_parameters(100),
            (self.graph.node(101), self.graph.node(102)),
        )
        self.assertEqual(self.graph.method_parameters(200), (self.graph.node(201),))
        self.assertEqual(self.graph.method_parameters(300), (self.graph.node(301),))

    def test_method_entry_and_exit(self) -> None:
        for method_id, exit_id in ((100, 190), (200, 290), (300, 390)):
            with self.subTest(method=method_id):
                self.assertIs(self.graph.method_entry(method_id), self.graph.node(method_id))
                self.assertIs(self.graph.method_exit(method_id), self.graph.node(exit_id))

    def test_method_queries_reject_other_node_kinds(self) -> None:
        for query in (self.graph.method_parameters, self.graph.method_entry, self.graph.method_exit):
            with self.subTest(query=query.__name__):
                with self.assertRaisesRegex(ValueError, "Node 110 is not a METHOD"):
                    query(110)

    def test_ownership_follows_ast_and_nested_methods_own_themselves(self) -> None:
        for method_id, owned_ids in (
            (100, (100, 101, 102, 110, 120, 121, 190)),
            (200, (200, 201, 290)),
            (300, (300, 301, 310, 320, 390)),
        ):
            for node_id in owned_ids:
                with self.subTest(node=node_id):
                    self.assertIs(self.graph.containing_method(node_id), self.graph.node(method_id))
        self.assertIsNone(self.graph.containing_method(400))
        self.assertIsNone(self.graph.containing_method(401))

    def test_partial_method_has_no_parameters_and_exit_query_fails(self) -> None:
        graph = ProgramGraph([GraphNode(1, "METHOD", "empty", 1)], [])
        self.assertEqual(graph.method_parameters(1), ())
        with self.assertRaisesRegex(ValueError, "expected one EXIT child, found 0"):
            graph.method_exit(1)

    def test_multiple_exit_children_are_not_chosen_silently(self) -> None:
        graph = ProgramGraph(
            [GraphNode(1, "METHOD", "f", 1), GraphNode(2, "EXIT", "", 2), GraphNode(3, "EXIT", "", 2)],
            [GraphEdge(1, 2, "AST"), GraphEdge(1, 3, "AST")],
        )
        with self.assertRaisesRegex(ValueError, "expected one EXIT child, found 2"):
            graph.method_exit(1)

    def test_invalid_graph_structure_fails_clearly(self) -> None:
        nodes = [GraphNode(i, "BLOCK", "", i) for i in (1, 2, 3)]
        invalid_graphs = (
            (nodes + [nodes[0]], [], "Duplicate node ID 1"),
            (nodes, [GraphEdge(1, 9, "AST")], "unknown node ID 9"),
            (nodes, [GraphEdge(9, 1, "CFG")], "unknown node ID 9"),
            (nodes, [GraphEdge(1, 3, "AST"), GraphEdge(2, 3, "AST")], "multiple AST parents"),
            (nodes, [GraphEdge(1, 2, "AST"), GraphEdge(2, 1, "AST")], "AST contains a cycle"),
        )
        for graph_nodes, edges, message in invalid_graphs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    ProgramGraph(graph_nodes, edges)

    def test_all_supplied_cases_have_consistent_indexes(self) -> None:
        cases_directory = Path(__file__).resolve().parents[1] / "testcases"
        for case in sorted(cases_directory.iterdir()):
            with self.subTest(case=case.name):
                graph, metadata = load_test_case(case)
                self.assertTrue(graph.methods)
                for node in graph.nodes:
                    self.assertIs(graph.node(node.id), node)
                    self.assertIn(graph.containing_method(node.id), graph.methods)
                    children = graph.ast_children(node.id)
                    self.assertEqual(list(children), sorted(children, key=lambda child: child.ast_order))
                    for successor in graph.cfg_successors(node.id):
                        self.assertIn(node, graph.cfg_predecessors(successor.id))
                for method in graph.methods:
                    self.assertIs(graph.method_entry(method.id), method)
                    self.assertTrue(graph.cfg_successors(method.id))
                    self.assertIn(graph.method_exit(method.id), graph.ast_children(method.id))
                    self.assertEqual(
                        graph.method_parameters(method.id),
                        tuple(child for child in graph.ast_children(method.id) if child.kind == "PARAMETER"),
                    )
                self.assertIsNotNone(graph.containing_method(metadata.source_node))
                self.assertIsNotNone(graph.containing_method(metadata.sink_node))


if __name__ == "__main__":
    unittest.main()
