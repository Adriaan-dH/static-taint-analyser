"""Read program graphs and the metadata permitted as analyser input."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar


@dataclass(frozen=True)
class GraphNode:
    id: int
    kind: str
    value: str
    ast_order: int


@dataclass(frozen=True)
class GraphEdge:
    source: int
    destination: int
    kind: str


@dataclass
class ProgramGraph:
    """Nodes and edges in the order supplied by graph.json."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]


@dataclass(frozen=True)
class TestMetadata:
    """Source and sink IDs only. Expected analysis results are excluded."""

    source_node: int
    sink_node: int


FieldType = TypeVar("FieldType")


def _required_field(
    data: object, name: str, expected_type: type[FieldType], context: str
) -> FieldType:
    if not isinstance(data, dict):
        raise ValueError(f"{context}: expected a JSON object")
    if name not in data:
        raise ValueError(f"{context}: missing required field {name!r}")
    value = data[name]
    # Exact types reject booleans as IDs and avoid coercing strings or floats.
    if type(value) is not expected_type:
        raise ValueError(
            f"{context}.{name}: expected {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
    return value


def _load_json_object(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def load_graph(path: str | Path) -> ProgramGraph:
    """Load a graph JSON file, requiring every node and edge field."""
    path = Path(path)
    data = _load_json_object(path)
    node_records = _required_field(data, "nodes", list, str(path))
    edge_records = _required_field(data, "edges", list, str(path))

    nodes = []
    for index, record in enumerate(node_records):
        context = f"{path}: nodes[{index}]"
        nodes.append(
            GraphNode(
                id=_required_field(record, "id", int, context),
                kind=_required_field(record, "kind", str, context),
                value=_required_field(record, "value", str, context),
                ast_order=_required_field(record, "astOrder", int, context),
            )
        )

    edges = []
    for index, record in enumerate(edge_records):
        context = f"{path}: edges[{index}]"
        edges.append(
            GraphEdge(
                source=_required_field(record, "source", int, context),
                destination=_required_field(record, "destination", int, context),
                kind=_required_field(record, "kind", str, context),
            )
        )
    return ProgramGraph(nodes=nodes, edges=edges)


def load_test_metadata(path: str | Path) -> TestMetadata:
    """Load sourceNode and sinkNode. Ignore all other fields, including reaches."""
    path = Path(path)
    data = _load_json_object(path)
    return TestMetadata(
        source_node=_required_field(data, "sourceNode", int, str(path)),
        sink_node=_required_field(data, "sinkNode", int, str(path)),
    )


def load_test_case(directory: str | Path) -> tuple[ProgramGraph, TestMetadata]:
    """Load graph.json and test_case.json from a test-case directory."""
    directory = Path(directory)
    return (
        load_graph(directory / "graph.json"),
        load_test_metadata(directory / "test_case.json"),
    )
