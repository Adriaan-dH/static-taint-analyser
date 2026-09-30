"""Read and index program graphs and permitted test metadata."""

import json
from dataclasses import dataclass, field
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
    """Original records with structural indexes built once at construction.

    Treat nodes and edges as read-only after construction. Relationship queries
    return tuples so callers cannot change the indexes.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    _nodes_by_id: dict[int, GraphNode] = field(init=False, repr=False, compare=False)
    _ast_children: dict[int, tuple[GraphNode, ...]] = field(init=False, repr=False, compare=False)
    _ast_parents: dict[int, int] = field(init=False, repr=False, compare=False)
    _cfg_successors: dict[int, tuple[GraphNode, ...]] = field(init=False, repr=False, compare=False)
    _cfg_predecessors: dict[int, tuple[GraphNode, ...]] = field(init=False, repr=False, compare=False)
    _methods: tuple[GraphNode, ...] = field(init=False, repr=False, compare=False)
    _method_parameters: dict[int, tuple[GraphNode, ...]] = field(init=False, repr=False, compare=False)
    _method_exits: dict[int, tuple[GraphNode, ...]] = field(init=False, repr=False, compare=False)
    _containing_methods: dict[int, GraphNode | None] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._nodes_by_id = {}
        for node in self.nodes:
            if node.id in self._nodes_by_id:
                raise ValueError(f"Duplicate node ID {node.id}")
            self._nodes_by_id[node.id] = node

        ast_children: dict[int, list[GraphNode]] = {}
        ast_parents: dict[int, int] = {}
        cfg_successors: dict[int, list[GraphNode]] = {}
        cfg_predecessors: dict[int, list[GraphNode]] = {}
        for edge in self.edges:
            for node_id in (edge.source, edge.destination):
                if node_id not in self._nodes_by_id:
                    raise ValueError(f"{edge.kind} edge references unknown node ID {node_id}")
            source = self._nodes_by_id[edge.source]
            destination = self._nodes_by_id[edge.destination]
            if edge.kind == "AST":
                if destination.id in ast_parents and ast_parents[destination.id] != source.id:
                    raise ValueError(f"Node {destination.id} has multiple AST parents")
                ast_parents[destination.id] = source.id
                ast_children.setdefault(source.id, []).append(destination)
            elif edge.kind == "CFG":
                cfg_successors.setdefault(source.id, []).append(destination)
                cfg_predecessors.setdefault(destination.id, []).append(source)

        # Python's stable sort preserves edge order when siblings share astOrder.
        self._ast_children = {
            node_id: tuple(sorted(children, key=lambda child: child.ast_order))
            for node_id, children in ast_children.items()
        }
        self._cfg_successors = {
            node_id: tuple(successors) for node_id, successors in cfg_successors.items()
        }
        self._cfg_predecessors = {
            node_id: tuple(predecessors) for node_id, predecessors in cfg_predecessors.items()
        }
        self._methods = tuple(node for node in self.nodes if node.kind == "METHOD")
        self._method_parameters = {}
        self._method_exits = {}
        for method in self._methods:
            children = self._ast_children.get(method.id, ())
            self._method_parameters[method.id] = tuple(
                child for child in children if child.kind == "PARAMETER"
            )
            self._method_exits[method.id] = tuple(
                child for child in children if child.kind == "EXIT"
            )
        self._ast_parents = ast_parents
        self._index_method_ownership(ast_parents)

    def _index_method_ownership(self, ast_parents: dict[int, int]) -> None:
        self._containing_methods = {}
        pending: list[tuple[GraphNode, GraphNode | None]] = [
            (node, None) for node in self.nodes if node.id not in ast_parents
        ]
        while pending:
            node, method = pending.pop()
            if node.id in self._containing_methods:
                continue
            if node.kind == "METHOD":
                method = node
            self._containing_methods[node.id] = method
            pending.extend(
                (child, method) for child in self._ast_children.get(node.id, ())
            )
        # With one parent per node, anything unreachable from a root is cyclic.
        if len(self._containing_methods) != len(self.nodes):
            raise ValueError("AST contains a cycle")

    def node(self, node_id: int) -> GraphNode:
        """Look up a node, raising KeyError for an unknown ID."""
        try:
            return self._nodes_by_id[node_id]
        except KeyError:
            raise KeyError(f"Unknown node ID {node_id}") from None

    def ast_children(self, node_id: int) -> tuple[GraphNode, ...]:
        """Return direct AST children in stable ast_order order."""
        self.node(node_id)
        return self._ast_children.get(node_id, ())

    def ast_parent(self, node_id: int) -> GraphNode | None:
        """Return the direct AST parent, or None for an AST root."""
        self.node(node_id)
        parent_id = self._ast_parents.get(node_id)
        return None if parent_id is None else self.node(parent_id)

    def cfg_successors(self, node_id: int) -> tuple[GraphNode, ...]:
        """Return immediate CFG successors in supplied edge order."""
        self.node(node_id)
        return self._cfg_successors.get(node_id, ())

    def cfg_predecessors(self, node_id: int) -> tuple[GraphNode, ...]:
        """Return immediate CFG predecessors in supplied edge order."""
        self.node(node_id)
        return self._cfg_predecessors.get(node_id, ())

    @property
    def methods(self) -> tuple[GraphNode, ...]:
        """All METHOD nodes in supplied node order."""
        return self._methods

    def _require_method(self, method_id: int) -> GraphNode:
        method = self.node(method_id)
        if method.kind != "METHOD":
            raise ValueError(f"Node {method_id} is not a METHOD")
        return method

    def method_parameters(self, method_id: int) -> tuple[GraphNode, ...]:
        """Return direct PARAMETER children in AST order."""
        self._require_method(method_id)
        return self._method_parameters[method_id]

    def method_entry(self, method_id: int) -> GraphNode:
        """The supplied CFG starts at the METHOD node itself."""
        return self._require_method(method_id)

    def method_exit(self, method_id: int) -> GraphNode:
        """Return the method's direct EXIT child; require exactly one."""
        self._require_method(method_id)
        exits = self._method_exits[method_id]
        if len(exits) != 1:
            raise ValueError(f"METHOD {method_id}: expected one EXIT child, found {len(exits)}")
        return exits[0]

    def containing_method(self, node_id: int) -> GraphNode | None:
        """Nearest AST METHOD, including self; None outside any method.

        A nested METHOD owns itself and its subtree, rather than belonging to
        the outer method. CFG edges do not determine ownership.
        """
        self.node(node_id)
        return self._containing_methods[node_id]


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
