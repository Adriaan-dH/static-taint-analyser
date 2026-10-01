"""Forward scalar taint analysis within one method, without call propagation."""

from collections import deque

from src.graph import GraphNode, ProgramGraph, TestMetadata
from src.state import TaintState


SCALAR_BINARY_OPERATORS = {"addition", "subtraction", "multiplication", "division"}
SCALAR_UNARY_OPERATORS = {"unaryPlus", "unaryMinus"}
SCALAR_OPERATORS = SCALAR_BINARY_OPERATORS | SCALAR_UNARY_OPERATORS
COMPARISON_OPERATORS = {
    "equal", "notEqual", "lessThan", "lessThanOrEqual",
    "greaterThan", "greaterThanOrEqual",
}


def evaluate_scalar_expression(graph: ProgramGraph, node: GraphNode, state: TaintState) -> bool:
    """Evaluate scalar taint only; never execute or evaluate Python values."""
    if node.kind in ("IDENTIFIER", "PARAMETER"):
        return state.is_tainted(node.value)
    if node.kind == "LITERAL":
        return False
    if node.kind == "OPERATOR" and node.value in SCALAR_OPERATORS | COMPARISON_OPERATORS:
        operands = graph.ast_children(node.id)
        expected_count = 1 if node.value in SCALAR_UNARY_OPERATORS else 2
        if len(operands) != expected_count:
            raise ValueError(
                f"Operator {node.id} ({node.value}): expected {expected_count} operands"
            )
        # Visit every operand so an unsupported child cannot be hidden by taint.
        operand_taint = [evaluate_scalar_expression(graph, operand, state) for operand in operands]
        if node.value in COMPARISON_OPERATORS:
            return False
        return any(operand_taint)
    raise NotImplementedError(f"Unsupported scalar expression: {node.kind} {node.value!r}")


def transfer(graph: ProgramGraph, node: GraphNode, state: TaintState) -> TaintState:
    """Overwrite a scalar assignment target; other CFG nodes preserve facts."""
    if node.kind != "OPERATOR" or node.value != "assignment":
        return state
    children = graph.ast_children(node.id)
    if len(children) != 2:
        raise ValueError(f"Assignment {node.id}: expected target and RHS children")
    target, expression = children
    if target.kind != "IDENTIFIER":
        raise NotImplementedError("Only scalar IDENTIFIER assignment targets are supported")
    if evaluate_scalar_expression(graph, expression, state):
        return state.taint(target.value)
    return state.clean(target.value)


def _sink_program_point(graph: ProgramGraph, sink: GraphNode, entry: GraphNode) -> GraphNode:
    """Locate the CFG expression that evaluates the specific sink AST node."""
    if sink.kind == "PARAMETER":
        return entry
    current: GraphNode | None = sink
    while current is not None and current != entry:
        if graph.cfg_predecessors(current.id) or graph.cfg_successors(current.id):
            return current
        # An unreachable expression may have no CFG edges. Do not fall back to
        # the method entry and accidentally regard its children as executed.
        parent = graph.ast_parent(current.id)
        if current.kind in ("CALL", "RETURN") or (
            current.kind == "OPERATOR"
            and (
                current.value == "assignment"
                or (parent is not None and parent.kind in ("BLOCK", "CONTROL_STRUCTURE"))
            )
        ):
            return current
        current = parent
    raise ValueError(f"Sink {sink.id} is not contained in a CFG expression")


def analyse_intraprocedural(graph: ProgramGraph, metadata: TestMetadata) -> bool:
    """Check a scalar sink in the source parameter's method.

    Only parameter sources and scalar sinks are supported. Calls and control
    predicates preserve state; their results do not change CFG reachability.
    """
    source = graph.node(metadata.source_node)
    sink = graph.node(metadata.sink_node)
    method = graph.containing_method(source.id)
    sink_method = graph.containing_method(sink.id)
    if method is None or sink_method is None:
        raise ValueError("Source and sink must belong to a METHOD")
    if method != sink_method:
        raise ValueError("Intraprocedural analysis requires source and sink in the same METHOD")
    if source not in graph.method_parameters(method.id):
        raise NotImplementedError("Only direct method PARAMETER sources are supported")
    if sink.kind not in ("IDENTIFIER", "LITERAL", "PARAMETER", "OPERATOR") or (
        sink.kind == "OPERATOR" and sink.value not in SCALAR_OPERATORS | COMPARISON_OPERATORS
    ):
        raise NotImplementedError(f"Unsupported scalar sink: {sink.kind} {sink.value!r}")

    entry = graph.method_entry(method.id)
    entry_state = TaintState().taint(source.value)
    sink_point = _sink_program_point(graph, sink, entry)
    in_states: dict[int, TaintState] = {}
    out_states: dict[int, TaintState] = {}
    worklist = deque([entry])
    queued = {entry.id}

    while worklist:
        node = worklist.popleft()
        queued.remove(node.id)
        incoming = entry_state if node == entry else TaintState()
        for predecessor in graph.cfg_predecessors(node.id):
            incoming = incoming.join(out_states.get(predecessor.id, TaintState()))
        in_states[node.id] = incoming
        outgoing = transfer(graph, node, incoming)
        # Absence differs from a clean state: even clean paths must be visited.
        if node.id in out_states and out_states[node.id] == outgoing:
            continue
        out_states[node.id] = outgoing
        # Changes to OUT re-schedule successors until loops reach a fixed point.
        for successor in graph.cfg_successors(node.id):
            if graph.containing_method(successor.id) != method:
                raise ValueError("CFG edge leaves the selected METHOD")
            if successor.id not in queued:
                worklist.append(successor)
                queued.add(successor.id)

    if sink_point.id not in in_states:
        return False
    sink_state = in_states[sink_point.id]
    if sink_point.kind == "OPERATOR" and sink_point.value == "assignment":
        # A target denotes the assigned value, whereas the RHS uses IN facts.
        if graph.ast_children(sink_point.id)[0] == sink:
            sink_state = out_states[sink_point.id]
    return evaluate_scalar_expression(graph, sink, sink_state)
