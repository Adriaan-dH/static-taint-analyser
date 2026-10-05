"""Forward scalar and list taint analysis within one method, without calls."""

from collections import deque
from collections.abc import Mapping
import re

from src.graph import GraphNode, ProgramGraph, TestMetadata
from src.state import AbstractValue, CLEAN_SCALAR, TaintState


SCALAR_BINARY_OPERATORS = {
    "addition", "subtraction", "multiplication", "division", "floorDiv",
    "modulo", "exponentiation", "and", "or", "xor", "shiftLeft", "arithmeticShiftRight",
}
SIGN_OPERATORS = {"unaryPlus", "unaryMinus", "plus", "minus"}
SCALAR_UNARY_OPERATORS = SIGN_OPERATORS | {"not"}
SCALAR_OPERATORS = SCALAR_BINARY_OPERATORS | SCALAR_UNARY_OPERATORS
COMPARISON_OPERATORS = {
    "equal", "notEqual", "lessThan", "lessThanOrEqual",
    "greaterThan", "greaterThanOrEqual",
    "equals", "notEquals", "lessEqualsThan", "greaterEqualsThan",
    "is", "isNot", "in", "notIn",
}
LOGICAL_OPERATORS = {"logicalAnd", "logicalOr", "logicalNot"}
BOOLEAN_OPERATORS = COMPARISON_OPERATORS | LOGICAL_OPERATORS
EXPRESSION_OPERATORS = SCALAR_OPERATORS | BOOLEAN_OPERATORS | {
    "listLiteral", "indexAccess", "conditional",
}
AUGMENTED_ASSIGNMENTS = {
    "assignmentPlus", "assignmentMinus", "assignmentMultiplication", "assignmentDivision",
    "assignmentFloorDiv", "assignmentModulo", "assignmentExponentiation", "assignmentAnd",
    "assignmentOr", "assignmentXor", "assignmentShiftLeft", "assignmentArithmeticShiftRight",
}
ASSIGNMENT_OPERATORS = {"assignment"} | AUGMENTED_ASSIGNMENTS
NO_EFFECT_OPERATORS = {"pass"}


def abstract_index(graph: ProgramGraph, node: GraphNode) -> int | None:
    """Recognise integer syntax only; all other expressions denote any index."""
    if node.kind == "LITERAL" and re.fullmatch(r"[0-9]+", node.value):
        return int(node.value)
    if node.kind == "OPERATOR" and node.value in SIGN_OPERATORS:
        children = graph.ast_children(node.id)
        if len(children) == 1 and children[0].kind == "LITERAL":
            index = abstract_index(graph, children[0])
            if index is not None:
                return -index if node.value in {"minus", "unaryMinus"} else index
    return None


def _index_operands(graph: ProgramGraph, node: GraphNode) -> tuple[GraphNode, GraphNode]:
    children = graph.ast_children(node.id)
    if len(children) != 2:
        raise ValueError(f"Index access {node.id}: expected base and index children")
    return children[0], children[1]


def _list_targets(
    graph: ProgramGraph, node: GraphNode, state: TaintState,
    call_values: Mapping[int, AbstractValue] | None = None,
) -> AbstractValue:
    base, index = _index_operands(graph, node)
    targets = evaluate_expression(graph, base, state, call_values=call_values)
    # Validate the index, but its taint never influences a load or store.
    evaluate_expression(graph, index, state, call_values=call_values)
    if not targets.list_objects:
        raise NotImplementedError("Index access requires a modelled list reference")
    return targets


def evaluate_expression(
    graph: ProgramGraph, node: GraphNode, state: TaintState,
    call_values: Mapping[int, AbstractValue] | None = None,
) -> AbstractValue:
    """Read values and references without executing Python or mutating state."""
    if node.kind == "CALL" and call_values is not None and node.id in call_values:
        return call_values[node.id]
    if node.kind in ("IDENTIFIER", "PARAMETER"):
        return state.value(node.value)
    if node.kind == "LITERAL":
        return CLEAN_SCALAR
    if node.kind == "BLOCK":
        children = graph.ast_children(node.id)
        if not children:
            raise NotImplementedError("Empty expression BLOCK is not supported")
        result = AbstractValue()
        for child in children:
            # Chained comparisons are lowered to temporary assignments followed
            # by a result expression. The CFG already performed those writes.
            if child.kind == "OPERATOR" and child.value == "assignment":
                operands = graph.ast_children(child.id)
                if len(operands) != 2:
                    raise ValueError(f"Assignment {child.id}: expected target and RHS children")
                result = evaluate_expression(graph, operands[1], state, call_values=call_values)
            else:
                result = evaluate_expression(graph, child, state, call_values=call_values)
        return result
    if node.kind == "OPERATOR" and node.value == "listLiteral":
        return AbstractValue(list_objects=frozenset({node.id}))
    if node.kind == "OPERATOR" and node.value == "indexAccess":
        targets = _list_targets(graph, node, state, call_values=call_values)
        _, index_node = _index_operands(graph, node)
        index = abstract_index(graph, index_node)
        result = AbstractValue()
        for object_id in targets.list_objects:
            result = result.join(state.list_memory(object_id).read(index))
        return result
    if node.kind == "OPERATOR" and node.value == "conditional":
        children = graph.ast_children(node.id)
        if len(children) != 3:
            raise ValueError(f"Conditional {node.id}: expected condition and two alternatives")
        evaluate_expression(graph, children[0], state, call_values=call_values)
        # Like an if/else assignment, both value alternatives are reachable;
        # the predicate does not introduce implicit control-flow taint.
        return evaluate_expression(graph, children[1], state, call_values=call_values).join(
            evaluate_expression(graph, children[2], state, call_values=call_values)
        )
    if node.kind == "OPERATOR" and node.value in SCALAR_OPERATORS | BOOLEAN_OPERATORS:
        operands = graph.ast_children(node.id)
        if node.value in {"logicalAnd", "logicalOr"}:
            if len(operands) < 2:
                raise ValueError(f"Operator {node.id} ({node.value}): expected at least 2 operands")
            expected_count = len(operands)
        else:
            expected_count = 1 if node.value in SCALAR_UNARY_OPERATORS | {"logicalNot"} else 2
        if len(operands) != expected_count:
            raise ValueError(
                f"Operator {node.id} ({node.value}): expected {expected_count} operands"
            )
        # Visit every operand so an unsupported child cannot be hidden by taint.
        values = [
            evaluate_expression(graph, operand, state, call_values=call_values)
            for operand in operands
        ]
        if node.value in BOOLEAN_OPERATORS:
            # Follow the assignment's Boolean abstraction, also for and/or.
            return CLEAN_SCALAR
        return _scalar_operator_value(values)
    raise NotImplementedError(f"Unsupported expression: {node.kind} {node.value!r}")


def _scalar_operator_value(values: list[AbstractValue]) -> AbstractValue:
    if any(value.list_objects for value in values):
        raise NotImplementedError("List arithmetic is not supported")
    return AbstractValue(
        scalar_tainted=any(value.scalar_tainted for value in values), may_be_scalar=True
    )


def _prepare_lists(
    graph: ProgramGraph, node: GraphNode, state: TaintState, refresh_unsequenced: bool = False,
    call_values: Mapping[int, AbstractValue] | None = None,
) -> TaintState:
    """Initialise literals omitted from a synthetic CFG, or used as a sink.

    Real builder graphs sequence each literal before its enclosing expression.
    Reading that expression must reuse the object, not allocate it a second time.
    """
    for child in graph.ast_children(node.id):
        state = _prepare_lists(graph, child, state, refresh_unsequenced, call_values)
    if node.kind == "OPERATOR" and node.value == "listLiteral":
        sequenced = graph.cfg_predecessors(node.id) or graph.cfg_successors(node.id)
        if node.id not in dict(state.lists) or (refresh_unsequenced and not sequenced):
            state = _allocate_list(graph, node, state, call_values=call_values)
    return state


def _allocate_list(
    graph: ProgramGraph, node: GraphNode, state: TaintState,
    call_values: Mapping[int, AbstractValue] | None = None,
) -> TaintState:
    elements = tuple(
        evaluate_expression(graph, child, state, call_values=call_values)
        for child in graph.ast_children(node.id)
    )
    return state.allocate(node.id, elements)


def evaluate_scalar_expression(
    graph: ProgramGraph, node: GraphNode, state: TaintState,
    call_values: Mapping[int, AbstractValue] | None = None,
) -> bool:
    """Compatibility taint query, also observing the contents of list values."""
    state = _prepare_lists(graph, node, state, call_values=call_values)
    return state.value_is_tainted(evaluate_expression(graph, node, state, call_values=call_values))


def _write_list(
    graph: ProgramGraph, target: GraphNode, value: AbstractValue, state: TaintState,
    call_values: Mapping[int, AbstractValue] | None = None,
) -> TaintState:
    targets = _list_targets(graph, target, state, call_values=call_values)
    _, index_node = _index_operands(graph, target)
    index = abstract_index(graph, index_node)
    definite_object = len(targets.list_objects) == 1 and not targets.may_be_scalar
    for object_id in targets.list_objects:
        memory = state.list_memory(object_id)
        strong = definite_object and index is not None and not memory.multiple_allocations
        state = state.with_list(object_id, memory.write(index, value, strong))
    return state


def transfer(
    graph: ProgramGraph, node: GraphNode, state: TaintState,
    call_values: Mapping[int, AbstractValue] | None = None,
) -> TaintState:
    """Allocate lists, rebind variables, or mutate shared list memory."""
    if node.kind == "OPERATOR" and node.value == "listLiteral":
        for child in graph.ast_children(node.id):
            state = _prepare_lists(graph, child, state, call_values=call_values)
        return _allocate_list(graph, node, state, call_values=call_values)
    if node.kind != "OPERATOR":
        return state
    if node.value not in ASSIGNMENT_OPERATORS:
        if node.value not in EXPRESSION_OPERATORS | NO_EFFECT_OPERATORS:
            raise NotImplementedError(f"Unsupported expression: {node.kind} {node.value!r}")
        return state
    children = graph.ast_children(node.id)
    if len(children) != 2:
        raise ValueError(f"Assignment {node.id}: expected target and RHS children")
    target, expression = children
    if expression.kind == "METHOD_REF" and node.value == "assignment":
        # A nested definition is a local CFG event. Callable bindings and the
        # nested body belong to Part 2 and are deliberately not modelled here.
        return state
    state = _prepare_lists(graph, expression, state, refresh_unsequenced=True, call_values=call_values)
    value = evaluate_expression(graph, expression, state, call_values=call_values)
    if node.value in AUGMENTED_ASSIGNMENTS:
        value = _scalar_operator_value([
            evaluate_expression(graph, target, state, call_values=call_values), value
        ])
    if target.kind == "IDENTIFIER":
        return state.bind(target.value, value)
    if target.kind == "OPERATOR" and target.value == "indexAccess":
        state = _prepare_lists(graph, target, state, refresh_unsequenced=True, call_values=call_values)
        return _write_list(graph, target, value, state, call_values=call_values)
    raise NotImplementedError("Only IDENTIFIER and indexAccess assignment targets are supported")


def _sink_program_point(graph: ProgramGraph, sink: GraphNode, entry: GraphNode) -> GraphNode:
    """Locate the CFG expression that evaluates the specific sink AST node."""
    if sink.kind == "PARAMETER":
        return entry
    parent = graph.ast_parent(sink.id)
    if parent is not None and parent.kind == "OPERATOR" and parent.value in ASSIGNMENT_OPERATORS:
        if graph.ast_children(parent.id)[0] == sink:
            return parent
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
                current.value in ASSIGNMENT_OPERATORS
                or (parent is not None and parent.kind in ("BLOCK", "CONTROL_STRUCTURE"))
            )
        ):
            return current
        current = parent
    raise ValueError(f"Sink {sink.id} is not contained in a CFG expression")


def analyse_intraprocedural(graph: ProgramGraph, metadata: TestMetadata) -> bool:
    """Check a scalar or list sink in the source parameter's method.

    Only parameter sources are supported. Calls and control
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
    if sink.kind not in ("IDENTIFIER", "LITERAL", "PARAMETER", "OPERATOR", "BLOCK") or (
        sink.kind == "OPERATOR"
        and sink.value not in EXPRESSION_OPERATORS
    ):
        raise NotImplementedError(f"Unsupported sink: {sink.kind} {sink.value!r}")

    entry = graph.method_entry(method.id)
    entry_state = TaintState()
    for parameter in graph.method_parameters(method.id):
        entry_state = entry_state.bind(parameter.value, CLEAN_SCALAR)
    entry_state = entry_state.taint(source.value)
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
    sink_state = sink_state_at(
        graph, sink, sink_point, in_states[sink_point.id], out_states[sink_point.id]
    )
    return evaluate_scalar_expression(graph, sink, sink_state)


def sink_state_at(
    graph: ProgramGraph, sink: GraphNode, sink_point: GraphNode,
    incoming: TaintState, outgoing: TaintState,
) -> TaintState:
    """Share sink timing between the intra- and interprocedural solvers."""
    if sink_point == sink and sink.kind == "OPERATOR" and sink.value == "listLiteral":
        return outgoing
    if sink_point.kind == "OPERATOR" and sink_point.value in ASSIGNMENT_OPERATORS:
        # Assignment targets observe the write; other expressions observe IN.
        if graph.ast_children(sink_point.id)[0] == sink:
            return outgoing
    return incoming
