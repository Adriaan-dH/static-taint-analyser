"""Flow-sensitive direct, indirect, and nested calls with shared list memory."""

from collections import deque
from dataclasses import dataclass, replace

from src.graph import GraphNode, ProgramGraph, TestMetadata
from src.intraprocedural import (
    ASSIGNMENT_OPERATORS, EXPRESSION_OPERATORS, _prepare_lists,
    _sink_program_point, evaluate_expression, evaluate_scalar_expression,
    sink_state_at, transfer,
)
from src.state import AbstractValue, CLEAN_SCALAR, TAINTED_SCALAR, TaintState


@dataclass(frozen=True)
class FlowFacts:
    """Call results travel along CFG paths, including loop back-edges."""

    state: TaintState
    call_values: tuple[tuple[int, AbstractValue], ...] = ()

    def join(self, other: "FlowFacts") -> "FlowFacts":
        values = dict(self.call_values)
        for node_id, value in other.call_values:
            values[node_id] = values.get(node_id, AbstractValue()).join(value)
        return FlowFacts(self.state.join(other.state), tuple(sorted(values.items())))


@dataclass(frozen=True)
class InvocationResult:
    value: AbstractValue
    state: TaintState
    sink_tainted: bool
    returns: bool


class InterproceduralAnalysis:
    def __init__(self, graph: ProgramGraph, metadata: TestMetadata) -> None:
        self.graph = graph
        self.sink = graph.node(metadata.sink_node)
        self.sink_method = graph.containing_method(self.sink.id)
        self.top_level: dict[str, list[GraphNode]] = {}
        self.nested_methods: dict[tuple[int, str], list[GraphNode]] = {}
        self.local_names: dict[int, set[str]] = {}
        self.call_ids: dict[int, list[int]] = {}
        for method in graph.methods:
            parent = graph.ast_parent(method.id)
            owner = graph.containing_method(parent.id) if parent is not None else None
            if owner is None:
                self.top_level.setdefault(method.value, []).append(method)
            else:
                self.nested_methods.setdefault((owner.id, method.value), []).append(method)
            self.local_names[method.id] = {
                parameter.value for parameter in graph.method_parameters(method.id)
            }
        for node in graph.nodes:
            method = graph.containing_method(node.id)
            if method is not None and node.kind == "CALL" and (
                graph.cfg_predecessors(node.id) or graph.cfg_successors(node.id)
            ):
                self.call_ids.setdefault(method.id, []).append(node.id)
            if method is not None and node.kind == "OPERATOR" and node.value in ASSIGNMENT_OPERATORS:
                children = graph.ast_children(node.id)
                if children and children[0].kind == "IDENTIFIER":
                    self.local_names[method.id].add(children[0].value)

    def resolve(
        self, call: GraphNode, method: GraphNode, state: TaintState,
    ) -> tuple[GraphNode, ...]:
        # Local names use their function references at this program point.
        # Other names can resolve to a module-level function below.
        if call.value in self.local_names[method.id]:
            value = state.value(call.value)
            if (
                not value.function_methods or value.may_be_scalar
                or value.scalar_tainted or value.list_objects
            ):
                raise NotImplementedError(
                    f"Local call {call.value!r} requires function targets without scalar/list values"
                )
            targets = tuple(self.graph.node(node_id) for node_id in sorted(value.function_methods))
            if any(target.kind != "METHOD" for target in targets):
                raise ValueError("Function references must identify METHOD nodes")
            return targets
        # A nested self-call is recursive even though its enclosing binding
        # is not captured into this invocation's local state.
        if call.value == method.value and self.graph.ast_parent(method.id) is not None:
            return (method,)
        targets = self.top_level.get(call.value, [])
        if len(targets) == 1:
            return (targets[0],)
        if targets:
            raise NotImplementedError(f"Ambiguous direct call target {call.value!r}")
        parent = self.graph.ast_parent(call.id)
        if call.value in {"sink", "print"} and parent is not None and parent.kind == "BLOCK":
            return ()
        raise NotImplementedError(f"Unresolved direct or indirect call {call.value!r}")

    def nested_reference(self, reference: GraphNode, method: GraphNode) -> AbstractValue:
        targets = self.nested_methods.get((method.id, reference.value), [])
        if len(targets) != 1:
            raise NotImplementedError(
                f"METHOD_REF {reference.value!r} in {method.value!r}: "
                f"expected one lexical nested METHOD, found {len(targets)}"
            )
        return AbstractValue(function_methods=frozenset({targets[0].id}))

    def invoke_targets(
        self, targets: tuple[GraphNode, ...], actuals: tuple[AbstractValue, ...],
        caller: TaintState, stack: tuple[int, ...],
    ) -> InvocationResult:
        # Reject inconsistent arity before analysing any alternative. Runtime
        # argument-error paths are outside the supported abstraction.
        for target in targets:
            count = len(self.graph.method_parameters(target.id))
            if count != len(actuals):
                raise ValueError(
                    f"Call to {target.value!r}: expected {count} arguments, got {len(actuals)}"
                )
        value = AbstractValue()
        heap = TaintState()
        sink_tainted = False
        returns = False
        # Targets are alternatives, so each starts from the same caller state.
        # Merge their return values and list effects after analysing each one.
        for target in targets:
            result = self.invoke(target, actuals, caller, stack)
            sink_tainted |= result.sink_tainted
            if result.returns:
                returns = True
                value = value.join(result.value)
                heap = heap.join(TaintState(lists=result.state.lists))
        state = replace(caller, lists=heap.lists) if returns else caller
        return InvocationResult(value, state, sink_tainted, returns)

    def invoke(
        self, method: GraphNode, actuals: tuple[AbstractValue, ...],
        caller: TaintState, stack: tuple[int, ...],
    ) -> InvocationResult:
        if method.id in stack:
            raise NotImplementedError("Recursive/cyclic direct calls are not supported")
        parameters = self.graph.method_parameters(method.id)
        if len(actuals) != len(parameters):
            raise ValueError(
                f"Call to {method.value!r}: expected {len(parameters)} arguments, got {len(actuals)}"
            )
        # Each call gets fresh local bindings but retains the caller's list memory.
        entry_state = TaintState(lists=caller.lists)
        # The exporter omits module definition events, so seed visible functions.
        for name, targets in self.top_level.items():
            if name not in self.local_names[method.id]:
                entry_state = entry_state.bind(
                    name, AbstractValue(function_methods=frozenset(m.id for m in targets))
                )
        for parameter, value in zip(parameters, actuals):
            entry_state = entry_state.bind(parameter.value, value)
        entry = self.graph.method_entry(method.id)
        exit_node = self.graph.method_exit(method.id)
        sink_point = (
            _sink_program_point(self.graph, self.sink, entry)
            if self.sink_method == method else None
        )
        incoming_facts, outgoing_facts, sink_tainted = self.solve(
            method, entry_state, stack + (method.id,)
        )

        if sink_point is not None and sink_point.id in outgoing_facts:
            incoming = incoming_facts[sink_point.id]
            outgoing = outgoing_facts[sink_point.id]
            state = sink_state_at(
                self.graph, self.sink, sink_point, incoming.state, outgoing.state
            )
            if self.sink.kind == "CALL" and self.sink == sink_point:
                state = outgoing.state
            values = dict(incoming.call_values)
            values.update(outgoing.call_values)
            sink_tainted |= evaluate_scalar_expression(self.graph, self.sink, state, values)

        value = AbstractValue()
        for node_id, facts in outgoing_facts.items():
            node = self.graph.node(node_id)
            if node.kind == "RETURN":
                expressions = self.graph.ast_children(node.id)
                if len(expressions) > 1:
                    raise ValueError(f"RETURN {node.id}: expected at most one expression")
                returned = (
                    evaluate_expression(
                        self.graph, expressions[0], facts.state, dict(facts.call_values)
                    )
                    if expressions else CLEAN_SCALAR
                )
                value = value.join(returned)
        exit_facts = outgoing_facts.get(exit_node.id)
        if exit_facts is not None:
            if any(
                p.id in outgoing_facts and p.kind != "RETURN"
                for p in self.graph.cfg_predecessors(exit_node.id)
            ):
                value = value.join(CLEAN_SCALAR)
            return InvocationResult(value, exit_facts.state, sink_tainted, True)
        return InvocationResult(value, entry_state, sink_tainted, False)

    def solve(
        self, method: GraphNode, entry_state: TaintState, stack: tuple[int, ...],
    ) -> tuple[dict[int, FlowFacts], dict[int, FlowFacts], bool]:
        """Run the method CFG to a fixed point, including call-result changes."""
        entry = self.graph.method_entry(method.id)
        incoming_facts: dict[int, FlowFacts] = {}
        outgoing_facts: dict[int, FlowFacts] = {}
        worklist = deque([entry])
        queued = {entry.id}
        sink_tainted = False
        # Bottom results let a join be visited before a longer call branch.
        # Unsequenced calls remain absent and fail explicitly when evaluated.
        entry_values = tuple(
            (node_id, AbstractValue()) for node_id in sorted(self.call_ids.get(method.id, []))
        )
        while worklist:
            node = worklist.popleft()
            queued.remove(node.id)
            incoming = (
                FlowFacts(entry_state, entry_values) if node == entry else FlowFacts(TaintState())
            )
            # Merge facts from every predecessor already reached by the analysis.
            for predecessor in self.graph.cfg_predecessors(node.id):
                if predecessor.id in outgoing_facts:
                    incoming = incoming.join(outgoing_facts[predecessor.id])
            incoming_facts[node.id] = incoming
            outgoing, observed = self.step(node, method, incoming, stack)
            sink_tainted |= observed
            if outgoing is None:
                continue
            if outgoing_facts.get(node.id) == outgoing:
                continue
            outgoing_facts[node.id] = outgoing
            # Changed facts revisit successors until loops reach a fixed point.
            for successor in self.graph.cfg_successors(node.id):
                if self.graph.containing_method(successor.id) != method:
                    raise ValueError("CFG edge leaves the selected METHOD")
                if successor.id not in queued:
                    worklist.append(successor)
                    queued.add(successor.id)

        return incoming_facts, outgoing_facts, sink_tainted

    def step(
        self, node: GraphNode, method: GraphNode, incoming: FlowFacts,
        stack: tuple[int, ...],
    ) -> tuple[FlowFacts | None, bool]:
        values = dict(incoming.call_values)
        state = incoming.state
        if node.kind == "CALL":
            targets = self.resolve(node, method, state)
            if not targets:
                return incoming, False
            arguments = self.graph.ast_children(node.id)
            for argument in arguments:
                state = _prepare_lists(self.graph, argument, state, call_values=values)
            actuals = tuple(
                evaluate_expression(self.graph, argument, state, values)
                for argument in arguments
            )
            result = self.invoke_targets(targets, actuals, state, stack)
            if not result.returns:
                return None, result.sink_tainted
            values[node.id] = result.value
            # Restore caller locals while retaining callee heap effects.
            state = replace(state, lists=result.state.lists)
            return FlowFacts(state, tuple(sorted(values.items()))), result.sink_tainted
        if node.kind == "OPERATOR" and node.value == "assignment":
            children = self.graph.ast_children(node.id)
            if len(children) == 2 and children[1].kind == "METHOD_REF":
                target, reference = children
                if target.kind != "IDENTIFIER":
                    raise NotImplementedError("Nested definitions require an IDENTIFIER target")
                state = state.bind(target.value, self.nested_reference(reference, method))
                return FlowFacts(state, incoming.call_values), False
        if node.kind == "RETURN":
            for expression in self.graph.ast_children(node.id):
                state = _prepare_lists(self.graph, expression, state, call_values=values)
        state = transfer(self.graph, node, state, values)
        return FlowFacts(state, incoming.call_values), False


def analyse_interprocedural(graph: ProgramGraph, metadata: TestMetadata) -> bool:
    """Analyse direct and indirect calls, including lexical nested definitions."""
    source = graph.node(metadata.source_node)
    sink = graph.node(metadata.sink_node)
    method = graph.containing_method(source.id)
    if method is None or graph.containing_method(sink.id) is None:
        raise ValueError("Source and sink must belong to a METHOD")
    parameters = graph.method_parameters(method.id)
    if source not in parameters:
        raise NotImplementedError("Only direct method PARAMETER sources are supported")
    if sink.kind not in {"IDENTIFIER", "LITERAL", "PARAMETER", "OPERATOR", "BLOCK", "CALL"} or (
        sink.kind == "OPERATOR" and sink.value not in EXPRESSION_OPERATORS
    ):
        raise NotImplementedError(f"Unsupported sink: {sink.kind} {sink.value!r}")
    actuals = tuple(TAINTED_SCALAR if p == source else CLEAN_SCALAR for p in parameters)
    return InterproceduralAnalysis(graph, metadata).invoke(method, actuals, TaintState(), ()).sink_tainted
