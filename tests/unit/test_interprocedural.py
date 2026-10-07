"""Real-fixture acceptance tests and focused direct-call CFG regressions."""

import unittest
from pathlib import Path

from src.graph import GraphEdge, GraphNode, ProgramGraph, TestMetadata, load_test_case
from src.interprocedural import (
    FlowFacts, InterproceduralAnalysis, InvocationResult, analyse_interprocedural,
)
from src.state import AbstractValue, CLEAN_SCALAR, TAINTED_SCALAR, TaintState


CASES = Path(__file__).resolve().parents[1] / "testcases"


class FixtureTests(unittest.TestCase):
    def check_case(self, name: str, expected: bool) -> None:
        self.assertEqual(analyse_interprocedural(*load_test_case(CASES / name)), expected)

    def test_supplied_direct_calls(self) -> None:
        for name, expected in [("test_inter1", True), ("test_inter2", True),
                               ("test_inter3", False), ("test_inter4", False)]:
            with self.subTest(case=name):
                self.check_case(name, expected)

    def test_tainted_argument(self) -> None:
        self.check_case("custom_p2_01_direct_arg_to_sink_true", True)

    def test_clean_argument(self) -> None:
        self.check_case("custom_p2_02_direct_clean_arg_false", False)

    def test_tainted_return(self) -> None:
        self.check_case("custom_p2_03_return_true", True)

    def test_clean_return_despite_tainted_input(self) -> None:
        self.check_case("custom_p2_04_return_clean_false", False)

    def test_separate_clean_invocation(self) -> None:
        self.check_case("custom_p2_05_context_separation_false", False)

    def test_separate_tainted_invocation(self) -> None:
        self.check_case("custom_p2_06_context_tainted_call_true", True)

    def test_transitive_direct_calls(self) -> None:
        self.check_case("custom_p2_07_direct_nested_calls_true", True)

    def test_list_mutation(self) -> None:
        self.check_case("custom_p2_11_list_argument_alias_true", True)

    def test_strong_list_clean(self) -> None:
        self.check_case("custom_p2_12_list_argument_clean_false", False)

    def test_returned_list_reference(self) -> None:
        self.check_case("custom_p2_13_return_list_alias_true", True)

    def test_supplied_indirect_alias_call(self) -> None:
        self.check_case("test_inter5", True)

    def test_simple_indirect_return(self) -> None:
        self.check_case("custom_p2_08_indirect_simple_true", True)

    def test_flow_sensitive_function_reassignment(self) -> None:
        self.check_case("custom_p2_09_indirect_reassignment_false", False)

    def test_multi_target_branch_call(self) -> None:
        self.check_case("custom_p2_10_indirect_branch_true", True)

    def test_nested_function_call(self) -> None:
        self.check_case("custom_p2_14_nested_function_direct_true", True)

    def test_nested_function_shadows_top_level(self) -> None:
        self.check_case("custom_p2_15_shadowed_function_target", False)

    def test_part_one_fixtures_through_interprocedural_entry(self) -> None:
        expected = [True, False, True, False, True, False, True, False,
                    True, False, True, False, True, True, True, True,
                    True, True, False, False, True, True, False]
        directories = sorted(CASES.glob("custom_p1_*"))
        self.assertEqual(len(directories), len(expected))
        for directory, answer in zip(directories, expected):
            with self.subTest(case=directory.name):
                self.check_case(directory.name, answer)


class TestGraph:
    """Small explicit AST/CFG builder; never parses or executes target code."""

    def __init__(self) -> None:
        self.nodes: list[GraphNode] = []
        self.edges: list[GraphEdge] = []

    def node(self, kind: str, value: str, order: int = 1) -> int:
        node_id = len(self.nodes) + 1
        self.nodes.append(GraphNode(node_id, kind, value, order))
        return node_id

    def ast(self, parent: int, *children: int) -> None:
        # Reverse insertion makes argument/parameter ordering tests meaningful.
        self.edges.extend(GraphEdge(parent, child, "AST") for child in reversed(children))

    def cfg(self, *points: int) -> None:
        self.edges.extend(GraphEdge(a, b, "CFG") for a, b in zip(points, points[1:]))

    def method(self, name: str, *parameters: str) -> tuple[int, int, int, tuple[int, ...]]:
        method = self.node("METHOD", name)
        block = self.node("BLOCK", "[ ... ]")
        exit_node = self.node("EXIT", "EXIT", 2)
        params = tuple(self.node("PARAMETER", p, i + 1) for i, p in enumerate(parameters))
        self.ast(method, block, exit_node, *params)
        return method, block, exit_node, params

    def expression(self, kind: str, value: str, *children: int, order: int = 1) -> int:
        node = self.node(kind, value, order)
        self.ast(node, *children)
        return node

    def assign(self, block: int, name: str, expression: int) -> int:
        target = self.node("IDENTIFIER", name)
        assignment = self.expression("OPERATOR", "assignment", target, expression)
        self.ast(block, assignment)
        return assignment

    def sink(self, block: int, name: str) -> tuple[int, int]:
        argument = self.node("IDENTIFIER", name)
        call = self.expression("CALL", "sink", argument)
        self.ast(block, call)
        return call, argument

    def analyse(self, source: int, sink: int) -> bool:
        return analyse_interprocedural(ProgramGraph(self.nodes, self.edges), TestMetadata(source, sink))


class DirectCallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.g = TestGraph()
        self.main, self.block, self.exit, (self.source,) = self.g.method("main", "x")

    def returned_call(self, target: str = "helper", *arguments: int) -> tuple[int, int, int]:
        if not arguments:
            arguments = (self.g.node("IDENTIFIER", "x"),)
        call = self.g.expression("CALL", target, *arguments, order=2)
        assignment = self.g.assign(self.block, "y", call)
        sink, sink_value = self.g.sink(self.block, "y")
        self.g.cfg(self.main, call, assignment, sink, self.exit)
        return call, sink, sink_value

    def returning_method(self, name: str, expression_kind: str, expression_value: str) -> int:
        method, block, exit_node, _ = self.g.method(name, "v")
        expression = self.g.node(expression_kind, expression_value)
        returned = self.g.expression("RETURN", "RETURN", expression)
        self.g.ast(block, returned)
        self.g.cfg(method, returned, exit_node)
        return method

    def test_argument_and_parameter_order(self) -> None:
        method, block, exit_node, _ = self.g.method("helper", "a", "b")
        returned = self.g.expression("RETURN", "RETURN", self.g.node("IDENTIFIER", "b"))
        self.g.ast(block, returned)
        self.g.cfg(method, returned, exit_node)
        _, _, sink = self.returned_call("helper", self.g.node("LITERAL", "0", 1),
                                        self.g.node("IDENTIFIER", "x", 2))
        self.assertTrue(self.g.analyse(self.source, sink))

    def test_arity_mismatch(self) -> None:
        self.returning_method("helper", "IDENTIFIER", "v")
        _, _, sink = self.returned_call("helper", self.g.node("IDENTIFIER", "x", 1),
                                        self.g.node("LITERAL", "0", 2))
        with self.assertRaisesRegex(ValueError, "expected 1 arguments, got 2"):
            self.g.analyse(self.source, sink)

    def test_missing_actual(self) -> None:
        self.returning_method("helper", "IDENTIFIER", "v")
        call = self.g.expression("CALL", "helper")
        self.g.ast(self.block, call)
        sink, value = self.g.sink(self.block, "x")
        self.g.cfg(self.main, call, sink, self.exit)
        with self.assertRaisesRegex(ValueError, "got 0"):
            self.g.analyse(self.source, value)

    def test_multiple_return_paths_join(self) -> None:
        method, block, exit_node, _ = self.g.method("helper", "v")
        tainted = self.g.expression("RETURN", "RETURN", self.g.node("IDENTIFIER", "v"))
        clean = self.g.expression("RETURN", "RETURN", self.g.node("LITERAL", "0"))
        self.g.ast(block, tainted, clean)
        self.g.cfg(method, tainted, exit_node)
        self.g.cfg(method, clean, exit_node)
        _, _, sink = self.returned_call()
        self.assertTrue(self.g.analyse(self.source, sink))

    def test_implicit_none_is_clean(self) -> None:
        method, _, exit_node, _ = self.g.method("helper", "v")
        self.g.cfg(method, exit_node)
        _, _, sink = self.returned_call()
        self.assertFalse(self.g.analyse(self.source, sink))

    def test_bare_return_is_clean(self) -> None:
        method, block, exit_node, _ = self.g.method("helper", "v")
        returned = self.g.expression("RETURN", "RETURN")
        self.g.ast(block, returned)
        self.g.cfg(method, returned, exit_node)
        _, _, sink = self.returned_call()
        self.assertFalse(self.g.analyse(self.source, sink))

    def test_callee_locals_do_not_overwrite_caller(self) -> None:
        method, block, exit_node, _ = self.g.method("helper", "v")
        assignment = self.g.assign(block, "x", self.g.node("LITERAL", "0", 2))
        self.g.cfg(method, assignment, exit_node)
        call = self.g.expression("CALL", "helper", self.g.node("IDENTIFIER", "x"))
        self.g.ast(self.block, call)
        sink, value = self.g.sink(self.block, "x")
        self.g.cfg(self.main, call, sink, self.exit)
        self.assertTrue(self.g.analyse(self.source, value))

    def test_sink_is_ored_across_invocations(self) -> None:
        method, block, exit_node, _ = self.g.method("helper", "v")
        sink, value = self.g.sink(block, "v")
        self.g.cfg(method, sink, exit_node)
        tainted = self.g.expression("CALL", "helper", self.g.node("IDENTIFIER", "x"))
        clean = self.g.expression("CALL", "helper", self.g.node("LITERAL", "0"))
        self.g.ast(self.block, tainted, clean)
        self.g.cfg(self.main, tainted, clean, self.exit)
        self.assertTrue(self.g.analyse(self.source, value))

    def test_uncalled_sink_is_unreachable(self) -> None:
        method, block, exit_node, _ = self.g.method("helper", "v")
        sink, value = self.g.sink(block, "v")
        self.g.cfg(method, sink, exit_node)
        self.g.cfg(self.main, self.exit)
        self.assertFalse(self.g.analyse(self.source, value))

    def test_loop_revisits_call_result(self) -> None:
        self.returning_method("helper", "IDENTIFIER", "v")
        initial = self.g.assign(self.block, "y", self.g.node("LITERAL", "0", 2))
        call = self.g.expression("CALL", "helper", self.g.node("IDENTIFIER", "y"), order=2)
        assignment = self.g.assign(self.block, "z", call)
        taint = self.g.assign(self.block, "y", self.g.node("IDENTIFIER", "x", 2))
        condition = self.g.node("IDENTIFIER", "c")
        self.g.ast(self.block, condition)
        sink, value = self.g.sink(self.block, "z")
        self.g.cfg(self.main, initial, call, assignment, taint, condition, call)
        self.g.cfg(condition, sink, self.exit)
        self.assertTrue(self.g.analyse(self.source, value))

    def test_nested_call_argument(self) -> None:
        self.returning_method("helper", "IDENTIFIER", "v")
        inner = self.g.expression("CALL", "helper", self.g.node("IDENTIFIER", "x"))
        outer = self.g.expression("CALL", "helper", inner, order=2)
        assignment = self.g.assign(self.block, "y", outer)
        sink, value = self.g.sink(self.block, "y")
        self.g.cfg(self.main, inner, outer, assignment, sink, self.exit)
        self.assertTrue(self.g.analyse(self.source, value))

    def test_call_result_in_operator(self) -> None:
        self.returning_method("helper", "IDENTIFIER", "v")
        call = self.g.expression("CALL", "helper", self.g.node("IDENTIFIER", "x"))
        addition = self.g.expression("OPERATOR", "addition", call,
                                     self.g.node("LITERAL", "1", 2), order=2)
        assignment = self.g.assign(self.block, "y", addition)
        sink, value = self.g.sink(self.block, "y")
        self.g.cfg(self.main, call, addition, assignment, sink, self.exit)
        self.assertTrue(self.g.analyse(self.source, value))

    def test_recursive_call_rejected(self) -> None:
        _, _, sink = self.returned_call("main")
        with self.assertRaisesRegex(NotImplementedError, "Recursive/cyclic"):
            self.g.analyse(self.source, sink)

    def test_call_cycle_rejected(self) -> None:
        for name, target in [("first", "second"), ("second", "first")]:
            method, block, exit_node, _ = self.g.method(name, "v")
            call = self.g.expression("CALL", target, self.g.node("IDENTIFIER", "v"))
            self.g.ast(block, call)
            self.g.cfg(method, call, exit_node)
        _, _, sink = self.returned_call("first")
        with self.assertRaisesRegex(NotImplementedError, "Recursive/cyclic"):
            self.g.analyse(self.source, sink)

    def test_unknown_statement_call_rejected(self) -> None:
        call = self.g.expression("CALL", "missing")
        self.g.ast(self.block, call)
        sink, value = self.g.sink(self.block, "x")
        self.g.cfg(self.main, call, sink, self.exit)
        with self.assertRaisesRegex(NotImplementedError, "Unresolved"):
            self.g.analyse(self.source, value)

    def test_print_statement_preserves_state(self) -> None:
        call = self.g.expression("CALL", "print", self.g.node("IDENTIFIER", "x"))
        self.g.ast(self.block, call)
        sink, value = self.g.sink(self.block, "x")
        self.g.cfg(self.main, call, sink, self.exit)
        self.assertTrue(self.g.analyse(self.source, value))

    def test_external_return_value_rejected(self) -> None:
        _, _, sink = self.returned_call("print")
        with self.assertRaises(NotImplementedError):
            self.g.analyse(self.source, sink)

    def test_local_binding_cannot_resolve_as_top_level(self) -> None:
        self.returning_method("helper", "IDENTIFIER", "v")
        assignment = self.g.assign(self.block, "helper", self.g.node("LITERAL", "0", 2))
        call = self.g.expression("CALL", "helper", self.g.node("IDENTIFIER", "x"))
        self.g.ast(self.block, call)
        sink, value = self.g.sink(self.block, "x")
        self.g.cfg(self.main, assignment, call, sink, self.exit)
        with self.assertRaisesRegex(NotImplementedError, "requires function targets"):
            self.g.analyse(self.source, value)

    def test_duplicate_top_level_targets_rejected(self) -> None:
        self.returning_method("helper", "IDENTIFIER", "v")
        self.returning_method("helper", "LITERAL", "0")
        _, _, sink = self.returned_call()
        with self.assertRaisesRegex(NotImplementedError, "Ambiguous"):
            self.g.analyse(self.source, sink)

    def test_sink_on_assignment_target_uses_call_result(self) -> None:
        self.returning_method("helper", "IDENTIFIER", "v")
        call, _, _ = self.returned_call()
        assignment = next(e.source for e in self.g.edges if e.kind == "AST" and e.destination == call)
        target = next(e.destination for e in self.g.edges
                      if e.kind == "AST" and e.source == assignment and e.destination != call)
        self.assertTrue(self.g.analyse(self.source, target))

    def test_returned_list_keeps_alias_after_caller_write(self) -> None:
        self.returning_method("helper", "IDENTIFIER", "v")
        literal = self.g.expression("OPERATOR", "listLiteral", self.g.node("LITERAL", "0"), order=2)
        allocate = self.g.assign(self.block, "a", literal)
        call = self.g.expression("CALL", "helper", self.g.node("IDENTIFIER", "a"), order=2)
        bind = self.g.assign(self.block, "b", call)
        access = self.g.expression("OPERATOR", "indexAccess", self.g.node("IDENTIFIER", "b"),
                                   self.g.node("LITERAL", "0", 2))
        write = self.g.expression("OPERATOR", "assignment", access,
                                  self.g.node("IDENTIFIER", "x", 2))
        self.g.ast(self.block, write)
        sink, value = self.g.sink(self.block, "a")
        self.g.cfg(self.main, literal, allocate, call, bind, access, write, sink, self.exit)
        self.assertTrue(self.g.analyse(self.source, value))

    def test_callee_allocated_list_is_returned(self) -> None:
        method, block, exit_node, _ = self.g.method("helper", "v")
        literal = self.g.expression("OPERATOR", "listLiteral", self.g.node("IDENTIFIER", "v"))
        returned = self.g.expression("RETURN", "RETURN", literal)
        self.g.ast(block, returned)
        self.g.cfg(method, literal, returned, exit_node)
        _, _, sink = self.returned_call()
        self.assertTrue(self.g.analyse(self.source, sink))

    def test_list_literal_can_contain_call_result(self) -> None:
        self.returning_method("helper", "IDENTIFIER", "v")
        call = self.g.expression("CALL", "helper", self.g.node("IDENTIFIER", "x"))
        literal = self.g.expression("OPERATOR", "listLiteral", call, order=2)
        assignment = self.g.assign(self.block, "y", literal)
        sink, value = self.g.sink(self.block, "y")
        self.g.cfg(self.main, call, literal, assignment, sink, self.exit)
        self.assertTrue(self.g.analyse(self.source, value))

    def test_sink_on_call_result(self) -> None:
        self.returning_method("helper", "IDENTIFIER", "v")
        call, _, _ = self.returned_call()
        self.assertTrue(self.g.analyse(self.source, call))

    def test_unreachable_recursive_call_is_ignored(self) -> None:
        call = self.g.expression("CALL", "main", self.g.node("IDENTIFIER", "x"))
        self.g.ast(self.block, call)
        sink, value = self.g.sink(self.block, "x")
        self.g.cfg(self.main, sink, self.exit)
        self.assertTrue(self.g.analyse(self.source, value))

    def test_nonreturning_callee_prunes_caller_continuation(self) -> None:
        method, block, _, _ = self.g.method("helper", "v")
        loop = self.g.node("IDENTIFIER", "c")
        self.g.ast(block, loop)
        self.g.cfg(method, loop, loop)
        _, _, sink = self.returned_call()
        self.assertFalse(self.g.analyse(self.source, sink))

    def test_sink_in_nonreturning_callee_is_observed(self) -> None:
        method, block, _, _ = self.g.method("helper", "v")
        sink, value = self.g.sink(block, "v")
        self.g.cfg(method, sink, sink)
        call = self.g.expression("CALL", "helper", self.g.node("IDENTIFIER", "x"))
        self.g.ast(self.block, call)
        self.g.cfg(self.main, call, self.exit)
        self.assertTrue(self.g.analyse(self.source, value))

    def test_conditional_calls_with_unequal_cfg_path_lengths(self) -> None:
        self.returning_method("helper", "IDENTIFIER", "v")
        condition = self.g.node("IDENTIFIER", "c", 1)
        clean = self.g.expression("CALL", "helper", self.g.node("LITERAL", "0"), order=2)
        tainted = self.g.expression("CALL", "helper", self.g.node("IDENTIFIER", "x"), order=3)
        conditional = self.g.expression("OPERATOR", "conditional", condition, clean, tainted, order=2)
        assignment = self.g.assign(self.block, "y", conditional)
        delay = self.g.node("IDENTIFIER", "c")
        self.g.ast(self.block, delay)
        sink, value = self.g.sink(self.block, "y")
        self.g.cfg(self.main, condition, clean, conditional, assignment, sink, self.exit)
        self.g.cfg(condition, delay, tainted, conditional)
        self.assertTrue(self.g.analyse(self.source, value))

    def test_call_result_change_reschedules_even_when_heap_is_cleaned(self) -> None:
        method, block, exit_node, _ = self.g.method("helper", "a")
        read = self.g.expression("OPERATOR", "indexAccess", self.g.node("IDENTIFIER", "a"),
                                 self.g.node("LITERAL", "0", 2), order=2)
        save = self.g.assign(block, "saved", read)
        target = self.g.expression("OPERATOR", "indexAccess", self.g.node("IDENTIFIER", "a"),
                                   self.g.node("LITERAL", "0", 2))
        clean = self.g.expression("OPERATOR", "assignment", target,
                                  self.g.node("LITERAL", "0", 2))
        returned = self.g.expression("RETURN", "RETURN", self.g.node("IDENTIFIER", "saved"))
        self.g.ast(block, clean, returned)
        self.g.cfg(method, read, save, target, clean, returned, exit_node)
        literal = self.g.expression("OPERATOR", "listLiteral", self.g.node("LITERAL", "0"), order=2)
        allocate = self.g.assign(self.block, "a", literal)
        initial = self.g.assign(self.block, "z", self.g.node("LITERAL", "0", 2))
        call = self.g.expression("CALL", "helper", self.g.node("IDENTIFIER", "a"), order=2)
        assignment = self.g.assign(self.block, "z", call)
        access = self.g.expression("OPERATOR", "indexAccess", self.g.node("IDENTIFIER", "a"),
                                   self.g.node("LITERAL", "0", 2))
        taint = self.g.expression("OPERATOR", "assignment", access,
                                  self.g.node("IDENTIFIER", "x", 2))
        condition = self.g.node("IDENTIFIER", "c")
        self.g.ast(self.block, taint, condition)
        sink, value = self.g.sink(self.block, "z")
        self.g.cfg(self.main, literal, allocate, initial, call, assignment, access, taint, condition, call)
        self.g.cfg(condition, sink, self.exit)
        self.assertTrue(self.g.analyse(self.source, value))

    def test_unsequenced_call_result_is_not_assumed_clean(self) -> None:
        self.returning_method("helper", "IDENTIFIER", "v")
        call = self.g.expression("CALL", "helper", self.g.node("IDENTIFIER", "x"), order=2)
        assignment = self.g.assign(self.block, "y", call)
        sink, value = self.g.sink(self.block, "y")
        self.g.cfg(self.main, assignment, sink, self.exit)
        with self.assertRaisesRegex(NotImplementedError, "Unsupported expression: CALL"):
            self.g.analyse(self.source, value)


class FunctionValueTests(unittest.TestCase):
    def test_function_reference_join_unions_method_ids(self) -> None:
        foo = AbstractValue(function_methods=frozenset({10}))
        bar = AbstractValue(function_methods=frozenset({20}))
        self.assertEqual(foo.join(bar).function_methods, frozenset({10, 20}))
        self.assertEqual(foo.join(foo), foo)
        self.assertEqual(foo.join(bar), bar.join(foo))
        self.assertEqual(foo.function_methods, frozenset({10}))

    def test_mixed_join_preserves_existing_scalar_and_list_fields(self) -> None:
        function = AbstractValue(function_methods=frozenset({10}))
        data = AbstractValue(scalar_tainted=True, list_objects=frozenset({20}), may_be_scalar=True)
        joined = function.join(data)
        self.assertEqual(joined.function_methods, frozenset({10}))
        self.assertEqual(joined.list_objects, data.list_objects)
        self.assertTrue(joined.scalar_tainted)
        self.assertTrue(joined.may_be_scalar)

    def test_bind_replaces_function_reference(self) -> None:
        foo = AbstractValue(function_methods=frozenset({10}))
        bar = AbstractValue(function_methods=frozenset({20}))
        initial = TaintState().bind("f", foo)
        overwritten = initial.bind("f", bar)
        self.assertEqual(overwritten.value("f"), bar)
        self.assertEqual(initial.value("f"), foo)
        self.assertEqual(overwritten.bind("f", CLEAN_SCALAR).value("f"), CLEAN_SCALAR)

    def test_state_and_flow_joins_union_function_references(self) -> None:
        left = TaintState().bind("f", AbstractValue(function_methods=frozenset({10})))
        right = TaintState().bind("f", AbstractValue(function_methods=frozenset({20})))
        self.assertEqual(left.join(right).value("f").function_methods, frozenset({10, 20}))
        self.assertEqual(FlowFacts(left).join(FlowFacts(right)).state, left.join(right))

    def test_function_reference_is_clean_even_inside_list(self) -> None:
        function = AbstractValue(function_methods=frozenset({10}))
        state = TaintState().allocate(20, (function,))
        self.assertFalse(state.value_is_tainted(function))
        self.assertFalse(state.value_is_tainted(AbstractValue(list_objects=frozenset({20}))))
        self.assertTrue(state.value_is_tainted(function.join(TAINTED_SCALAR)))

    def test_function_scalar_and_list_rebindings_are_independent(self) -> None:
        function = AbstractValue(function_methods=frozenset({10}))
        state = TaintState().taint("x").allocate(20, (CLEAN_SCALAR,)).bind("f", function)
        state = state.bind("a", AbstractValue(list_objects=frozenset({20})))
        self.assertTrue(state.is_tainted("x"))
        self.assertFalse(state.is_tainted("f"))
        self.assertFalse(state.is_tainted("a"))
        self.assertEqual(state.value("a").function_methods, frozenset())
        self.assertEqual(state.value("f").list_objects, frozenset())


class FunctionReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.g = TestGraph()
        self.main, self.block, self.exit, (self.source,) = self.g.method("main", "x")
        self.foo = self.make_function("foo", "IDENTIFIER", "v")
        self.bar = self.make_function("bar", "LITERAL", "0")

    def make_function(self, name: str, kind: str, value: str) -> int:
        method, block, exit_node, _ = self.g.method(name, "v")
        returned = self.g.expression("RETURN", "RETURN", self.g.node(kind, value))
        self.g.ast(block, returned)
        self.g.cfg(method, returned, exit_node)
        return method

    def alias(self, target: str, value: str) -> int:
        return self.g.assign(self.block, target, self.g.node("IDENTIFIER", value, 2))

    def call_and_sink(self, name: str, *preceding: int) -> bool:
        call = self.g.expression("CALL", name, self.g.node("IDENTIFIER", "x"), order=2)
        assignment = self.g.assign(self.block, "y", call)
        sink, value = self.g.sink(self.block, "y")
        self.g.cfg(self.main, *preceding, call, assignment, sink, self.exit)
        return self.g.analyse(self.source, value)

    def resolver(self) -> tuple[InterproceduralAnalysis, ProgramGraph, int]:
        self.alias("f", "foo")
        call = self.g.expression("CALL", "f", self.g.node("IDENTIFIER", "x"))
        self.g.ast(self.block, call)
        graph = ProgramGraph(self.g.nodes, self.g.edges)
        return InterproceduralAnalysis(graph, TestMetadata(self.source, self.source)), graph, call

    def test_top_level_reference_assignment_then_indirect_return(self) -> None:
        assignment = self.alias("f", "foo")
        self.assertTrue(self.call_and_sink("f", assignment))

    def test_function_reference_copy(self) -> None:
        first = self.alias("f", "foo")
        second = self.alias("g", "f")
        self.assertTrue(self.call_and_sink("g", first, second))

    def test_multiple_function_reference_copies(self) -> None:
        points = [self.alias("f", "foo"), self.alias("g", "f"), self.alias("h", "g")]
        self.assertTrue(self.call_and_sink("h", *points))

    def test_function_reassignment_changes_target(self) -> None:
        first = self.alias("f", "foo")
        second = self.alias("f", "bar")
        self.assertFalse(self.call_and_sink("f", first, second))

    def test_function_to_scalar_reassignment_prevents_call(self) -> None:
        first = self.alias("f", "foo")
        second = self.g.assign(self.block, "f", self.g.node("LITERAL", "0", 2))
        with self.assertRaisesRegex(NotImplementedError, "requires function targets"):
            self.call_and_sink("f", first, second)

    def test_function_to_list_reassignment_prevents_call(self) -> None:
        first = self.alias("f", "foo")
        literal = self.g.expression("OPERATOR", "listLiteral", self.g.node("LITERAL", "0"), order=2)
        second = self.g.assign(self.block, "f", literal)
        with self.assertRaisesRegex(NotImplementedError, "requires function targets"):
            self.call_and_sink("f", first, literal, second)

    def test_function_variable_can_shadow_top_level_name(self) -> None:
        assignment = self.alias("foo", "bar")
        self.assertFalse(self.call_and_sink("foo", assignment))

    def test_empty_or_mixed_local_target_is_rejected(self) -> None:
        analysis, graph, call = self.resolver()
        function = AbstractValue(function_methods=frozenset({self.foo}))
        for value in [AbstractValue(), CLEAN_SCALAR, function.join(CLEAN_SCALAR),
                      function.join(AbstractValue(list_objects=frozenset({100})))]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(NotImplementedError, "requires function targets"):
                    analysis.resolve(graph.node(call), graph.node(self.main), TaintState().bind("f", value))

    def test_single_target_resolution_uses_method_id(self) -> None:
        analysis, graph, call = self.resolver()
        value = AbstractValue(function_methods=frozenset({self.bar}))
        target = analysis.resolve(graph.node(call), graph.node(self.main), TaintState().bind("f", value))
        self.assertEqual(target, (graph.node(self.bar),))

    def test_multi_target_resolution_returns_all_method_ids(self) -> None:
        analysis, graph, call = self.resolver()
        value = AbstractValue(function_methods=frozenset({self.foo, self.bar}))
        targets = analysis.resolve(graph.node(call), graph.node(self.main), TaintState().bind("f", value))
        self.assertEqual({target.id for target in targets}, {self.foo, self.bar})

    def test_branch_fixture_has_both_targets_at_call(self) -> None:
        from unittest.mock import patch

        graph, metadata = load_test_case(CASES / "custom_p2_10_indirect_branch_true")
        analysis = InterproceduralAnalysis(graph, metadata)
        method = graph.containing_method(metadata.source_node)
        actuals = tuple(
            TAINTED_SCALAR if p.id == metadata.source_node else CLEAN_SCALAR
            for p in graph.method_parameters(method.id)
        )
        recorded: list[AbstractValue] = []
        resolve = analysis.resolve

        def record(call: GraphNode, owner: GraphNode, state: TaintState) -> tuple[GraphNode, ...]:
            if call.value == "f":
                recorded.append(state.value("f"))
            return resolve(call, owner, state)

        with patch.object(analysis, "resolve", side_effect=record):
            self.assertTrue(analysis.invoke(method, actuals, TaintState(), ()).sink_tainted)
        expected_ids = frozenset(m.id for m in graph.methods if m.value in {"identity", "clean"})
        self.assertEqual(recorded[-1].function_methods, expected_ids)
        self.assertFalse(recorded[-1].may_be_scalar)

    def test_function_reference_sink_is_clean(self) -> None:
        assignment = self.alias("f", "foo")
        sink, value = self.g.sink(self.block, "f")
        self.g.cfg(self.main, assignment, sink, self.exit)
        self.assertFalse(self.g.analyse(self.source, value))

    def test_unassigned_local_name_does_not_fall_back_to_top_level(self) -> None:
        # The disconnected assignment still declares a local name.
        self.g.assign(self.block, "foo", self.g.node("LITERAL", "0", 2))
        with self.assertRaisesRegex(NotImplementedError, "requires function targets"):
            self.call_and_sink("foo")

    def test_indirect_list_mutation_uses_shared_heap(self) -> None:
        method, block, exit_node, _ = self.g.method("mutate", "a", "v")
        access = self.g.expression("OPERATOR", "indexAccess", self.g.node("IDENTIFIER", "a"),
                                   self.g.node("LITERAL", "0", 2))
        write = self.g.expression("OPERATOR", "assignment", access,
                                  self.g.node("IDENTIFIER", "v", 2))
        self.g.ast(block, write)
        self.g.cfg(method, access, write, exit_node)
        alias = self.alias("f", "mutate")
        literal = self.g.expression("OPERATOR", "listLiteral", self.g.node("LITERAL", "0"), order=2)
        allocate = self.g.assign(self.block, "a", literal)
        call = self.g.expression("CALL", "f", self.g.node("IDENTIFIER", "a", 1),
                                 self.g.node("IDENTIFIER", "x", 2))
        self.g.ast(self.block, call)
        sink, value = self.g.sink(self.block, "a")
        self.g.cfg(self.main, alias, literal, allocate, call, sink, self.exit)
        self.assertTrue(self.g.analyse(self.source, value))

    def test_function_parameter_can_be_called(self) -> None:
        method, block, exit_node, _ = self.g.method("apply", "f", "v")
        call = self.g.expression("CALL", "f", self.g.node("IDENTIFIER", "v"))
        returned = self.g.expression("RETURN", "RETURN", call)
        self.g.ast(block, returned)
        self.g.cfg(method, call, returned, exit_node)
        outer = self.g.expression("CALL", "apply", self.g.node("IDENTIFIER", "foo", 1),
                                  self.g.node("IDENTIFIER", "x", 2), order=2)
        assignment = self.g.assign(self.block, "y", outer)
        sink, value = self.g.sink(self.block, "y")
        self.g.cfg(self.main, outer, assignment, sink, self.exit)
        self.assertTrue(self.g.analyse(self.source, value))

    def test_function_return_can_be_called(self) -> None:
        self.make_function("choose", "IDENTIFIER", "foo")
        choose = self.g.expression("CALL", "choose", self.g.node("LITERAL", "0"), order=2)
        bind = self.g.assign(self.block, "f", choose)
        self.assertTrue(self.call_and_sink("f", choose, bind))

    def test_indirect_recursion_remains_rejected(self) -> None:
        assignment = self.alias("f", "main")
        with self.assertRaisesRegex(NotImplementedError, "Recursive/cyclic"):
            self.call_and_sink("f", assignment)

    def test_function_arithmetic_is_not_silently_scalarised(self) -> None:
        addition = self.g.expression("OPERATOR", "addition", self.g.node("IDENTIFIER", "foo"),
                                     self.g.node("LITERAL", "1", 2), order=2)
        assignment = self.g.assign(self.block, "f", addition)
        sink, value = self.g.sink(self.block, "f")
        self.g.cfg(self.main, addition, assignment, sink, self.exit)
        with self.assertRaisesRegex(NotImplementedError, "Function arithmetic"):
            self.g.analyse(self.source, value)

    def test_part_one_identifier_semantics_remain_local(self) -> None:
        from src.intraprocedural import analyse_intraprocedural

        assignment = self.alias("f", "foo")
        sink, value = self.g.sink(self.block, "f")
        self.g.cfg(self.main, assignment, sink, self.exit)
        graph = ProgramGraph(self.g.nodes, self.g.edges)
        self.assertFalse(analyse_intraprocedural(graph, TestMetadata(self.source, value)))

    def test_function_copy_keeps_value_after_original_is_rebound(self) -> None:
        first = self.alias("f", "foo")
        copy = self.alias("g", "f")
        rebind = self.alias("f", "bar")
        self.assertTrue(self.call_and_sink("g", first, copy, rebind))

    def test_branch_join_with_same_target_remains_definite(self) -> None:
        condition = self.g.node("IDENTIFIER", "c")
        self.g.ast(self.block, condition)
        left = self.alias("f", "foo")
        right = self.alias("f", "foo")
        call = self.g.expression("CALL", "f", self.g.node("IDENTIFIER", "x"), order=2)
        assignment = self.g.assign(self.block, "y", call)
        sink, value = self.g.sink(self.block, "y")
        self.g.cfg(self.main, condition, left, call, assignment, sink, self.exit)
        self.g.cfg(condition, right, call)
        self.assertTrue(self.g.analyse(self.source, value))


class MultiTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.g = TestGraph()
        self.main, self.block, self.exit, (self.source,) = self.g.method("main", "x")

    def returning(self, name: str, kind: str, value: str) -> int:
        method, block, exit_node, _ = self.g.method(name, "v")
        returned = self.g.expression("RETURN", "RETURN", self.g.node(kind, value))
        self.g.ast(block, returned)
        self.g.cfg(method, returned, exit_node)
        return method

    def mutator(self, name: str, kind: str, value: str, diverges: bool = False) -> int:
        method, block, exit_node, _ = self.g.method(name, "a", "v")
        access = self.g.expression("OPERATOR", "indexAccess", self.g.node("IDENTIFIER", "a"),
                                   self.g.node("LITERAL", "0", 2))
        write = self.g.expression("OPERATOR", "assignment", access, self.g.node(kind, value, 2))
        self.g.ast(block, write)
        if diverges:
            loop = self.g.node("IDENTIFIER", "c")
            self.g.ast(block, loop)
            self.g.cfg(method, access, write, loop, loop)
        else:
            self.g.cfg(method, access, write, exit_node)
        return method

    def invoke(self, targets: tuple[int, ...], actuals: tuple[AbstractValue, ...],
               caller: TaintState = TaintState(), sink: int | None = None) -> InvocationResult:
        graph = ProgramGraph(self.g.nodes, self.g.edges)
        analysis = InterproceduralAnalysis(
            graph, TestMetadata(self.source, self.source if sink is None else sink)
        )
        return analysis.invoke_targets(tuple(graph.node(i) for i in targets), actuals, caller, ())

    def test_tainted_and_clean_returns_join(self) -> None:
        identity = self.returning("identity", "IDENTIFIER", "v")
        clean = self.returning("clean", "LITERAL", "0")
        result = self.invoke((identity, clean), (TAINTED_SCALAR,))
        self.assertTrue(result.value.scalar_tainted)
        self.assertTrue(result.returns)

    def test_two_clean_returns_remain_clean(self) -> None:
        first = self.returning("first", "LITERAL", "0")
        second = self.returning("second", "LITERAL", "1")
        result = self.invoke((first, second), (TAINTED_SCALAR,))
        self.assertEqual(result.value, CLEAN_SCALAR)

    def test_list_side_effects_join_without_local_leaks(self) -> None:
        taint = self.mutator("taint", "IDENTIFIER", "v")
        clean = self.mutator("clean", "LITERAL", "0")
        caller = TaintState().allocate(1000, (CLEAN_SCALAR,)).taint("caller")
        reference = AbstractValue(list_objects=frozenset({1000}))
        result = self.invoke((taint, clean), (reference, TAINTED_SCALAR), caller)
        self.assertTrue(result.state.list_memory(1000).read(0).scalar_tainted)
        self.assertEqual(result.state.bindings, caller.bindings)
        self.assertFalse(caller.list_memory(1000).read(0).scalar_tainted)

    def test_alternatives_start_from_same_heap(self) -> None:
        clean = self.mutator("clean", "LITERAL", "0")
        passive, _, exit_node, _ = self.g.method("passive", "a", "v")
        self.g.cfg(passive, exit_node)
        caller = TaintState().allocate(1000, (TAINTED_SCALAR,))
        actuals = (AbstractValue(list_objects=frozenset({1000})), CLEAN_SCALAR)
        for targets in [(clean, passive), (passive, clean)]:
            result = self.invoke(targets, actuals, caller)
            self.assertTrue(result.state.list_memory(1000).read(0).scalar_tainted)

    def test_nonreturning_alternative_does_not_contribute_heap(self) -> None:
        diverging = self.mutator("diverging", "IDENTIFIER", "v", diverges=True)
        clean = self.mutator("clean", "LITERAL", "0")
        caller = TaintState().allocate(1000, (CLEAN_SCALAR,))
        actuals = (AbstractValue(list_objects=frozenset({1000})), TAINTED_SCALAR)
        result = self.invoke((diverging, clean), actuals, caller)
        self.assertTrue(result.returns)
        self.assertFalse(result.state.list_memory(1000).read(0).scalar_tainted)

    def test_all_nonreturning_targets_prune_continuation(self) -> None:
        first = self.mutator("first", "IDENTIFIER", "v", diverges=True)
        second = self.mutator("second", "LITERAL", "0", diverges=True)
        caller = TaintState().allocate(1000, (CLEAN_SCALAR,))
        actuals = (AbstractValue(list_objects=frozenset({1000})), TAINTED_SCALAR)
        self.assertFalse(self.invoke((first, second), actuals, caller).returns)

    def test_sink_in_only_one_nonreturning_alternative_is_observed(self) -> None:
        diverging, block, _, _ = self.g.method("diverging", "v")
        sink, value = self.g.sink(block, "v")
        self.g.cfg(diverging, sink, sink)
        returning = self.returning("returning", "LITERAL", "0")
        result = self.invoke((returning, diverging), (TAINTED_SCALAR,), sink=value)
        self.assertTrue(result.sink_tainted)
        self.assertTrue(result.returns)
        self.assertEqual(result.value, CLEAN_SCALAR)

    def test_any_target_arity_mismatch_is_explicit(self) -> None:
        first = self.returning("first", "IDENTIFIER", "v")
        other, _, exit_node, _ = self.g.method("other", "a", "b")
        self.g.cfg(other, exit_node)
        with self.assertRaisesRegex(ValueError, "expected 2 arguments, got 1"):
            self.invoke((first, other), (TAINTED_SCALAR,))
        with self.assertRaisesRegex(ValueError, "got 0"):
            self.invoke((first, other), ())

    def test_multi_target_cycle_remains_explicit(self) -> None:
        other = self.returning("other", "IDENTIFIER", "v")
        graph = ProgramGraph(self.g.nodes, self.g.edges)
        analysis = InterproceduralAnalysis(graph, TestMetadata(self.source, self.source))
        with self.assertRaisesRegex(NotImplementedError, "Recursive/cyclic"):
            analysis.invoke_targets((graph.node(other), graph.node(self.main)),
                                    (TAINTED_SCALAR,), TaintState(), (self.main,))

    def test_multi_target_call_in_loop_revisits_result(self) -> None:
        identity = self.returning("identity", "IDENTIFIER", "v")
        clean = self.returning("clean", "LITERAL", "0")
        f = self.g.node("PARAMETER", "f", 2)
        self.g.ast(self.main, f)
        initial = self.g.assign(self.block, "y", self.g.node("LITERAL", "0", 2))
        call = self.g.expression("CALL", "f", self.g.node("IDENTIFIER", "y"), order=2)
        bind = self.g.assign(self.block, "z", call)
        taint = self.g.assign(self.block, "y", self.g.node("IDENTIFIER", "x", 2))
        condition = self.g.node("IDENTIFIER", "c")
        self.g.ast(self.block, condition)
        sink, value = self.g.sink(self.block, "z")
        self.g.cfg(self.main, initial, call, bind, taint, condition, call)
        self.g.cfg(condition, sink, self.exit)
        graph = ProgramGraph(self.g.nodes, self.g.edges)
        analysis = InterproceduralAnalysis(graph, TestMetadata(self.source, value))
        functions = AbstractValue(function_methods=frozenset({identity, clean}))
        self.assertTrue(analysis.invoke(graph.node(self.main), (TAINTED_SCALAR, functions),
                                        TaintState(), ()).sink_tainted)


class NestedFunctionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.g = TestGraph()
        self.main, self.block, self.exit, (self.source,) = self.g.method("main", "x")

    def nested(self, owner: int, block: int, name: str, kind: str = "IDENTIFIER",
               value: str = "v") -> tuple[int, int]:
        method, body, exit_node, _ = self.g.method(name, "v")
        self.g.ast(owner, method)
        returned = self.g.expression("RETURN", "RETURN", self.g.node(kind, value))
        self.g.ast(body, returned)
        self.g.cfg(method, returned, exit_node)
        definition = self.g.assign(block, name, self.g.node("METHOD_REF", name, 2))
        return method, definition

    def call_and_sink(self, name: str, *preceding: int) -> bool:
        call = self.g.expression("CALL", name, self.g.node("IDENTIFIER", "x"), order=2)
        bind = self.g.assign(self.block, "y", call)
        sink, value = self.g.sink(self.block, "y")
        self.g.cfg(self.main, *preceding, call, bind, sink, self.exit)
        return self.g.analyse(self.source, value)

    def test_nested_definition_binds_lexical_method_id(self) -> None:
        nested, definition = self.nested(self.main, self.block, "inner")
        graph = ProgramGraph(self.g.nodes, self.g.edges)
        analysis = InterproceduralAnalysis(graph, TestMetadata(self.source, self.source))
        facts, _ = analysis.step(graph.node(definition), graph.node(self.main),
                                 FlowFacts(TaintState()), (self.main,))
        self.assertEqual(facts.state.value("inner").function_methods, frozenset({nested}))

    def test_nested_reference_can_be_copied(self) -> None:
        _, definition = self.nested(self.main, self.block, "inner")
        first = self.g.assign(self.block, "f", self.g.node("IDENTIFIER", "inner", 2))
        second = self.g.assign(self.block, "g", self.g.node("IDENTIFIER", "f", 2))
        self.assertTrue(self.call_and_sink("g", definition, first, second))

    def test_call_before_definition_cannot_fall_back_to_module_function(self) -> None:
        root, body, exit_node, _ = self.g.method("inner", "v")
        returned = self.g.expression("RETURN", "RETURN", self.g.node("IDENTIFIER", "v"))
        self.g.ast(body, returned)
        self.g.cfg(root, returned, exit_node)
        self.nested(self.main, self.block, "inner", "LITERAL", "0")
        with self.assertRaisesRegex(NotImplementedError, "requires function targets"):
            self.call_and_sink("inner")

    def test_same_name_nested_and_module_methods_keep_distinct_ids(self) -> None:
        root, _, root_exit, _ = self.g.method("f", "v")
        self.g.cfg(root, root_exit)
        nested, definition = self.nested(self.main, self.block, "f", "LITERAL", "0")
        self.assertNotEqual(root, nested)
        graph = ProgramGraph(self.g.nodes, self.g.edges)
        analysis = InterproceduralAnalysis(graph, TestMetadata(self.source, self.source))
        reference = next(n for n in graph.ast_children(definition) if n.kind == "METHOD_REF")
        self.assertEqual(analysis.nested_reference(reference, graph.node(self.main)).function_methods,
                         frozenset({nested}))

    def test_duplicate_nested_names_are_explicitly_ambiguous(self) -> None:
        self.nested(self.main, self.block, "inner")
        _, definition = self.nested(self.main, self.block, "inner", "LITERAL", "0")
        with self.assertRaisesRegex(NotImplementedError, "found 2"):
            self.call_and_sink("inner", definition)

    def test_reference_does_not_resolve_nested_method_in_another_scope(self) -> None:
        outer, body, _, _ = self.g.method("other", "v")
        self.nested(outer, body, "inner")
        definition = self.g.assign(self.block, "inner", self.g.node("METHOD_REF", "inner", 2))
        with self.assertRaisesRegex(NotImplementedError, "found 0"):
            self.call_and_sink("inner", definition)

    def test_deeper_nesting_uses_same_definition_and_invocation_rules(self) -> None:
        outer, body, exit_node, _ = self.g.method("outer", "v")
        self.g.ast(self.main, outer)
        _, definition = self.nested(outer, body, "inner")
        call = self.g.expression("CALL", "inner", self.g.node("IDENTIFIER", "v"))
        returned = self.g.expression("RETURN", "RETURN", call)
        self.g.ast(body, returned)
        self.g.cfg(outer, definition, call, returned, exit_node)
        define_outer = self.g.assign(self.block, "outer", self.g.node("METHOD_REF", "outer", 2))
        self.assertTrue(self.call_and_sink("outer", define_outer))

    def test_top_level_and_nested_targets_join(self) -> None:
        root, body, root_exit, _ = self.g.method("identity", "v")
        returned = self.g.expression("RETURN", "RETURN", self.g.node("IDENTIFIER", "v"))
        self.g.ast(body, returned)
        self.g.cfg(root, returned, root_exit)
        _, definition = self.nested(self.main, self.block, "inner", "LITERAL", "0")
        condition = self.g.node("IDENTIFIER", "c")
        self.g.ast(self.block, condition)
        left = self.g.assign(self.block, "f", self.g.node("IDENTIFIER", "identity", 2))
        right = self.g.assign(self.block, "f", self.g.node("IDENTIFIER", "inner", 2))
        call = self.g.expression("CALL", "f", self.g.node("IDENTIFIER", "x"), order=2)
        bind = self.g.assign(self.block, "y", call)
        sink, value = self.g.sink(self.block, "y")
        self.g.cfg(self.main, definition, condition, left, call, bind, sink, self.exit)
        self.g.cfg(condition, right, call)
        self.assertTrue(self.g.analyse(self.source, value))

    def test_nested_direct_recursion_is_rejected(self) -> None:
        inner, body, exit_node, _ = self.g.method("inner", "v")
        self.g.ast(self.main, inner)
        recursive = self.g.expression("CALL", "inner", self.g.node("IDENTIFIER", "v"))
        self.g.ast(body, recursive)
        self.g.cfg(inner, recursive, exit_node)
        definition = self.g.assign(self.block, "inner", self.g.node("METHOD_REF", "inner", 2))
        with self.assertRaisesRegex(NotImplementedError, "Recursive/cyclic"):
            self.call_and_sink("inner", definition)

    def test_nested_indirect_recursion_is_rejected(self) -> None:
        inner, body, exit_node, _ = self.g.method("inner", "f", "v")
        self.g.ast(self.main, inner)
        recursive = self.g.expression("CALL", "f", self.g.node("IDENTIFIER", "f", 1),
                                      self.g.node("IDENTIFIER", "v", 2))
        self.g.ast(body, recursive)
        self.g.cfg(inner, recursive, exit_node)
        definition = self.g.assign(self.block, "inner", self.g.node("METHOD_REF", "inner", 2))
        call = self.g.expression("CALL", "inner", self.g.node("IDENTIFIER", "inner", 1),
                                 self.g.node("IDENTIFIER", "x", 2))
        self.g.ast(self.block, call)
        self.g.cfg(self.main, definition, call, self.exit)
        with self.assertRaisesRegex(NotImplementedError, "Recursive/cyclic"):
            self.g.analyse(self.source, self.source)


if __name__ == "__main__":
    unittest.main()
