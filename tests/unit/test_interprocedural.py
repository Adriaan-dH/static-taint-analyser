"""Real-fixture acceptance tests and focused direct-call CFG regressions."""

import unittest
from pathlib import Path

from src.graph import GraphEdge, GraphNode, ProgramGraph, TestMetadata, load_test_case
from src.interprocedural import analyse_interprocedural


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

    def test_deferred_cases_raise(self) -> None:
        names = ["test_inter5", "custom_p2_08_indirect_simple_true",
                 "custom_p2_09_indirect_reassignment_false",
                 "custom_p2_10_indirect_branch_true",
                 "custom_p2_14_nested_function_direct_true",
                 "custom_p2_15_shadowed_function_target"]
        for name in names:
            with self.subTest(case=name):
                with self.assertRaises(NotImplementedError):
                    analyse_interprocedural(*load_test_case(CASES / name))

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
        with self.assertRaisesRegex(NotImplementedError, "Indirect or nested"):
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


if __name__ == "__main__":
    unittest.main()
