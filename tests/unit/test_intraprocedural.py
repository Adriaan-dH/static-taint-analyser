"""Scalar state and CFG-driven intraprocedural analysis tests."""

import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from src.graph import GraphEdge, GraphNode, ProgramGraph, TestMetadata, load_test_case
from src.intraprocedural import analyse_intraprocedural, evaluate_scalar_expression
from src.state import TaintState


class TaintStateTests(unittest.TestCase):
    def test_initial_state_is_clean(self) -> None:
        state = TaintState()
        self.assertEqual(state.tainted_variables, frozenset())
        self.assertFalse(state.is_tainted("x"))

    def test_taint_and_clean_return_new_states(self) -> None:
        initial = TaintState()
        tainted = initial.taint("x").taint("y")
        cleaned = tainted.clean("x")
        self.assertFalse(initial.is_tainted("x"))
        self.assertTrue(tainted.is_tainted("x"))
        self.assertFalse(cleaned.is_tainted("x"))
        self.assertTrue(cleaned.is_tainted("y"))
        self.assertEqual(cleaned.clean("absent"), cleaned)

    def test_join_is_union_and_does_not_mutate_inputs(self) -> None:
        left = TaintState().taint("x")
        right = TaintState().taint("y")
        self.assertEqual(left.join(right), TaintState(frozenset({"x", "y"})))
        self.assertEqual(left.join(right), right.join(left))
        self.assertEqual(left.join(left), left)
        self.assertEqual(left.tainted_variables, frozenset({"x"}))
        self.assertEqual(right.tainted_variables, frozenset({"y"}))

    def test_equality_and_immutability(self) -> None:
        state = TaintState().taint("x")
        self.assertEqual(state, TaintState(frozenset({"x"})))
        self.assertNotEqual(state, TaintState())
        with self.assertRaises(FrozenInstanceError):
            state.tainted_variables = frozenset()


class IntraproceduralTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes = [
            GraphNode(1, "METHOD", "main", 1),
            GraphNode(2, "PARAMETER", "x", 1),
            GraphNode(3, "PARAMETER", "y", 2),
            GraphNode(4, "BLOCK", "[ ... ]", 1),
            GraphNode(5, "EXIT", "EXIT", 2),
            GraphNode(6, "CONTROL_STRUCTURE", "IF", 1),
            GraphNode(7, "CALL", "sink", 2),
            GraphNode(8, "IDENTIFIER", "x", 1),
        ]
        self.edges = [
            GraphEdge(1, child, "AST") for child in (2, 3, 4, 5)
        ] + [GraphEdge(4, 6, "AST"), GraphEdge(4, 7, "AST"), GraphEdge(7, 8, "AST")]

    def set_sink_expression(self, kind: str, value: str) -> None:
        self.nodes = [
            GraphNode(8, kind, value, 1) if node.id == 8 else node
            for node in self.nodes
        ]

    def add_assignment(self, node_id: int, target: str, kind: str, value: str) -> None:
        self.nodes.extend([
            GraphNode(node_id, "OPERATOR", "assignment", node_id),
            GraphNode(node_id + 2, kind, value, 2),
            GraphNode(node_id + 1, "IDENTIFIER", target, 1),
        ])
        # Deliberately supply RHS before target to exercise ast_order.
        self.edges.extend([
            GraphEdge(4, node_id, "AST"),
            GraphEdge(node_id, node_id + 2, "AST"),
            GraphEdge(node_id, node_id + 1, "AST"),
        ])

    def add_operands(
        self, operator_id: int, *operands: tuple[int, str, str, int]
    ) -> None:
        for node_id, kind, value, order in operands:
            self.nodes.append(GraphNode(node_id, kind, value, order))
            self.edges.append(GraphEdge(operator_id, node_id, "AST"))

    def graph_with_cfg(self, connections: list[tuple[int, int]]) -> ProgramGraph:
        return ProgramGraph(
            self.nodes,
            self.edges + [GraphEdge(source, destination, "CFG") for source, destination in connections],
        )

    def analyse(self, connections: list[tuple[int, int]], sink_id: int = 8) -> bool:
        return analyse_intraprocedural(self.graph_with_cfg(connections), TestMetadata(2, sink_id))

    def test_direct_source_to_sink(self) -> None:
        self.assertTrue(self.analyse([(1, 7), (7, 5)]))

    def test_only_designated_parameter_is_tainted(self) -> None:
        self.set_sink_expression("IDENTIFIER", "y")
        self.assertFalse(self.analyse([(1, 7), (7, 5)]))

    def test_metadata_can_designate_a_different_parameter(self) -> None:
        self.set_sink_expression("IDENTIFIER", "y")
        graph = self.graph_with_cfg([(1, 7), (7, 5)])
        self.assertTrue(analyse_intraprocedural(graph, TestMetadata(3, 8)))

    def test_tainted_sibling_argument_does_not_make_selected_sink_tainted(self) -> None:
        self.set_sink_expression("IDENTIFIER", "y")
        self.nodes.append(GraphNode(9, "IDENTIFIER", "x", 2))
        self.edges.append(GraphEdge(7, 9, "AST"))
        self.assertFalse(self.analyse([(1, 7), (7, 5)]))

    def test_scalar_copy_propagates_taint(self) -> None:
        self.add_assignment(10, "y", "IDENTIFIER", "x")
        self.set_sink_expression("IDENTIFIER", "y")
        self.assertTrue(self.analyse([(1, 10), (10, 7), (7, 5)]))

    def test_binary_operator_taints_from_either_operand(self) -> None:
        for first, second in (("x", "y"), ("y", "x")):
            with self.subTest(first=first, second=second):
                self.add_assignment(10, "z", "OPERATOR", "addition")
                self.add_operands(12, (13, "IDENTIFIER", first, 1), (14, "IDENTIFIER", second, 2))
                self.set_sink_expression("IDENTIFIER", "z")
                self.assertTrue(self.analyse([(1, 10), (10, 7), (7, 5)]))
                self.setUp()

    def test_binary_operator_with_clean_operands_is_clean(self) -> None:
        self.add_assignment(10, "z", "OPERATOR", "addition")
        self.add_operands(12, (13, "IDENTIFIER", "y", 1), (14, "LITERAL", "1", 2))
        self.set_sink_expression("IDENTIFIER", "z")
        self.assertFalse(self.analyse([(1, 10), (10, 7), (7, 5)]))

    def test_unary_operator_propagates_taint(self) -> None:
        self.add_assignment(10, "z", "OPERATOR", "unaryMinus")
        self.add_operands(12, (13, "IDENTIFIER", "x", 1))
        self.set_sink_expression("IDENTIFIER", "z")
        self.assertTrue(self.analyse([(1, 10), (10, 7), (7, 5)]))

    def test_nested_operators_evaluate_recursively(self) -> None:
        for inner_variable, expected in (("x", True), ("y", False)):
            with self.subTest(inner_variable=inner_variable):
                self.add_assignment(10, "z", "OPERATOR", "multiplication")
                self.add_operands(12, (13, "OPERATOR", "addition", 1), (14, "LITERAL", "2", 2))
                self.add_operands(13, (15, "IDENTIFIER", inner_variable, 1), (16, "LITERAL", "1", 2))
                self.set_sink_expression("IDENTIFIER", "z")
                self.assertEqual(self.analyse([(1, 10), (10, 7), (7, 5)]), expected)
                self.setUp()

    def test_deeply_nested_operators(self) -> None:
        self.add_assignment(10, "z", "OPERATOR", "multiplication")
        self.add_operands(12, (13, "OPERATOR", "subtraction", 1), (14, "LITERAL", "2", 2))
        self.add_operands(13, (15, "OPERATOR", "unaryMinus", 1), (16, "LITERAL", "1", 2))
        self.add_operands(15, (17, "OPERATOR", "addition", 1))
        self.add_operands(17, (18, "IDENTIFIER", "x", 1), (19, "LITERAL", "1", 2))
        self.set_sink_expression("IDENTIFIER", "z")
        self.assertTrue(self.analyse([(1, 10), (10, 7), (7, 5)]))

    def test_operator_ast_order_is_used_despite_edge_and_node_order(self) -> None:
        self.add_assignment(10, "z", "OPERATOR", "subtraction")
        self.add_operands(12, (14, "LITERAL", "1", 2), (13, "IDENTIFIER", "x", 1))
        self.nodes.reverse()
        graph = self.graph_with_cfg([(1, 10), (10, 7), (7, 5)])
        self.assertEqual(tuple(child.id for child in graph.ast_children(12)), (13, 14))
        self.assertTrue(analyse_intraprocedural(graph, TestMetadata(2, 12)))

    def test_comparison_results_are_clean(self) -> None:
        for operator in (
            "greaterThan", "lessThan", "greaterThanOrEqual",
            "lessThanOrEqual", "equal", "notEqual",
        ):
            with self.subTest(operator=operator):
                self.add_assignment(10, "z", "OPERATOR", operator)
                self.add_operands(12, (13, "IDENTIFIER", "x", 1), (14, "LITERAL", "0", 2))
                self.set_sink_expression("IDENTIFIER", "z")
                self.assertFalse(self.analyse([(1, 10), (10, 7), (7, 5)]))
                self.setUp()

    def test_comparison_overwrites_tainted_target_with_clean_result(self) -> None:
        self.add_assignment(10, "y", "IDENTIFIER", "x")
        self.add_assignment(20, "y", "OPERATOR", "greaterThan")
        self.add_operands(22, (23, "IDENTIFIER", "x", 1), (24, "LITERAL", "0", 2))
        self.set_sink_expression("IDENTIFIER", "y")
        self.assertFalse(self.analyse([(1, 10), (10, 20), (20, 7), (7, 5)]))

    def test_operator_expression_can_be_designated_sink(self) -> None:
        for operator, expected in (("addition", True), ("addition", False), ("greaterThan", False)):
            with self.subTest(operator=operator, expected=expected):
                self.set_sink_expression("OPERATOR", operator)
                variable = "x" if expected or operator == "greaterThan" else "y"
                self.add_operands(8, (9, "IDENTIFIER", variable, 1), (10, "LITERAL", "1", 2))
                self.assertEqual(self.analyse([(1, 7), (7, 5)]), expected)
                self.setUp()

    def test_nested_operator_rhs_sink_uses_pre_assignment_state(self) -> None:
        self.add_assignment(10, "x", "OPERATOR", "addition")
        self.add_operands(12, (13, "IDENTIFIER", "x", 1), (14, "LITERAL", "1", 2))
        self.assertTrue(self.analyse([(1, 10), (10, 5)], sink_id=12))

    def test_comparison_predicate_does_not_prune_cfg_branches(self) -> None:
        self.nodes.append(GraphNode(30, "OPERATOR", "greaterThan", 1))
        self.edges.append(GraphEdge(6, 30, "AST"))
        self.add_operands(30, (31, "IDENTIFIER", "x", 1), (32, "LITERAL", "0", 2))
        self.add_assignment(10, "y", "IDENTIFIER", "x")
        self.add_assignment(20, "y", "LITERAL", "0")
        self.set_sink_expression("IDENTIFIER", "y")
        self.assertTrue(self.analyse([(1, 30), (30, 10), (30, 20), (10, 7), (20, 7), (7, 5)]))

    def test_literal_overwrite_removes_taint(self) -> None:
        self.add_assignment(10, "x", "LITERAL", '"clean"')
        self.assertFalse(self.analyse([(1, 10), (10, 7), (7, 5)]))

    def test_clean_variable_overwrite_removes_taint(self) -> None:
        self.add_assignment(10, "x", "IDENTIFIER", "y")
        self.assertFalse(self.analyse([(1, 10), (10, 7), (7, 5)]))

    def test_clean_assignment_does_not_introduce_taint(self) -> None:
        self.add_assignment(10, "y", "LITERAL", "42")
        self.set_sink_expression("IDENTIFIER", "y")
        self.assertFalse(self.analyse([(1, 10), (10, 7), (7, 5)]))

    def test_scalar_copy_is_not_an_alias(self) -> None:
        self.add_assignment(10, "y", "IDENTIFIER", "x")
        self.add_assignment(20, "x", "LITERAL", "0")
        self.set_sink_expression("IDENTIFIER", "y")
        self.assertTrue(self.analyse([(1, 10), (10, 20), (20, 7), (7, 5)]))

    def test_branch_join_keeps_taint_from_either_path(self) -> None:
        self.add_assignment(10, "y", "LITERAL", "0")
        self.add_assignment(20, "y", "IDENTIFIER", "x")
        self.set_sink_expression("IDENTIFIER", "y")
        for branches in ([(6, 10), (6, 20)], [(6, 20), (6, 10)]):
            with self.subTest(branches=branches):
                self.assertTrue(self.analyse([(1, 6)] + branches + [(10, 7), (20, 7), (7, 5)]))

    def test_loop_back_edge_requires_revisiting_assignment_and_sink(self) -> None:
        self.add_assignment(10, "y", "IDENTIFIER", "z")
        self.add_assignment(20, "z", "IDENTIFIER", "x")
        self.set_sink_expression("IDENTIFIER", "y")
        # First visit to 10 and 7 is clean. Only a second iteration copies
        # tainted z to y; visiting each CFG node once would return False.
        self.assertTrue(self.analyse([(1, 10), (10, 7), (10, 20), (20, 10), (7, 5)]))

    def test_loop_with_clean_overwrite_converges_without_reseeding_source(self) -> None:
        self.add_assignment(10, "x", "LITERAL", "0")
        self.assertFalse(self.analyse([(1, 10), (10, 6), (6, 10), (6, 7), (7, 5)]))

    def test_sink_is_specific_occurrence_before_or_after_overwrite(self) -> None:
        self.add_assignment(10, "x", "LITERAL", "0")
        self.nodes.extend([GraphNode(20, "CALL", "sink", 3), GraphNode(21, "IDENTIFIER", "x", 1)])
        self.edges.extend([GraphEdge(4, 20, "AST"), GraphEdge(20, 21, "AST")])
        connections = [(1, 7), (7, 10), (10, 20), (20, 5)]
        self.assertTrue(self.analyse(connections, sink_id=8))
        self.assertFalse(self.analyse(connections, sink_id=21))

    def test_literal_sink_is_clean(self) -> None:
        self.set_sink_expression("LITERAL", "0")
        self.assertFalse(self.analyse([(1, 7), (7, 5)]))

    def test_parameter_sink_uses_entry_state(self) -> None:
        self.assertTrue(self.analyse([(1, 5)], sink_id=2))
        self.assertFalse(self.analyse([(1, 5)], sink_id=3))

    def test_assignment_rhs_uses_in_state_and_target_uses_out_state(self) -> None:
        self.add_assignment(10, "y", "IDENTIFIER", "x")
        connections = [(1, 10), (10, 5)]
        self.assertTrue(self.analyse(connections, sink_id=12))
        self.assertTrue(self.analyse(connections, sink_id=11))

    def test_unreachable_sink_is_clean_even_when_source_is_tainted(self) -> None:
        self.assertFalse(self.analyse([(1, 5), (7, 5)]))
        self.assertFalse(self.analyse([(1, 5)]))

    def test_unreachable_predecessor_does_not_introduce_taint(self) -> None:
        self.add_assignment(10, "y", "IDENTIFIER", "x")
        self.set_sink_expression("IDENTIFIER", "y")
        self.assertFalse(self.analyse([(1, 7), (10, 7), (7, 5)]))

    def test_cfg_order_overrides_raw_node_order(self) -> None:
        self.add_assignment(10, "x", "LITERAL", "0")
        self.nodes.reverse()
        self.assertFalse(self.analyse([(1, 10), (10, 7), (7, 5)]))

    def test_predicate_read_does_not_taint_other_variables(self) -> None:
        self.nodes.extend([GraphNode(10, "OPERATOR", "greaterThan", 1), GraphNode(11, "IDENTIFIER", "x", 1)])
        self.edges.extend([GraphEdge(6, 10, "AST"), GraphEdge(10, 11, "AST")])
        self.set_sink_expression("IDENTIFIER", "y")
        self.assertFalse(self.analyse([(1, 10), (10, 7), (10, 5), (7, 5)]))

    def test_different_methods_are_explicitly_unsupported(self) -> None:
        self.nodes.append(GraphNode(100, "METHOD", "other", 2))
        self.edges.remove(GraphEdge(4, 7, "AST"))
        self.edges.append(GraphEdge(100, 7, "AST"))
        with self.assertRaisesRegex(ValueError, "same METHOD"):
            self.analyse([(1, 5)])

    def test_nodes_outside_a_method_are_rejected(self) -> None:
        self.edges.remove(GraphEdge(7, 8, "AST"))
        with self.assertRaisesRegex(ValueError, "belong to a METHOD"):
            self.analyse([(1, 7), (7, 5)])

    def test_non_parameter_source_is_explicitly_unsupported(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "PARAMETER sources"):
            analyse_intraprocedural(self.graph_with_cfg([(1, 7)]), TestMetadata(8, 8))

    def test_unsupported_assignment_rhs_is_not_assumed_clean(self) -> None:
        for kind, value in (("CALL", "foo"), ("OPERATOR", "unknown"), ("LIST", "[]")):
            with self.subTest(kind=kind):
                nodes = self.nodes + [
                    GraphNode(10, "OPERATOR", "assignment", 1),
                    GraphNode(11, "IDENTIFIER", "y", 1),
                    GraphNode(12, kind, value, 2),
                ]
                edges = self.edges + [
                    GraphEdge(4, 10, "AST"), GraphEdge(10, 11, "AST"), GraphEdge(10, 12, "AST"),
                    GraphEdge(1, 10, "CFG"), GraphEdge(10, 7, "CFG"), GraphEdge(7, 5, "CFG"),
                ]
                with self.assertRaisesRegex(NotImplementedError, "Unsupported scalar expression"):
                    analyse_intraprocedural(ProgramGraph(nodes, edges), TestMetadata(2, 8))

    def test_unsupported_nested_operand_is_not_hidden_by_taint_or_comparison(self) -> None:
        for operator in ("addition", "greaterThan"):
            with self.subTest(operator=operator):
                self.add_assignment(10, "z", "OPERATOR", operator)
                self.add_operands(12, (13, "IDENTIFIER", "x", 1), (14, "CALL", "foo", 2))
                with self.assertRaisesRegex(NotImplementedError, "Unsupported scalar expression"):
                    self.analyse([(1, 10), (10, 7), (7, 5)])
                self.setUp()

    def test_malformed_scalar_operator_is_rejected(self) -> None:
        self.add_assignment(10, "z", "OPERATOR", "addition")
        self.add_operands(12, (13, "IDENTIFIER", "x", 1))
        with self.assertRaisesRegex(ValueError, "expected 2 operands"):
            self.analyse([(1, 10), (10, 7), (7, 5)])

    def test_assignment_itself_is_not_a_scalar_sink(self) -> None:
        self.add_assignment(10, "z", "IDENTIFIER", "x")
        with self.assertRaisesRegex(NotImplementedError, "Unsupported scalar sink"):
            self.analyse([(1, 10), (10, 5)], sink_id=10)

    def test_scalar_expression_evaluation(self) -> None:
        state = TaintState().taint("x")
        nodes = [
            GraphNode(1, "IDENTIFIER", "x", 1),
            GraphNode(2, "IDENTIFIER", "y", 1),
            GraphNode(3, "LITERAL", "x", 1),
        ]
        graph = ProgramGraph(nodes, [])
        self.assertTrue(evaluate_scalar_expression(graph, nodes[0], state))
        self.assertFalse(evaluate_scalar_expression(graph, nodes[1], state))
        self.assertFalse(evaluate_scalar_expression(graph, nodes[2], state))

    def test_five_supplied_intraprocedural_cases(self) -> None:
        cases_directory = Path(__file__).resolve().parents[1] / "testcases"
        for case, expected in (("test1", True), ("test2", False), ("test3", True), ("test4", False), ("test5", False)):
            with self.subTest(case=case):
                graph, metadata = load_test_case(cases_directory / case)
                self.assertEqual(analyse_intraprocedural(graph, metadata), expected)


if __name__ == "__main__":
    unittest.main()
