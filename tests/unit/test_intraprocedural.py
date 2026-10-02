"""Value, alias, list memory, and CFG-driven intraprocedural analysis tests."""

import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from src.graph import GraphEdge, GraphNode, ProgramGraph, TestMetadata, load_test_case
from src.intraprocedural import analyse_intraprocedural, evaluate_scalar_expression
from src.state import AbstractValue, CLEAN_SCALAR, TAINTED_SCALAR, TaintState


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
        self.assertEqual(left.join(right), TaintState().taint("x").taint("y"))
        self.assertEqual(left.join(right), right.join(left))
        self.assertEqual(left.join(left), left)
        self.assertEqual(left.tainted_variables, frozenset({"x"}))
        self.assertEqual(right.tainted_variables, frozenset({"y"}))

    def test_equality_and_immutability(self) -> None:
        state = TaintState().taint("x")
        self.assertEqual(state, TaintState().taint("x"))
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
                with self.assertRaisesRegex(NotImplementedError, "Unsupported expression"):
                    analyse_intraprocedural(ProgramGraph(nodes, edges), TestMetadata(2, 8))

    def test_unsupported_nested_operand_is_not_hidden_by_taint_or_comparison(self) -> None:
        for operator in ("addition", "greaterThan"):
            with self.subTest(operator=operator):
                self.add_assignment(10, "z", "OPERATOR", operator)
                self.add_operands(12, (13, "IDENTIFIER", "x", 1), (14, "CALL", "foo", 2))
                with self.assertRaisesRegex(NotImplementedError, "Unsupported expression"):
                    self.analyse([(1, 10), (10, 7), (7, 5)])
                self.setUp()

    def test_malformed_scalar_operator_is_rejected(self) -> None:
        self.add_assignment(10, "z", "OPERATOR", "addition")
        self.add_operands(12, (13, "IDENTIFIER", "x", 1))
        with self.assertRaisesRegex(ValueError, "expected 2 operands"):
            self.analyse([(1, 10), (10, 7), (7, 5)])

    def test_assignment_itself_is_not_a_scalar_sink(self) -> None:
        self.add_assignment(10, "z", "IDENTIFIER", "x")
        with self.assertRaisesRegex(NotImplementedError, "Unsupported sink"):
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


class ListIntraproceduralTests(unittest.TestCase):
    """Small ASTs use the builder's operators and expression-level CFG order.

    In the fixture notation, strings are identifiers, integers are literals,
    lists are list literals, and tuples name an operator and its operands.
    """

    def setUp(self) -> None:
        self.nodes = [
            GraphNode(1, "METHOD", "main", 1),
            GraphNode(2, "PARAMETER", "x", 1),
            GraphNode(3, "BLOCK", "[ ... ]", 2),
            GraphNode(4, "EXIT", "EXIT", 3),
        ]
        self.edges = [GraphEdge(1, child, "AST") for child in (2, 3, 4)]
        self.next_id = 5
        self.tail = 1
        self.entries: dict[int, int] = {}

    def node(self, kind: str, value: str, order: int = 1) -> int:
        node_id = self.next_id
        self.next_id += 1
        self.nodes.append(GraphNode(node_id, kind, value, order))
        return node_id

    def expression(self, parent: int, order: int, value: object) -> tuple[int, list[int]]:
        if isinstance(value, str):
            node_id = self.node("IDENTIFIER", value, order)
            children = ()
        elif isinstance(value, int):
            node_id = self.node("LITERAL", str(value), order)
            children = ()
        elif isinstance(value, list):
            node_id = self.node("OPERATOR", "listLiteral", order)
            children = value
        elif isinstance(value, tuple):
            node_id = self.node("OPERATOR", value[0], order)
            children = value[1:]
        else:
            raise TypeError(f"Unknown fixture expression {value!r}")
        self.edges.append(GraphEdge(parent, node_id, "AST"))
        points = []
        for child_order, child in enumerate(children, 1):
            _, child_points = self.expression(node_id, child_order, child)
            points.extend(child_points)
        if isinstance(value, (list, tuple)):
            points.append(node_id)
        return node_id, points

    def connect(self, source: int, destination: int) -> None:
        self.edges.append(GraphEdge(source, destination, "CFG"))

    def sequence(self, statement: int, points: list[int], predecessors: list[int] | None) -> None:
        points = points + [statement]
        self.entries[statement] = points[0]
        for predecessor in [self.tail] if predecessors is None else predecessors:
            self.connect(predecessor, points[0])
        for source, destination in zip(points, points[1:]):
            self.connect(source, destination)
        self.tail = statement

    def assign(self, target: object, value: object, predecessors: list[int] | None = None) -> int:
        statement = self.node("OPERATOR", "assignment")
        self.edges.append(GraphEdge(3, statement, "AST"))
        _, target_points = self.expression(statement, 1, target)
        _, value_points = self.expression(statement, 2, value)
        self.sequence(statement, target_points + value_points, predecessors)
        return statement

    def sink(self, value: object, predecessors: list[int] | None = None) -> int:
        statement = self.node("CALL", "sink")
        self.edges.append(GraphEdge(3, statement, "AST"))
        sink_id, points = self.expression(statement, 1, value)
        self.sequence(statement, points, predecessors)
        self.connect(statement, 4)
        return sink_id

    def graph(self) -> ProgramGraph:
        return ProgramGraph(self.nodes, self.edges)

    def analyse(self, value: object, predecessors: list[int] | None = None) -> bool:
        sink_id = self.sink(value, predecessors)
        return analyse_intraprocedural(self.graph(), TestMetadata(2, sink_id))

    def test_list_creation_and_exact_read(self) -> None:
        for element, expected in (("x", True), (0, False)):
            with self.subTest(element=element):
                self.setUp()
                self.assign("a", [element])
                self.assign("y", ("indexAccess", "a", 0))
                self.assertEqual(self.analyse("y"), expected)

    def test_exact_reads_distinguish_elements(self) -> None:
        for index, expected in ((0, False), (1, True)):
            with self.subTest(index=index):
                self.setUp()
                self.assign("a", [0, "x"])
                self.assertEqual(self.analyse(("indexAccess", "a", index)), expected)

    def test_literal_element_order_uses_ast_order(self) -> None:
        self.assign("a", [0, "x"])
        self.nodes.reverse()
        self.edges.reverse()
        self.assertFalse(self.analyse(("indexAccess", "a", 0)))
        self.assertTrue(self.analyse(("indexAccess", "a", 1)))

    def test_unknown_read_joins_all_elements(self) -> None:
        self.assign("a", [0, "x"])
        self.assertTrue(self.analyse(("indexAccess", "a", "i")))

    def test_tainted_index_does_not_taint_read(self) -> None:
        self.assign("a", [0])
        self.assertFalse(self.analyse(("indexAccess", "a", "x")))

    def test_alias_write_is_visible_through_original(self) -> None:
        self.assign("a", [0])
        self.assign("b", "a")
        self.assign(("indexAccess", "b", 0), "x")
        self.assertTrue(self.analyse(("indexAccess", "a", 0)))

    def test_alias_can_strongly_clean_shared_object(self) -> None:
        self.assign("a", ["x"])
        self.assign("b", "a")
        self.assign(("indexAccess", "b", 0), 0)
        self.assertFalse(self.analyse(("indexAccess", "a", 0)))

    def test_rebinding_leaves_old_alias_clean(self) -> None:
        self.assign("a", [0])
        self.assign("b", "a")
        self.assign("a", ["x"])
        self.assertFalse(self.analyse(("indexAccess", "b", 0)))

    def test_rebinding_keeps_original_alias_mutable(self) -> None:
        self.assign("a", [0])
        self.assign("b", "a")
        self.assign("a", [1])
        self.assign(("indexAccess", "b", 0), "x")
        self.assertTrue(self.analyse(("indexAccess", "b", 0)))

    def test_mutating_old_alias_does_not_mutate_rebound_object(self) -> None:
        self.assign("a", [0])
        self.assign("b", "a")
        self.assign("a", [1])
        self.assign(("indexAccess", "b", 0), "x")
        self.assertFalse(self.analyse(("indexAccess", "a", 0)))

    def test_scalar_rebinding_drops_reference_only(self) -> None:
        self.assign("a", ["x"])
        self.assign("b", "a")
        self.assign("a", 0)
        self.assertFalse(self.analyse("a"))
        self.assertTrue(self.analyse("b"))

    def test_list_rebinding_drops_scalar_taint(self) -> None:
        self.assign("a", "x")
        self.assign("a", [0])
        self.assertFalse(self.analyse("a"))

    def test_strong_overwrite_kills_taint(self) -> None:
        self.assign("a", ["x"])
        self.assign(("indexAccess", "a", 0), 0)
        self.assertFalse(self.analyse(("indexAccess", "a", 0)))

    def test_strong_overwrite_introduces_taint(self) -> None:
        self.assign("a", [0])
        self.assign(("indexAccess", "a", 0), "x")
        self.assertTrue(self.analyse(("indexAccess", "a", 0)))

    def test_unknown_clean_write_preserves_old_taint(self) -> None:
        self.assign("a", ["x", 0])
        self.assign(("indexAccess", "a", "i"), 0)
        self.assertTrue(self.analyse(("indexAccess", "a", 0)))

    def test_unknown_tainted_write_affects_any_exact_read(self) -> None:
        for index in (0, 1):
            with self.subTest(index=index):
                self.setUp()
                self.assign("a", [0, 0])
                self.assign(("indexAccess", "a", "i"), "x")
                self.assertTrue(self.analyse(("indexAccess", "a", index)))

    def test_tainted_write_index_does_not_taint_contents(self) -> None:
        self.assign("a", [0])
        self.assign(("indexAccess", "a", "x"), 0)
        self.assertFalse(self.analyse("a"))

    def test_strong_clean_after_unknown_tainted_write(self) -> None:
        self.assign("a", [0, 0])
        self.assign(("indexAccess", "a", "i"), "x")
        self.assign(("indexAccess", "a", 0), 0)
        self.assertFalse(self.analyse(("indexAccess", "a", 0)))
        self.assertTrue(self.analyse(("indexAccess", "a", 1)))

    def test_multiple_targets_receive_weak_clean_write(self) -> None:
        for observed in ("first", "second"):
            with self.subTest(observed=observed):
                self.setUp()
                self.assign("first", ["x"])
                branch = self.assign("second", ["x"])
                left = self.assign("a", "first", [branch])
                right = self.assign("a", "second", [branch])
                self.assign(("indexAccess", "a", 0), 0, [left, right])
                self.assertTrue(self.analyse(("indexAccess", observed, 0)))

    def test_multiple_targets_receive_weak_tainted_write(self) -> None:
        self.assign("first", [0])
        branch = self.assign("second", [0])
        left = self.assign("a", "first", [branch])
        right = self.assign("a", "second", [branch])
        self.assign(("indexAccess", "a", 0), "x", [left, right])
        self.assertTrue(self.analyse(("indexAccess", "second", 0)))

    def test_branch_join_preserves_scalar_and_list_possibilities(self) -> None:
        branch = self.assign("first", [0])
        left = self.assign("a", "first", [branch])
        right = self.assign("a", "x", [branch])
        self.assertTrue(self.analyse("a", [left, right]))

    def test_mixed_scalar_and_list_target_prevents_strong_write(self) -> None:
        branch = self.assign("first", ["x"])
        left = self.assign("a", "first", [branch])
        right = self.assign("a", 0, [branch])
        self.assign(("indexAccess", "a", 0), 0, [left, right])
        self.assertTrue(self.analyse(("indexAccess", "first", 0)))

    def test_clean_parameter_remains_a_scalar_possibility_at_join(self) -> None:
        parameter = self.node("PARAMETER", "clean", 2)
        self.edges.append(GraphEdge(1, parameter, "AST"))
        branch = self.assign("first", ["x"])
        left = self.assign("a", "first", [branch])
        right = self.assign("a", "clean", [branch])
        self.assign(("indexAccess", "a", 0), 0, [left, right])
        self.assertTrue(self.analyse(("indexAccess", "first", 0)))

    def test_list_memory_joins_across_branches(self) -> None:
        branch = self.assign("a", [0])
        left = self.assign(("indexAccess", "a", 0), "x", [branch])
        right = self.assign(("indexAccess", "a", 0), 0, [branch])
        self.assertTrue(self.analyse(("indexAccess", "a", 0), [left, right]))

    def test_back_edge_requires_another_iteration_to_read_taint(self) -> None:
        self.assign("a", [0])
        read = self.assign("y", ("indexAccess", "a", 0))
        sink_id = self.sink("y", [read])
        write = self.assign(("indexAccess", "a", 0), "x", [read])
        self.connect(write, self.entries[read])
        self.assertTrue(analyse_intraprocedural(self.graph(), TestMetadata(2, sink_id)))

    def test_loop_allocation_preserves_older_aliased_instances(self) -> None:
        allocation = self.assign("a", [0])
        save = self.assign("b", "a", [allocation])
        write = self.assign(("indexAccess", "a", 0), "x", [save, allocation])
        self.connect(write, self.entries[allocation])
        self.assign(("indexAccess", "a", 0), 0, [write])
        # b may retain a previous iteration's instance, while a is cleaned.
        self.assertTrue(self.analyse(("indexAccess", "b", 0)))

    def test_allocation_outside_loop_still_allows_strong_updates(self) -> None:
        self.assign("a", ["x"])
        write = self.assign(("indexAccess", "a", 0), 0)
        self.connect(write, self.entries[write])
        self.assertFalse(self.analyse(("indexAccess", "a", 0)))

    def test_whole_list_sink_observes_contents(self) -> None:
        self.assign("a", [0, "x"])
        self.assertTrue(self.analyse("a"))

    def test_clean_and_empty_whole_lists(self) -> None:
        for elements in ([], [0]):
            with self.subTest(elements=elements):
                self.setUp()
                self.assign("a", elements)
                self.assertFalse(self.analyse("a"))

    def test_list_literal_can_be_sink(self) -> None:
        self.assertTrue(self.analyse(["x"]))

    def test_literal_sink_in_loop_observes_new_initialisation(self) -> None:
        self.assign("y", 0)
        sink_id = self.sink(["y"])
        sink_statement = self.tail
        taint = self.assign("y", "x", [sink_statement])
        self.connect(taint, self.entries[sink_statement])
        self.assertTrue(analyse_intraprocedural(self.graph(), TestMetadata(2, sink_id)))

    def test_empty_list_can_receive_an_abstract_write(self) -> None:
        self.assign("a", [])
        self.assign(("indexAccess", "a", 0), "x")
        self.assertTrue(self.analyse(("indexAccess", "a", 0)))

    def test_negative_index_selects_last_element(self) -> None:
        self.assign("a", [0, "x"])
        self.assertTrue(self.analyse(("indexAccess", "a", ("minus", 1))))
        self.assertFalse(self.analyse(("indexAccess", "a", ("minus", 2))))

    def test_negative_index_write_normalises_to_positive_cell(self) -> None:
        self.assign("a", [0, "x"])
        self.assign(("indexAccess", "a", ("minus", 1)), 0)
        self.assertFalse(self.analyse(("indexAccess", "a", 1)))

    def test_computed_index_is_unknown(self) -> None:
        self.assign("a", [0, "x"])
        self.assertTrue(self.analyse(("indexAccess", "a", ("addition", 0, 0))))

    def test_reference_valued_element_preserves_alias(self) -> None:
        self.assign("inner", [0])
        self.assign("outer", ["inner"])
        self.assign("alias", ("indexAccess", "outer", 0))
        self.assign(("indexAccess", "alias", 0), "x")
        self.assertTrue(self.analyse(("indexAccess", "inner", 0)))

    def test_nested_literal_and_read(self) -> None:
        self.assign("a", [["x"], [0]])
        self.assertTrue(self.analyse(("indexAccess", ("indexAccess", "a", 0), 0)))
        self.assertFalse(self.analyse(("indexAccess", ("indexAccess", "a", 1), 0)))

    def test_nested_target_write(self) -> None:
        self.assign("a", [[0]])
        self.assign(("indexAccess", ("indexAccess", "a", 0), 0), "x")
        self.assertTrue(self.analyse("a"))

    def test_reference_can_be_stored_in_existing_cell(self) -> None:
        self.assign("a", [0])
        self.assign("b", ["x"])
        self.assign(("indexAccess", "a", 0), "b")
        self.assertTrue(self.analyse("a"))

    def test_unknown_write_can_store_list_references(self) -> None:
        self.assign("a", [0, 0])
        self.assign("b", ["x"])
        self.assign(("indexAccess", "a", "i"), "b")
        self.assertTrue(self.analyse(("indexAccess", "a", 0)))

    def test_overwriting_reference_cell_does_not_clean_referenced_object(self) -> None:
        self.assign("b", ["x"])
        self.assign("a", ["b"])
        self.assign(("indexAccess", "a", 0), 0)
        self.assertFalse(self.analyse("a"))
        self.assertTrue(self.analyse("b"))

    def test_scalar_read_is_a_copy_after_list_mutation(self) -> None:
        self.assign("a", ["x"])
        self.assign("y", ("indexAccess", "a", 0))
        self.assign(("indexAccess", "a", 0), 0)
        self.assertTrue(self.analyse("y"))

    def test_list_comparison_is_clean(self) -> None:
        self.assign("a", ["x"])
        self.assertFalse(self.analyse(("equal", "a", "a")))

    def test_unsupported_list_element_is_not_hidden_by_taint(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "Unsupported expression"):
            self.analyse(["x", ("unknown", 0)])

    def test_cyclic_list_references_terminate(self) -> None:
        self.assign("a", [0, 0])
        self.assign(("indexAccess", "a", 0), "a")
        self.assertFalse(self.analyse("a"))
        self.assign(("indexAccess", "a", 1), "x")
        self.assertTrue(self.analyse("a"))

    def test_specific_index_sink_occurrence_before_and_after_write(self) -> None:
        self.assign("a", ["x"])
        before = self.sink(("indexAccess", "a", 0))
        self.assign(("indexAccess", "a", 0), 0)
        after = self.sink(("indexAccess", "a", 0))
        graph = self.graph()
        self.assertTrue(analyse_intraprocedural(graph, TestMetadata(2, before)))
        self.assertFalse(analyse_intraprocedural(graph, TestMetadata(2, after)))

    def test_index_target_sink_observes_written_value(self) -> None:
        self.assign("a", [0])
        assignment = self.assign(("indexAccess", "a", 0), "x")
        graph = self.graph()
        target = graph.ast_children(assignment)[0]
        self.assertTrue(analyse_intraprocedural(graph, TestMetadata(2, target.id)))

    def test_unsequenced_literal_in_synthetic_assignment(self) -> None:
        assignment = self.assign("a", ["x"])
        literal = self.entries[assignment]
        self.edges = [edge for edge in self.edges if edge.kind != "CFG"]
        self.connect(1, assignment)
        self.assertNotEqual(literal, assignment)
        self.assertTrue(self.analyse(("indexAccess", "a", 0)))

    def test_malformed_index_is_rejected(self) -> None:
        self.assign("a", [0])
        with self.assertRaisesRegex(ValueError, "base and index"):
            self.analyse(("indexAccess", "a"))

    def test_unmodelled_list_parameter_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "modelled list reference"):
            self.analyse(("indexAccess", "x", 0))

    def test_slices_and_list_arithmetic_fail_explicitly(self) -> None:
        self.assign("a", ["x"])
        with self.assertRaisesRegex(NotImplementedError, "Unsupported sink"):
            self.analyse(("slice", "a", 0, 1, 1))
        with self.assertRaisesRegex(NotImplementedError, "List arithmetic"):
            self.analyse(("addition", "a", "a"))


class ListStateTests(unittest.TestCase):
    def test_clean_binding_preserves_scalar_possibility(self) -> None:
        initial = TaintState().allocate(10, (TAINTED_SCALAR,))
        initial = initial.bind("a", AbstractValue(list_objects=frozenset({10})))
        cleaned = initial.clean("a")
        joined = initial.join(cleaned)
        self.assertTrue(joined.value("a").may_be_scalar)
        self.assertEqual(joined.value("a").list_objects, frozenset({10}))
        self.assertFalse(cleaned.is_tainted("a"))

    def test_join_unions_reference_sets(self) -> None:
        left = TaintState().allocate(10, (CLEAN_SCALAR,))
        left = left.bind("a", AbstractValue(list_objects=frozenset({10})))
        right = TaintState().allocate(20, (TAINTED_SCALAR,))
        right = right.bind("a", AbstractValue(list_objects=frozenset({20})))
        joined = left.join(right)
        self.assertEqual(joined.value("a").list_objects, frozenset({10, 20}))
        self.assertTrue(joined.is_tainted("a"))
        self.assertFalse(left.is_tainted("a"))
        self.assertEqual(joined, right.join(left))
        self.assertEqual(joined.join(joined), joined)

    def test_memory_write_does_not_mutate_stored_state(self) -> None:
        initial = TaintState().allocate(10, (TAINTED_SCALAR,))
        initial = initial.bind("a", AbstractValue(list_objects=frozenset({10})))
        memory = initial.list_memory(10).write(0, CLEAN_SCALAR, strong=True)
        cleaned = initial.with_list(10, memory)
        self.assertTrue(initial.is_tainted("a"))
        self.assertFalse(cleaned.is_tainted("a"))
        with self.assertRaises(FrozenInstanceError):
            memory.unknown = TAINTED_SCALAR

    def test_join_missing_cell_uses_unknown_write_summary(self) -> None:
        initial = TaintState().allocate(10, ())
        left = initial.with_list(10, initial.list_memory(10).write(None, TAINTED_SCALAR, False))
        right = initial.with_list(10, initial.list_memory(10).write(0, CLEAN_SCALAR, True))
        joined = left.join(right)
        self.assertTrue(joined.list_memory(10).read(0).scalar_tainted)

    def test_repeated_allocation_preserves_old_contents_and_identity(self) -> None:
        initial = TaintState().allocate(10, (TAINTED_SCALAR,))
        repeated = initial.allocate(10, (CLEAN_SCALAR,))
        self.assertEqual(tuple(object_id for object_id, _ in repeated.lists), (10,))
        self.assertTrue(repeated.list_memory(10).multiple_allocations)
        self.assertTrue(repeated.list_memory(10).read(0).scalar_tainted)


if __name__ == "__main__":
    unittest.main()
