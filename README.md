# Static Taint Analyser

A static data-flow analyser for tracking taint propagation through Python
control-flow graphs, aliases, lists, and function calls.

## Project Structure

- `main` - command-line entry point.
- `build.sh` - build/setup script.
- `src/` - analyser implementation.
- `tests/unit/` - unit tests.
- `tests/testcases/` - analyser test cases.

## Input Parsing

Requires Python 3.10 or later and uses only the standard library.

```python
from src.graph import load_test_case

graph, metadata = load_test_case("tests/testcases/test1")
```

The loader reads `graph.json` and `test_case.json`, preserving node IDs,
`astOrder` (as `ast_order`), and edge kinds. Metadata contains only
`source_node` and `sink_node`; `reaches` is ignored. No taint analysis is
implemented yet.

Run the parsing tests from the repository root:

```bash
python3 -B -m unittest discover -s tests/unit -v
```
