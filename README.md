# Static Taint Analyser

A command-line tool for checking whether tainted data can flow from a source
to a sink in Python program graphs, including across function calls.

## Prerequisites

- Linux x86_64, Bash, and `chmod`.
- Python 3.10 or later, available as `python3`. Only the standard library is used.
- Validated on Linux x86_64 with Python 3.12.3 and GNU Bash 5.2.21.

No compiler, dependency installation, or virtual environment is required.

## Build

From the repository root:

```bash
./build.sh
```

The build checks that Python 3.10+ is available and makes the root `main`
script executable.

## Run

```bash
./main -i tests/testcases/test1
./main -i tests/testcases/test_inter1
```

The input is a testcase directory containing `graph.json` and `test_case.json`.
Only these JSON files are needed. The expected `reaches` field is ignored,
and the target program is never executed. The same executable supports
intraprocedural and interprocedural analysis.

Successful analysis exits with code 0 and prints exactly one JSON object:

```json
{"reaches": true}
```

Argument errors exit with code 2. Input or unsupported-analysis errors exit with
code 1. Diagnostics go to stderr, with no result on stdout on failure.

## Test results

Observed through `./main -i <directory>`:

| Testcase | Expected | Observed | Status |
| --- | --- | --- | --- |
| test1 | true | true | Pass |
| test2 | false | false | Pass |
| test3 | true | true | Pass |
| test4 | false | false | Pass |
| test5 | false | false | Pass |
| test_inter1 | true | true | Pass |
| test_inter2 | true | true | Pass |
| test_inter3 | false | false | Pass |
| test_inter4 | false | false | Pass |
| test_inter5 | true | true | Pass |

All 66 test fixtures passed through the CLI, each completing in under 5 seconds
on the validation machine. Run the regression suite with:

```bash
python3 -B -m unittest discover -s tests/unit -v
```

## Assumptions and known limitations

- Branches and while loops are supported. For loops are outside the supported subset.
- Captured/shared scope, classes, and imports are outside the supported subset.
- Recursive/cyclic calls and ambiguous nested `METHOD_REF` resolution currently
  raise `NotImplementedError`.
- Sources must currently be direct method `PARAMETER` nodes.
