# Gradient Boosting Tree

[![codecov](https://codecov.io/gh/tpoppo/gradient-boosting-tree/branch/main/graph/badge.svg?token=HKJRZ7JHN5)](https://codecov.io/gh/tpoppo/gradient-boosting-tree)

The project implements the gradient boosting tree algorithm with the following details:

- Header-only library
- Single Thread
- Only oblivious decision trees

# How to run

Required tools: cmake, cppcheck
Optional tools: lcov (for coverage)

To run the tests

```bash
sh test.sh
```

To get the code coverage

```bash
sh coverage.sh
```

To get perf evaluation

```bash
sh perf.sh
```
