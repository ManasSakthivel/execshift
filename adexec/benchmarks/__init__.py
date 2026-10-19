"""Benchmark framework: declarative suites, a policy-agnostic runner, recorded runs."""

from .catalog import SUITES, get_suite
from .runner import BenchmarkResult, record_run, run_suite, suite_from_config
from .suite import BenchmarkSuite, StatisticalProcedure, named_policies

__all__ = ["SUITES", "BenchmarkResult", "BenchmarkSuite", "StatisticalProcedure",
           "get_suite", "named_policies", "record_run", "run_suite", "suite_from_config"]
