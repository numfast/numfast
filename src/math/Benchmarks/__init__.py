"""NumFast Benchmark Suite.

Modular performance benchmarks for compute engine:
  - AST parsing
  - AST -> ISA compilation
  - ISA execution (Python evaluator)
  - Indicators (SMA, EMA, ATR)
  - Full expression pipeline
  - QuoteTable operations

Each module exposes run() -> dict with results.
Run all: python -m Benchmarks.run_all
"""
