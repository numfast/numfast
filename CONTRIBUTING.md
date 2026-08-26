# Contributing to NumFast

## Architecture-first

NumFast develops via ADR (Architecture Decision Records).
Before implementing a feature, write an ADR.

## Development process

1. Write ADR -> `docs_ru/ADR-NNN_name.md`
2. Spec review
3. Implementation (CPU first, then GPU)
4. Golden tests
5. Conformance tests
6. Guardian review

## Rules

- Runtime is frozen (ADR-005). No changes to Runtime, Driver ABC, MemoryManager, ExecutionPacket.
- CPU is always reference. CPU == GPU is mandatory.
- Extensions are the only way to add functionality.
- No LLM-specific code in the SDK (AI-agnostic principle).

## Code style

- Python 3.11+
- Type hints required
- Docstrings in Russian (specs) or English (code)
