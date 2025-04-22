# Code Diffing Strategy Decision (TASK-002.3)

**Date:** 2024-07-26

## Goal

Select an advanced code diffing technique suitable for comparing code changes semantically across multiple languages (Python, JS/TS, Go, C#) as part of FEAT-002.

## Options Considered

1.  **Abstract Syntax Tree (AST) Diffing:**
    *   Pros: Understands code structure, focuses on semantic changes, mature parsers available for many languages.
    *   Cons: Requires language-specific parsers and diff algorithms, potentially complex setup.
    *   Tools/Libs: `tree-sitter`, `GumTree`, language-specific AST modules.

2.  **Control Flow Graph (CFG) Analysis:**
    *   Pros: Detects changes in execution logic.
    *   Cons: Highly complex, less common tooling, high performance cost.
    *   Suitability: Likely overkill for this project's scope.

3.  **LLM-Based Diffing:**
    *   Pros: Potential language agnosticism, can interpret "intent", can generate natural language summaries.
    *   Cons: Variable accuracy/consistency, potential cost/latency, non-deterministic, structured output challenges.
    *   Suitability: More promising for *analyzing/summarizing* diff results rather than being the primary diff mechanism. Can be integrated later (e.g., in TASK-004.1).

4.  **Text-Based Diffing (Standard Diff):**
    *   Pros: Simple, fast, universal.
    *   Cons: Doesn't understand code structure or semantics, sensitive to formatting.
    *   Suitability: Insufficient for semantic comparison needs.

## Decision: AST Diffing using `tree-sitter`

**Rationale:**

*   **Semantic Understanding:** AST diffing provides the best balance between understanding structural/semantic code changes and implementation feasibility compared to CFG or pure LLM approaches.
*   **Multi-Language Support:** `tree-sitter` offers a framework for using parsers for numerous languages, including the target languages (Python, JS/TS, Go, C#), reducing the need to find entirely separate solutions for each.
*   **Maturity:** Both `tree-sitter` and AST diffing concepts (like GumTree) are relatively mature areas.
*   **Integration with LLM:** Standard diff outputs from an AST comparison can still be fed into an LLM later for higher-level analysis or risk assessment, combining the strengths of both approaches.

## Next Steps (for TASK-002.4 Implementation)

1.  **Integrate `tree-sitter`:** Add the `tree-sitter` Python library to the project dependencies.
2.  **Acquire Language Grammars:** Download or build the necessary `tree-sitter` grammar files (`.so` or `.wasm`) for Python, JavaScript, TypeScript, Go, and C#.
3.  **Implement AST Parsing:** Create a utility function that takes code content and language identifier, uses the appropriate `tree-sitter` grammar to parse it into an AST.
4.  **Implement Diff Algorithm:** Research and implement or adapt an AST diffing algorithm. Options:
    *   Use a library built on top of `tree-sitter` for diffing if one exists and is suitable.
    *   Implement a simpler node-matching and comparison algorithm (e.g., based on node type, identifier, and content).
    *   Adapt concepts from algorithms like GumTree.
5.  **Define Diff Output:** Determine a structured format for the diff results (e.g., list of changes with type, location, old/new nodes).

## Risks & Considerations

*   **Grammar Availability/Quality:** Ensure high-quality, maintained `tree-sitter` grammars are available for all target languages.
*   **Diff Algorithm Complexity:** Implementing a robust AST diff algorithm can be complex.
*   **Performance:** Parsing large files into ASTs and diffing them might have performance implications. 