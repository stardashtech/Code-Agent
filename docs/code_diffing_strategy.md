# Code Diffing Strategy Research and Decision

**Task:** TASK-002.3 - Research and Select Advanced Code Diffing Technique

**Goal:** Investigate options for semantic code diffing (AST diffing, control-flow graph analysis, LLM-based) and select an approach suitable for comparing code changes across Go, C#, Python, TypeScript, and JavaScript. The chosen technique should support identifying semantic differences, not just textual ones, to aid in understanding the impact of code changes (e.g., dependency updates, refactoring).

## Options Investigated

1.  **Abstract Syntax Tree (AST) Diffing:**
    *   **Pros:** Precise structural comparison, inherently language-aware, effective at identifying moved/renamed code blocks, numerous existing libraries (e.g., GumTree, language-specific parsers + diff algorithms).
    *   **Cons:** High implementation complexity for integrating robust parsers across all five target languages, potentially sensitive to minor syntactic changes without semantic impact, may not easily capture high-level intent or semantic equivalence of different structures.

2.  **Control-Flow Graph (CFG) Analysis:**
    *   **Pros:** Compares program execution flow, potentially better than AST diffing for semantic equivalence in some cases (e.g., recognizing different loop structures with the same outcome).
    *   **Cons:** Even higher implementation complexity than AST diffing (requires CFG generation tools for all target languages), likely computationally expensive, probably overkill for the primary use case of analyzing dependency updates or standard refactoring patterns.

3.  **Large Language Model (LLM) Based Diffing:**
    *   **Pros:** Potential to understand semantic meaning and intent, leverages existing multi-lingual capabilities of LLMs, aligns well with the project's existing LLM usage (e.g., TASK-001.5, TASK-003.4), can provide natural language summaries of changes, relatively lower integration complexity compared to building/integrating multiple language-specific parsers/analyzers.
    *   **Cons:** Potentially non-deterministic results, higher latency/cost per comparison than algorithmic methods, results can be harder to verify objectively, effectiveness depends heavily on LLM capabilities and prompt engineering, may be less precise than AST for detailed structural changes.

## Decision: LLM-Based Semantic Diffing

**Rationale:**

Given the requirement for multi-language support (Go, C#, Python, TS, JS) and the emphasis on *semantic* understanding (impact of changes) rather than just textual or purely structural diffs, the LLM-based approach offers the most pragmatic path forward.

*   **Multi-Language Support:** Leverages the inherent multi-lingual capabilities of modern LLMs, avoiding the significant effort required to integrate and maintain separate AST/CFG parsers and analyzers for five different languages.
*   **Semantic Understanding:** Aligns better with the goal of understanding the *meaning* and *implications* of code changes (e.g., identifying breaking changes, summarizing refactoring intent) compared to purely structural methods.
*   **Synergy with Project Architecture:** Fits well with the existing use of LLMs within the agent for tasks like information extraction and solution generation.
*   **Manageable Downsides:** Potential issues like cost, latency, and determinism can be mitigated through careful prompt design, caching strategies, setting appropriate expectations for precision, and potentially combining LLM analysis with basic textual diffs for a preliminary overview.

**Implementation Plan (for TASK-002.4):**

The implementation in `services/code_differ.py` will involve:
1.  Retrieving the code versions (e.g., file contents) to be compared.
2.  Developing carefully crafted prompts for the LLM service (`models/llm.py`). These prompts will guide the LLM to:
    *   Receive two code snippets/files as input.
    *   Identify key semantic differences (e.g., changes in logic, dependency usage, function signatures, error handling, added/removed features).
    *   Assess potential risks or breaking changes.
    *   Provide a structured output summarizing the findings (e.g., JSON or Markdown).
3.  The `code_differ.py` service will manage the interaction with the LLM and process its response. 