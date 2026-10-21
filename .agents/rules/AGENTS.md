# Agent Permissions and Project Rules

## Execution Scope & Auto-Approval
- The user grants full execution permission for all terminal commands, scripts, analysis, and file operations within the workspace `/Users/jeswintobias/attack-sequence-mining`.
- Do NOT prompt the user for permission when running commands, inspecting files, or executing scripts within `/Users/jeswintobias/attack-sequence-mining`.
- ALWAYS run all python commands, shell commands, scripts, tests, and data operations automatically without asking for user confirmation.
- Only ask for explicit confirmation if an operation would touch paths strictly outside this workspace directory or perform irreversible destructive system-level operations.

## Project Guidelines & Core Methodology
- Workspace: `/Users/jeswintobias/attack-sequence-mining`
- Language: Python only (`pandas`, `numpy`, `matplotlib`, `mlxtend`).
- Do not use external PrefixSpan libraries for core implementation (tests only).
- All numbers reported must originate strictly from executed code.
- Reproducibility: Always fix random seeds (`config.py`).
- Modularity: Store configurations and paths in `config.py`.
- Maintain concise explanatory comments for viva defense.

## Validation & Quality Invariants (Validation is Key)
- **Self-Validation**: Always validate code, scripts, outputs, dimensions, and data partitions automatically before reporting to the user.
- **Strict Split Isolation**: Fit on train only, tune on validation only. Never touch test until STOP 3 is approved.
- **Metric Integrity**: Headline metrics are unweighted top-1, top-3, top-5 accuracy and MRR over unique sequences. Weighted metrics are secondary. Every reported metric must be labelled MEASURED (from executed code) or ASSUMPTION (design choice).
- **"What Can Be Wrong" Requirement**: After completing every unit of work, explicitly present a self-critique detailing what could be wrong, potential pitfalls, failure modes, data leakage risks, or methodological threats to validity.
- **Decisions Log**: Maintain and update `docs/DECISIONS_LOG.md` with every design decision, alternative rejected, and viva justification.
