# Agent Permissions and Project Rules

## Execution Scope & Auto-Approval
- The user grants full execution permission for all terminal commands, scripts, analysis, and file operations within the workspace `/Users/jeswintobias/attack-sequence-mining`.
- Do NOT prompt the user for permission when running commands, inspecting files, or executing scripts within `/Users/jeswintobias/attack-sequence-mining`.
- ALWAYS run all python commands, shell commands, scripts, tests, and data operations automatically without asking for user confirmation.
- Only ask for explicit confirmation if an operation would touch paths strictly outside this workspace directory or perform irreversible destructive system-level operations.

## Project Guidelines
- Workspace: `/Users/jeswintobias/attack-sequence-mining`
- Language: Python only (`pandas`, `numpy`, `matplotlib`, `mlxtend`).
- Do not use external PrefixSpan libraries for core implementation.
- All numbers reported must originate strictly from executed code.
- Reproducibility: Always fix random seeds.
- Modularity: Store configurations in `config.py`.
- Maintain concise explanatory comments for viva defense.
