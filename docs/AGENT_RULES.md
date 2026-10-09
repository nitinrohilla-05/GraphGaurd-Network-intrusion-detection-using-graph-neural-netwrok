# GraphGuard — Agent Rules (add as a workspace rule in Antigravity)

1. Before any task, read `docs/SPEC.md` and `docs/decisions.md`. The spec is the source of truth.
2. Work on exactly one phase per task. Never start the next phase without my approval.
3. Always produce an implementation-plan artifact first and wait for approval before writing code.
4. Do not change the definitions of features N1–N6 in the spec. If you think a change is needed, propose it and wait.
5. Never add a public git remote, push to a public repository, publish packages, or paste project code into public websites. The repository is private (possible patent).
6. Never run `sudo`, Mininet, Open vSwitch, `hping3`, `nmap` or any traffic-generation command without asking me first. Attack tools run only inside the isolated Mininet lab.
7. Use `DryRunEnforcer` unless the task explicitly says to use the live lab.
8. Every new module needs pytest unit tests. Run `make test` before saying a task is finished. Show me the real output, never claim tests passed without running them.
9. No secrets or absolute paths in code; use `.env` and `config/*.yaml`.
10. Finish every task with a walkthrough artifact (what was built, commands run, results, known issues) and a dated entry in `docs/decisions.md` stating whether each idea came from me or from the agent.
11. Run the SDN controller (os-ken) in its own container with its own Python version; the rest of the project uses Python 3.11.
12. If something is ambiguous, ask instead of guessing.
