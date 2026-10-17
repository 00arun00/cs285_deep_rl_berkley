# Reinforcement learning, in practice

This repository collects reinforcement-learning implementations and experiments, from imitation learning and policy gradients to offline RL and language-model training. Use this overview to choose a project, check its current status, and find its instructions.

## Choose a project

| Project | Topic | Status | Start here |
| --- | --- | --- | --- |
| HW1 · `hw1/` | Imitation learning with Push-T. | Implemented | [Orientation and practical entry path](hw1/index.md) |
| HW2 · `hw2/` | Policy gradients. | Implementation pending | [Project README](https://github.com/00arun00/cs285_deep_rl_berkley/blob/main/hw2/README.md) |
| HW3 · `hw3/` | Q-learning and actor–critic methods: DQN and SAC. | Implementation pending | [Project README](https://github.com/00arun00/cs285_deep_rl_berkley/blob/main/hw3/README.md) |
| HW4 · `hw4/` | Reinforcement learning for language models. | Implementation pending | [Project README](https://github.com/00arun00/cs285_deep_rl_berkley/blob/main/hw4/README.md) |
| HW5 · `hw5/` | Offline reinforcement learning. | Implementation pending | [Project README](https://github.com/00arun00/cs285_deep_rl_berkley/blob/main/hw5/README.md) |

HW2–HW5 contain starter material and example commands. Their presence does not mean the project implementations are complete or those workflows verified here.

## Final projects

Both final projects are **Scope under development**. The existing starter material suggests the directions below; the final topics and scope are not settled.

| Location | Tentative direction | Existing material |
| --- | --- | --- |
| `final_project_llm_rl/` | RL for language models and preference-based learning. | [Starter README](https://github.com/00arun00/cs285_deep_rl_berkley/blob/main/final_project_llm_rl/README.md) |
| `final_project_offline_online/problem/` | Learning from offline data, then improving through online interaction. | [Starter README](https://github.com/00arun00/cs285_deep_rl_berkley/blob/main/final_project_offline_online/problem/README.md) |

## Use the right environment

Each project manages its own Python dependencies. Python requirements differ between projects.

- **For project work:** use the chosen project's directory, README, and `pyproject.toml`. The offline-to-online project's dependency file is under `final_project_offline_online/problem/`.
- **For this documentation site:** use the repository-root environment. Authored pages live in `docs/`; [documentation authoring](contributing/index.md) explains the tooling and workflow.

## Explore the documentation

- [Push-T orientation](hw1/index.md): the implemented HW1 project, its workflow, and where to begin.
- [Python reference](reference/index.md): HW1 interfaces, docstrings, and source details.
- [Documentation authoring](contributing/index.md): how to maintain the site and its reference material.

Detailed authored guides currently focus on HW1. Use the other project READMEs as starter references alongside the status above.
