# Internal projects

Experiments developed inside this project.

| Directory | Repository | Topic |
|---|---|---|
| `project_1` | [SAE-spot](https://github.com/NLeSC-Knowledge-Development/SAE-spot) | SAE feature on a spot-detection model |
| `project_2` | [SAE-MNIST](https://github.com/NLeSC-Knowledge-Development/SAE-MNIST) | SAE on MNIST classifiers |
| `project_3` | [SAE-Lorenz96](https://github.com/NLeSC-Knowledge-Development/SAE-Lorenz96) | SAE on a Lorenz-96 emulator |
| `project_4` | [SAE-GrayScott](https://github.com/NLeSC-Knowledge-Development/SAE-GrayScott) | SAE on a Gray-Scott emulator |

Each project is a submodule with its own history, dependencies and
`.gitignore`.

## Committing changes: two steps

A submodule is a separate repository. This repository does not store the
project's files; it stores one commit hash per submodule. A change therefore
needs two commits.

**1. Commit and push inside the project**

```bash
cd internal_projects/project_1
git add .
git commit -m "Add steering decay plot"
git push
```

**2. Record the new hash in this repository**

```bash
cd ../..
git status          # modified: internal_projects/project_1 (new commits)
git add internal_projects/project_1
git commit -m "Update project_1"
git push
```

The order matters: push the submodule commit before the parent commit, otherwise
the parent points at a commit that is not on the remote. Skipping step 2 leaves
the work only in the submodule; a fresh `git clone --recurse-submodules` of this
repository checks out the previous commit.

## Local data

Generated `data/`, `model/` and `__pycache__/` are ignored inside each
submodule. `git status` stays clean, but these files are deleted by
`git submodule deinit`, `git submodule update --force` or `git clean -xdf`.
Normal commit, checkout and push do not touch them.

## Environment

The shared environment is at the repository root:

```bash
source ../../.venv/bin/activate
```
