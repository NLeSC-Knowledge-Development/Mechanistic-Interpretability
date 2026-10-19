# Mechanistic Interpretability

Sparse-autoencoder and causal-steering experiments on neural models, from toy
dynamical systems up to ECMWF's WeatherGenerator.

## Repository layout

```
.
├── internal_projects/          # our own experiments
│   ├── project_1/              # submodule: SAE-spot
│   ├── project_2/              # submodule: SAE-MNIST
│   ├── project_3/              # submodule: SAE-Lorenz96
│   └── project_4/              # submodule: SAE-GrayScott
├── external_projects/          # third-party code and integrations
└── extra_materials/            # papers, figures, notes
    └── papers/
```

`internal_projects/project_1` … `project_4` are git submodules pointing at their
own repositories under `NLeSC-Knowledge-Development`. `external_projects` and
`extra_materials` are plain directories tracked by this repository.

## Cloning

```bash
git clone --recurse-submodules git@github.com:NLeSC-Knowledge-Development/Mechanistic-Interpretability.git
```

Existing clone:

```bash
git submodule update --init --recursive
```

## Working inside a submodule

Commit and push inside the submodule first, then commit the updated gitlink in
this repository:

```bash
cd internal_projects/project_1
git add . && git commit -m "..." && git push
cd ../..
git add internal_projects/project_1
git commit -m "Update project_1"
```
