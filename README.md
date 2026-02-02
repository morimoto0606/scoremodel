# scoremodel

Research workspace for score-based generative models.

This repository is a mother repository designed to:
- manage multiple external research implementations (as git submodules)
- host my own extensions and experiments
- provide a reproducible development environment via VS Code Dev Containers

The goal is to compare, extend, and experiment with approaches such as:
- Song et al. (Score-based SDEs)
- Malliavin-based methods
- Manifold-based methods

without modifying upstream code directly.


Repository structure
--------------------

    scoremodel/
    ├─ upstream/                 # External research code (git submodules)
    │  └─ score_sde_pytorch/     # Song et al. (original implementation)
    │
    ├─ src/scoremodel_ext/       # My own extensions (editable install)
    │  └─ __init__.py
    │
    ├─ scripts/                  # Entry points for experiments
    │   ├─ train_*.py
    │   └─ sample_*.py
    │
    ├─ configs/                  # Experiment configurations (YAML / etc.)
    ├─ notebooks/                # Exploration / analysis (no core logic)
    ├─ tests/
    │
    ├─ requirements.txt          # Single source of truth for Python deps
    ├─ pyproject.toml            # Minimal config for `pip install -e .`
    └─ .devcontainer/            # Reproducible dev environment


Design principles
-----------------

1. Upstream code is not modified

All external implementations live under `upstream/`.
They are added as git submodules and treated as read-only references.

If modification is unavoidable, use a fork explicitly and document it.


2. All original work lives outside upstream

- New models, samplers, losses, wrappers → src/scoremodel_ext/
- Experiment entry points → scripts/
- Configuration → configs/

This keeps diffs minimal and makes upstream updates manageable.


3. Single environment definition

`requirements.txt` defines the entire Python environment.

Upstream `requirements.txt` files are reference only.
Dev Container, local runs, and cloud runs all rely on this single file.

This file is the single source of truth for dependencies.


4. Editable install for stable imports

The project is installed via:

    pip install -e .

This allows:
- stable imports (import scoremodel_ext)
- consistent behavior across scripts, notebooks, and tests
- no manual PYTHONPATH hacks


Setup
-----

Clone with submodules:

    git clone --recurse-submodules git@github.com:morimoto0606/scoremodel.git

If already cloned:

    git submodule update --init --recursive


Development environment (Dev Container)
---------------------------------------

This repository is intended to be used with VS Code Dev Containers.

1. Open the repository in VS Code
2. Select "Reopen in Container"
3. Dependencies are installed automatically

After startup, the following should work:

    python -c "import scoremodel_ext"


Dev Container and SSH (Git access)
---------------------------------

This repository uses SSH authentication inside the Dev Container
without copying private keys into the container.

How it works:
- SSH keys live on the host machine (macOS)
- The host ssh-agent is forwarded into the Dev Container
- Git commands inside the container use the same SSH identity

No private keys are stored in the container.


Host requirements:

1. SSH key exists (e.g. ~/.ssh/id_rsa or id_ed25519)
2. SSH key is added to ssh-agent:

       ssh-add ~/.ssh/id_rsa

3. The public key is registered on GitHub

Verify on host:

    ssh -T git@github.com


Dev Container configuration:

The following settings are used in .devcontainer/devcontainer.json:

    mounts:
      - SSH_AUTH_SOCK forwarded from host
    remoteEnv:
      - SSH_AUTH_SOCK=/ssh-agent


Verify SSH inside the container:

    ssh -T git@github.com

Expected output:

    Hi <github-username>! You've successfully authenticated...

If this works, the following will also work:
- git pull
- git push
- git submodule update
- git pull --recurse-submodules


Git and submodule notes
----------------------

Submodules are tracked by commit hash.

Updating an upstream implementation:

    cd upstream/<repo>
    git fetch
    git checkout <commit>
    cd ../..
    git add upstream/<repo>
    git commit -m "bump upstream <repo>"

Always clone with --recurse-submodules.


What this repository is not
---------------------------

- Not a polished library
- Not a PyPI package
- Not intended for direct reuse without context

This is a research workspace optimized for:
- flexibility
- traceability
- reproducibility


Notes for future me
-------------------

- Keep upstream clean
- Prefer adapters or wrappers over patches
- Treat requirements.txt as the single truth
- If something becomes central, consider splitting it into its own repo