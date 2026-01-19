# Probing-JP-Politeness (base_LineDistilBERT) — Setup with `uv`

This repo consists of:
- `pyproject.toml` (project metadata + dependencies)
- `uv.lock` (locked, reproducible dependency set) 
- `.python-version` (preferred Python version) 
- `config.yaml` (experiment config)  
- `requirements.txt` (pip fallback)  
- Code under `src_base/`

The recommended way to set up this project is with [uv](#1-uv-1), which manages:
- Python installation (optional)
- a local virtual environment
- dependency syncing from `uv.lock`

If you don't use uv, you can still keep working with [pip](#2-pip-1) + a standard environment.

---
### 1. uv
a. [Install uv](#install-uv)

b.  [Clone the repo](#clone-the-repo)

c.  [Create/sync the environment](#create/sync-the-environment)

d.  [Run scripts using uv run](#run-scripts-using-uv-run)


### 2. pip
a. [Install Python 3.11+](#install-python-311)(optional)

b. [Create and activate a virtual environment](#create-and-activate-a-virtual-environment)

c. [Install dependencies](#install-dependencies)

d. [Run scripts](#run-scripts)

---

## 1. uv


## Install `uv`

Before getting into the setup with the project dependencies, you need to [install uv on your device](https://docs.astral.sh/uv/).

### macOS / Linux

From the **terminal**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You can also use wget:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

### Windows

From **PowerShell**

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

To verify you've installed uv correctly, run:
```bash
uv --version
```
If the installation is succsessful, the version of uv should be returned.

---

## Clone the repo
```bash
git clone https://github.com/shuhashi0352/Probing-JP-Politeness.git
cd Probing-JP-Politeness
```

**For now...** 

Since LINE-DistilBERT is implemented in the different branch, you need to switch to the branch `base_LineDistiBERT`.

```bash
git checkout base_LineDistilBERT
```
---
## Create/sync the environment
#### Option A) From the repo root...
```bash
uv sync
```
This creates a local environment (most likely `.venv/`) and installs exactly what’s located in `uv.lock`.

#### Option B) if you need a specific Python version:

Since this repo includes `.python-version`, you can ask uv to manage Python.
```bash
uv python install
uv sync
```

---

## Run scripts using uv run
```bash
uv run src_base/run_line.py
```

> **Caution**: Don’t activate the venv manually. Just run commands through uv

---

> **NOTE:** You can edit `config.yaml` to change experiment settings / paths.

---

## 2. pip
This project can also be run using a standard virtual environment + `pip`. So, if you don't use uv, please follow the instruction below.

---
## Install Python 3.11+
This project uses Python 3.11+, so check if your Python version is higher than 3.11:
```bash
python --version
```

If any version lower than 3.11 is returned, you need to install a newer Python (see below):

#### macOS
From the **terminal** (I'd recommend installing it via Homebrew):
```bash
brew install python@3.11
python3.11 --version
```
#### Windows
[Install the latest Python on your device](https://www.python.org/)

Then from **powershell**:
```bash
py -3.11 --version
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Linux
From the **terminal**, use your package manager:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
```
---
## Create and activate a virtual environment

### macOS / Linux
From the **terminal**:
```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows
From **powershell**:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```
---

## Install dependencies
Install all the dependencies from `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run scripts
Run your training/evaluation scripts the same way as with uv, just using python:
```bash
python src_base/run.line.py
```