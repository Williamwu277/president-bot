## Setup

1. Setup the virtual environment
2. Install the requirements
3. Set up `pre-commit`

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pre-commit install
pre-commit run --all-files
```

## Running Commands

Run from repository root.

```bash
python -m src.play
python -m pytest
```
