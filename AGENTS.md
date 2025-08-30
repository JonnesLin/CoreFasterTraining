# Repository Guidelines

## Project Structure & Module Organization
- `timm/`: core Python package (models, layers, data, optim, utils).
- `tests/`: pytest suite (`test_*.py`) covering layers, models, optimizers, utils.
- Top-level scripts: `train.py`, `validate.py`, `inference.py`, `benchmark.py`, `avg_checkpoints.py`.
- Aux folders: `convert/` (model conversion), `hfdocs/` (docs), `results/` (checkpoints/logs, git-ignored).

## Build, Test, and Development Commands
- Create env and install deps:
  - `python -m pip install -r requirements.txt`
  - `python -m pip install -r requirements-dev.txt`
  - `python -m pip install -e .`  (editable install)
- Run tests:
  - `pytest tests/` (all)
  - `pytest -k "substring" -n 4 tests/` (filter + parallel)
- Common scripts:
  - `python train.py --model resnet50 --data-dir /path --epochs 90`
  - `python validate.py --model resnet50 --data-dir /path`
  - `python inference.py --model resnet50 --input /path/to/img.jpg`

## Coding Style & Naming Conventions
- Follow Google Python style with tweaks:
  - 4-space indents; max line length 120.
  - Prefer hanging indents; avoid aligning with closing brackets.
- Names: `snake_case` for functions/vars/modules, `CamelCase` for classes, UPPER_SNAKE for constants.
- Type hints encouraged for public APIs; docstrings follow Google style.
- Formatting (suggested, not enforced):
  - `black --skip-string-normalization --line-length 120 <paths>`

## Testing Guidelines
- Framework: `pytest` with markers (`base`, `cfg`, `torchscript`, `features`, `fxforward`, `fxbackward`).
- Naming: place tests in `tests/` as `test_*.py`; use descriptive test names.
- Scope: add unit tests for new features/bug fixes; mark slow/integration sensibly and use `-k` to target.

## Commit & Pull Request Guidelines
- Commits: imperative mood and concise summary, e.g., `Add ConvNeXt-V2 factory`.
- Reference issues in the body (`Fixes #123`). Group unrelated refactors separately from feature/bugfix.
- PRs: include clear description, rationale, usage examples/CLI args, and test coverage. For training changes, share minimal logs or metrics.
- Before opening: run `pytest`, ensure style/typing sanity, and avoid unrelated reformatting.

## Security & Configuration Tips
- Do not commit large artifacts or secrets (tokens, datasets, checkpoints). Use `results/` for local outputs and Hugging Face Hub for sharing weights.
