# synthlang

SynthLang, renamed PolyThLang in September 2025: an unfinished Rust compiler and a Python package for a small pipeline language; the code is on the `master` branch, `main` holds only the license.

## Status

archived

Development stopped on 2025-09-16 (last commit on `master`, 0b1ada1, "Fix PolyThLang basic functionality and add working CLI"). The `main` branch has one commit from 2025-09-10 containing only `LICENSE`, and shares no history with `master`. The code is kept for reference.

Tried on 2026-09-10, all on `master`:

- Python: `pip install -e .` into a fresh venv succeeded. Every CLI entry point fails before it starts: `polythlang/cli.py` has a nested triple-quoted string at line 211 to 215 and Python raises `IndentationError: unexpected indent (cli.py, line 214)`. `simple_test.py`, the bundled smoke test, printed nothing and was killed after 20 seconds.
- Rust: `cargo metadata --no-deps --offline` fails to load the manifest. `Cargo.toml` names nine workspace members; only `compiler` and `stdlib` exist. The other seven directories (a runtime, a semantic pass, a quantum module, and three under `tools/`, among them) are absent.
- GitHub Actions: the six CI runs on `master` (2025-09-16 to 2025-09-17) all failed.

## Install and first run

Not maintained. No supported install path.

`polythlang-mbc` 1.0.0, 1.0.1, and 1.0.2 were uploaded to PyPI on 2025-09-16. The 1.0.2 wheel downloaded today contains the same broken `cli.py`; `polythlang --help` cannot run from it. Do not install it expecting a working tool. The npm name in `package.json` (`@michaelcrowe11/polythlang`) and the crate name in `Cargo.toml` (`synthlang`) were never published (404 on both registries today). The PyPI project named `synthlang` belongs to someone else.

## What runs today

Nothing is maintained.

What the `master` branch contains:

- `polythlang/`: 14 Python modules (parser, compiler, runtime, executor, evaluator, monitor, optimization, polyglot, and others). 13 pass `python -m py_compile`; `cli.py` does not. `pyproject.toml` and `setup.py` define package `polythlang-mbc` 1.0.2 with `click`, `rich`, `pyyaml`, `requests`, `aiohttp`, `pydantic` as dependencies and entry points `polythlang` and `polyth`.
- `compiler/`: Rust crate `synth-compiler` 0.1.0 (lexer, parser, type checker, JavaScript and WebAssembly code generators). `stdlib/`: Rust crate with core, math, string, json, net, crypto, and other modules. `src/`: 18 Rust files for a `synth` binary. None of it builds as a workspace because of the missing members above.
- `examples/`: 11 `.synth` files and one `.poly` file. `docs/SYNTAX.md` and `docs/USER_GUIDE.md`.
- `tools/vscode-extension/`: a TextMate grammar, snippets, and an `extension.ts` for `.synth` files.
- `ide/index.html`, `vercel.json` (routes to an `api/index.js` that does not exist), `Dockerfile`, two GitHub Actions workflows.
- Committed build artifacts: `polythlang/__pycache__/` and `polythlang_mbc.egg-info/`.
- Status notes written at the time: `PUBLISHING_STATUS.md`, `PUBLISH_INSTRUCTIONS.md`, `REAL_WORLD_ROADMAP.md`, `COMPILER_DEMO.md`.

## Limits

- This is not a working programming language or pipeline tool. No command in the repository runs to completion.
- The example programs describe quantum circuits, tensor autodiff, probabilistic programming, and model pipelines. No code in this repository executes any of them. The `run` command in `cli.py` is labelled "Simulate pipeline execution" in its own comment.
- The Python `Compiler` class is meant to emit Python and JavaScript source text from `.synth` input. The bundled test that exercises it hangs.
- `cli.py` still reports itself as "SynthLang" version 1.0.0 while the package is `polythlang-mbc` 1.0.2. The two names refer to the same project before and after the 2025-09-15 rename.
- `pyproject.toml` and `package.json` point to `github.com/MichaelCrowe11/polythlang`, which does not exist. This repository is the only home of the code.
- The `main` and `master` branches are unrelated. Anyone cloning the default branch gets only the license.
- No tests run. `test_compiler.py` and `test_enhancement.py` compile but were not executed today because the modules they import include the hanging compiler.

## License and contact

Apache License 2.0 (see `LICENSE`).

Contact: michael@crowelogic.com
