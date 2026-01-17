# Context-Engine Installation Guide

## Quick Start

### Development Installation

Install the package in editable mode to enable the `ctx` CLI:

```bash
pip install -e .
```

### Using the CLI

After installation, the CLI is available in three ways:

1. **As `ctx` command** (recommended):
   ```bash
   ctx --help
   ctx status
   ctx status --json
   ctx status --verbose
   ```

2. **As `ctx-cli` command**:
   ```bash
   ctx-cli --help
   ctx-cli status
   ```

3. **As a Python module**:
   ```bash
   python -m scripts.ctx_cli --help
   python -m scripts.ctx_cli status
   ```

### Verify Installation

Check that the CLI is properly installed:

```bash
# Check version
ctx --version

# Show help
ctx --help

# Test status command
ctx status --help
```

Expected output:
```
ctx-cli 0.1.0
```

## Entry Points

The package defines two entry points in `pyproject.toml`:

```toml
[project.scripts]
ctx = "scripts.ctx_cli:main"
ctx-cli = "scripts.ctx_cli:main"
```

Both commands invoke the same `main()` function from `scripts/ctx_cli/main.py`.

## Exit Codes

The CLI follows standard Unix conventions:

- `0`: Success
- `1`: General error (no command specified, command failed)
- Other non-zero values: Command-specific errors

## Troubleshooting

### Command not found after installation

If `ctx` or `ctx-cli` commands are not found after installation:

1. Check if the package is installed:
   ```bash
   pip list | grep context-engine
   ```

2. Find where entry points are installed:
   ```bash
   which ctx
   which ctx-cli
   ```

3. Ensure your Python bin directory is in PATH:
   ```bash
   echo $PATH
   ```

4. Reinstall the package:
   ```bash
   pip uninstall context-engine
   pip install -e .
   ```

### Import errors

If you get import errors when running the CLI:

1. Install in development mode with dependencies:
   ```bash
   pip install -e .
   ```

2. Or install without dependencies (if you have them already):
   ```bash
   pip install -e . --no-deps
   ```

### Module invocation always works

Even if entry points fail, you can always use:
```bash
python -m scripts.ctx_cli
```

This works as long as the `scripts/ctx_cli/__main__.py` file exists.

## Development

### Adding New Commands

1. Create command handler in `scripts/ctx_cli/commands/`
2. Register command in `scripts/ctx_cli/commands/__init__.py`
3. The command will be automatically available via all entry points

### Testing Entry Points

```bash
# Test all three invocation methods
ctx --version
ctx-cli --version
python -m scripts.ctx_cli --version

# All should output: ctx-cli 0.1.0
```
