# Using uv for Python Package Management

This project now uses [uv](https://github.com/astral-sh/uv) for Python package management instead of pip. `uv` is a fast Python package installer and resolver written in Rust.

## Benefits of uv

- **10-100x faster** than pip
- **Better dependency resolution** - handles complex dependency conflicts more reliably
- **Drop-in replacement** - uses the same `requirements.txt` format
- **Cross-platform** - works on Linux, macOS, and Windows

## Installation

The setup scripts will automatically install `uv` if it's not found. To install manually:

**Linux/macOS/WSL:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, restart your terminal or run:
```bash
source ~/.bashrc  # or ~/.zshrc
```

## Usage

### Setup (Automatic)
The setup scripts (`scripts/setup.sh` or `scripts/setup.ps1`) will automatically use `uv` to:
1. Create the virtual environment
2. Install all dependencies from `requirements.txt`

### Manual Usage

**Create virtual environment:**
```bash
cd backend
uv venv
```

**Install dependencies:**
```bash
# Activate venv first
source venv/bin/activate  # Linux/macOS/WSL
# or
.\venv\Scripts\Activate.ps1  # Windows PowerShell

# Install with uv
uv pip install -r requirements.txt

# Or specify Python explicitly
uv pip install --python venv/bin/python -r requirements.txt
```

**Update dependencies:**
```bash
uv pip install --upgrade -r requirements.txt
```

**Add a new package:**
```bash
uv pip install package-name
# Then update requirements.txt
uv pip freeze > requirements.txt
```

## Requirements.txt Format

`uv` uses the same `requirements.txt` format as pip, so no changes are needed. The existing file works as-is.

## Troubleshooting

**If uv is not found after installation:**
- Restart your terminal
- Add `~/.cargo/bin` to your PATH:
  ```bash
  export PATH="$HOME/.cargo/bin:$PATH"
  ```

**If you encounter dependency conflicts:**
`uv` has better conflict resolution than pip, but if issues persist:
1. Check `requirements.txt` for version conflicts
2. Try `uv pip install --resolution=highest -r requirements.txt`
3. Review the error messages - `uv` provides clearer conflict information

## Migration from pip

If you already have a venv created with pip, you can:
1. Keep using it (uv works with existing venvs)
2. Or recreate it: `rm -rf venv && uv venv && uv pip install -r requirements.txt`

The setup scripts handle this automatically.


