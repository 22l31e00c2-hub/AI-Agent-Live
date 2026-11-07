"""Entry point so the agent runs as ``python -m cadagent``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
