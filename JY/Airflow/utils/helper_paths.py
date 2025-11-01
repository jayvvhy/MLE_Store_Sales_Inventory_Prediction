import os

def resolve_relative_path(relative_path: str) -> str:
    """Resolves a path relative to project root (one level above utils/)."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, relative_path)