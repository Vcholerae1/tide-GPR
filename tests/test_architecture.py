"""Static dependency boundaries for the maintainable package layers."""

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1] / "src" / "tide"


def _internal_imports(path: Path) -> set[str]:
    relative_module = ".".join(
        ("tide", *path.relative_to(ROOT).with_suffix("").parts)
    )
    package = relative_module.split(".")[:-1]
    tree = ast.parse(path.read_text())
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names if alias.name.startswith("tide."))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("tide."):
                result.add(node.module)
            elif node.level:
                prefix = package[: len(package) - node.level + 1]
                target = ".".join((*prefix, *(node.module or "").split(".")))
                if target:
                    result.add(target)
    return result


def test_core_does_not_import_runtime_layers() -> None:
    forbidden = ("tide.maxwell", "tide.workflow", "tide.optim", "tide.backend_utils")
    imports = {
        imported
        for path in (ROOT / "core").glob("*.py")
        for imported in _internal_imports(path)
    }
    assert not any(
        imported == prefix or imported.startswith(prefix + ".")
        for imported in imports
        for prefix in forbidden
    )


def test_maxwell_does_not_import_consumer_layers() -> None:
    forbidden = ("tide.workflow", "tide.optim")
    imports = {
        imported
        for path in (ROOT / "maxwell").glob("*.py")
        for imported in _internal_imports(path)
    }
    assert not any(
        imported == prefix or imported.startswith(prefix + ".")
        for imported in imports
        for prefix in forbidden
    )


def test_optim_is_independent_of_solver_layers() -> None:
    forbidden = ("tide.maxwell", "tide.workflow", "tide.core")
    imports = {
        imported
        for path in (ROOT / "optim").glob("*.py")
        for imported in _internal_imports(path)
    }
    assert not any(
        imported == prefix or imported.startswith(prefix + ".")
        for imported in imports
        for prefix in forbidden
    )
