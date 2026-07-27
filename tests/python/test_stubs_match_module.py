"""``python/fitsy/__init__.pyi`` is hand-written; this checks it
against reality.

A stub that disagrees with the module is worse than no stub: the call
type-checks and then fails at runtime. These tests compare the names,
parameter lists, parameter kinds, **and default values** declared in
the stub against the compiled extension.

Defaults matter as much as names: a stub that claims a required
argument is optional makes the type checker bless a call that raises
``TypeError``.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import fitsy
import pytest

STUB_PATH = pathlib.Path(__file__).parents[2] / "python" / "fitsy" / "__init__.pyi"
STUB = ast.parse(STUB_PATH.read_text())

# Sentinel for "this parameter has no default", distinct from a
# declared default of `None`.
REQUIRED = "<required>"


def _stub_all() -> set[str]:
    for node in STUB.body:
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "__all__"
        ):
            return {elt.value for elt in node.value.elts}
    raise AssertionError(f"{STUB_PATH.name} declares no __all__")


def _stub_functions() -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in STUB.body if isinstance(n, ast.FunctionDef)}


def _stub_classes() -> dict[str, ast.ClassDef]:
    return {n.name: n for n in STUB.body if isinstance(n, ast.ClassDef)}


def _normalize_default(text: str) -> str:
    """Compare defaults by value, not by source spelling.

    The stub says ``"readonly"`` while ``repr()`` of the runtime
    default says ``'readonly'``; both denote the same string. Ints and
    floats are unified too, so a stub ``0`` matches a runtime ``0.0``
    only if they really are equal.
    """
    if text == REQUIRED:
        return REQUIRED
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
    if isinstance(value, float) and value.is_integer():
        return repr(int(value))
    return repr(value)


def _declared_params(node: ast.FunctionDef) -> list[tuple[str, str, str]]:
    """(name, kind, default) triples; kind is 'positional'/'keyword-only'."""
    args = node.args
    # `defaults` right-aligns against `args`; `kw_defaults` is
    # positionally aligned with `kwonlyargs` and holds None for
    # "no default".
    padding = [None] * (len(args.args) - len(args.defaults))
    out = [
        (
            a.arg,
            "positional",
            REQUIRED if d is None else _normalize_default(ast.unparse(d)),
        )
        for a, d in zip(args.args, padding + list(args.defaults))
    ]
    out += [
        (
            a.arg,
            "keyword-only",
            REQUIRED if d is None else _normalize_default(ast.unparse(d)),
        )
        for a, d in zip(args.kwonlyargs, args.kw_defaults)
    ]
    return out


def _runtime_params(obj) -> list[tuple[str, str, str]] | None:
    """Same shape as `_declared_params`, or None if not introspectable."""
    try:
        params = inspect.signature(obj).parameters.values()
    except (ValueError, TypeError):
        return None
    kinds = {
        inspect.Parameter.KEYWORD_ONLY: "keyword-only",
        inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional",
        inspect.Parameter.POSITIONAL_ONLY: "positional",
    }
    return [
        (
            p.name,
            kinds[p.kind],
            (
                REQUIRED
                if p.default is inspect.Parameter.empty
                else _normalize_default(repr(p.default))
            ),
        )
        for p in params
        if p.kind in kinds
    ]


def test_all_matches_runtime():
    assert _stub_all() == set(fitsy.__all__)


@pytest.mark.parametrize(
    "name", sorted(n for n in fitsy.__all__ if not n.startswith("__"))
)
def test_every_exported_name_is_declared(name):
    assert name in _stub_functions() | _stub_classes()


@pytest.mark.parametrize("name", sorted(_stub_functions()))
def test_module_function_signatures(name):
    obj = getattr(fitsy, name, None)
    assert obj is not None, f"stub declares {name}, module does not export it"
    runtime = _runtime_params(obj)
    if runtime is None:
        pytest.skip(f"{name} exposes no introspectable signature")
    assert runtime == _declared_params(_stub_functions()[name])


@pytest.mark.parametrize("cls_name", sorted(_stub_classes()))
def test_class_methods_exist(cls_name):
    """Every method a stub declares must exist on the runtime class.

    Only presence is checked: PyO3 does not expose introspectable
    signatures for `#[pymethods]`, so parameter comparison is not
    possible here the way it is for module-level functions.
    """
    cls = getattr(fitsy, cls_name, None)
    assert cls is not None, f"stub declares class {cls_name}, module does not export it"
    declared = {
        n.name
        for n in _stub_classes()[cls_name].body
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("__")
    }
    missing = {m for m in declared if not hasattr(cls, m)}
    assert not missing, f"{cls_name} is missing {sorted(missing)}"
