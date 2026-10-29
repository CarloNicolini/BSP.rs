"""Check documented Python calls against the installed SDK; compile Rust examples."""
from __future__ import annotations

import argparse
import ast
import doctest
import inspect
import json
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap

import pybsp_rust as bsp

ROOT = Path(__file__).resolve().parents[1]


def snippets(path):
    """Yield language, location, and source from Markdown/RST documentation."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        for match in re.finditer(r"^```([^\n]*)\n(.*?)^```", text, re.M | re.S):
            language, body = match.groups()
            line = text.count("\n", 0, match.start()) + 2
            if language.strip() in ("python", "rust"):
                yield language.strip(), line, body
            elif language.strip() in ("sh", "bash"):
                for heredoc in re.finditer(
                    r"[^\n]*\bpython(?:[0-9.]*)?\s+-\s*<<\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)^\1\s*$",
                    body, re.M | re.S,
                ):
                    yield "python", line + body.count("\n", 0, heredoc.start()) + 1, heredoc[2]
    else:
        lines = text.splitlines()
        for index, line in enumerate(lines):
            directive = re.fullmatch(r"\.\. (?:code-block|code):: (python|rust)", line)
            if directive:
                end = index + 1
                while end < len(lines) and (not lines[end].strip() or lines[end].startswith("   ")):
                    end += 1
                body = "\n".join(lines[index + 1:end]).strip("\n")
                yield directive[1], index + 2, textwrap.dedent(body)
        examples = doctest.DocTestParser().get_examples(text)
        if examples:
            yield "python", 1, "\n".join(example.source for example in examples)


def check_python(source, location, symbols=None):
    """Parse Python and bind recognizable SDK calls without executing workloads."""
    symbols = symbols if symbols is not None else {name: (getattr(bsp, name), False) for name in bsp.__all__}
    symbols.setdefault("pybsp_rust", (bsp, False))

    def resolve(node):
        if isinstance(node, ast.Name):
            return symbols.get(node.id, (None, False))
        if isinstance(node, ast.Attribute):
            parent, instance = resolve(node.value)
            if parent is not None:
                if parent in (bsp.Formula, bsp.Solver, bsp.Options, bsp.ExecutionOptions, bsp.CancellationToken):
                    if not hasattr(parent, node.attr):
                        # Two public instance attributes are assigned during construction.
                        if parent is bsp.Solver and node.attr == "formula":
                            return bsp.Formula, True
                        if parent is bsp.Solver and node.attr == "cancellation_token":
                            return bsp.CancellationToken, True
                        raise ValueError(f"{location}: {parent.__name__} has no attribute {node.attr!r}")
                value = getattr(parent, node.attr, None)
                return value, instance and callable(value)
        return None, False

    tree = ast.parse(source, filename=location)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pybsp_rust":
            for item in node.names:
                if item.name != "*":
                    value = getattr(bsp, item.name)  # Also catches removed imports.
                    symbols[item.asname or item.name] = value, False
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            target, _ = resolve(node.value.func)
            result = None
            if target in (bsp.Formula, bsp.Solver, bsp.Options, bsp.ExecutionOptions):
                result = target
            elif target is not None and isinstance(node.value.func, ast.Attribute):
                name = node.value.func.attr
                if name in ("random", "from_dimacs", "residual_formula"):
                    result = bsp.Formula
                elif name == "solve":
                    result = bsp.SolveResult
            elif target is bsp.solve:
                result = bsp.SolveResult
            if result is not None:
                for name in node.targets:
                    if isinstance(name, ast.Name):
                        symbols[name.id] = result, True
        elif isinstance(node, ast.Call):
            target, bound = resolve(node.func)
            if target is None or not callable(target):
                continue
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(k.arg is None for k in node.keywords):
                continue  # Expanded dictionaries cannot be bound statically.
            try:
                signature = inspect.signature(target)
            except ValueError:
                continue
            if bound and "self" in signature.parameters:
                signature = signature.replace(parameters=[p for name, p in signature.parameters.items() if name != "self"])
            try:
                signature.bind(*([None] * len(node.args)), **{k.arg: None for k in node.keywords})
            except TypeError as error:
                raise ValueError(f"{location}:{node.lineno}: {ast.unparse(node.func)}: {error}") from error


def check_documents(root=ROOT, compile_rust=False):
    """Validate tracked documentation, optionally compiling every Rust block."""
    paths = subprocess.check_output(["git", "-C", str(root), "ls-files"], text=True).splitlines()
    python_count = 0
    rust = []
    for name in paths:
        path = root / name
        if path.suffix != ".md" and not (name.startswith("docs/") and path.suffix == ".rst"):
            continue
        symbols = {name: (getattr(bsp, name), False) for name in bsp.__all__}
        for language, line, source in snippets(path):
            location = f"{name}:{line}"
            if language == "python":
                check_python(source, location, symbols)
                python_count += 1
            else:
                rust.append((location, source))
    if compile_rust and rust:
        with tempfile.TemporaryDirectory(prefix="bsp-rust-docs-") as directory:
            project = Path(directory)
            manifest = ('[package]\nname = "bsp-doc-examples"\nversion = "0.0.0"\nedition = "2021"\n'
                        '[dependencies]\nbsp-rust = { path = ' + json.dumps(str(root)) + ' }\n')
            for index, (location, source) in enumerate(rust):
                filename = f"snippet_{index}.rs"
                (project / filename).write_text(source)
                manifest += f'[[bin]]\nname = "snippet-{index}"\npath = "{filename}"\n'
                print(f"Compiling {location}", flush=True)
            (project / "Cargo.toml").write_text(manifest)
            subprocess.run(["cargo", "check", "--offline", "--manifest-path", str(project / "Cargo.toml"),
                            "--target-dir", str(root / "target" / "doc-examples")], check=True)
    return python_count, len(rust)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--rust", action="store_true", help="compile Rust examples against this checkout")
    args = parser.parse_args()
    python_count, rust_count = check_documents(compile_rust=args.rust)
    print(f"Checked {python_count} Python snippets and {rust_count} Rust snippets (compiled={args.rust}).")
