"""Regenerate checked-in SVG figures from their original TikZ sources."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "diagrams"
OUTPUT = ROOT / "docs" / "_static" / "diagrams"
SVG = "http://www.w3.org/2000/svg"
TITLES = {
    "factor-graph": "A signed SAT factor graph and a satisfying assignment",
    "cavity-messages": "Cavity message directions on a signed factor graph",
    "survey-states": "True, False, and unfrozen warning-survey states",
}


def main():
    for program in ("latex", "dvisvgm"):
        if not shutil.which(program):
            raise SystemExit(f"Install a TeX distribution providing {program}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ET.register_namespace("", SVG)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    with tempfile.TemporaryDirectory(prefix="bsp-tikz-") as directory:
        for name, title in TITLES.items():
            commands = [
                ["latex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape",
                 f"-output-directory={directory}", f"{name}.tex"],
                ["dvisvgm", "--no-fonts", "--exact",
                 f"--output={OUTPUT / (name + '.svg')}",
                 str(Path(directory) / f"{name}.dvi")],
            ]
            for command in commands:
                run = subprocess.run(command, cwd=SOURCE, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if run.returncode:
                    raise SystemExit(run.stdout[-6000:])
            path = OUTPUT / f"{name}.svg"
            tree = ET.parse(path)
            root = tree.getroot()
            root.set("role", "img")
            root.set("aria-labelledby", "title desc")
            text = ET.Element(f"{{{SVG}}}title", id="title")
            text.text = title
            root.insert(0, text)
            description = ET.Element(f"{{{SVG}}}desc", id="desc")
            description.text = (
                "Clauses are left, variables right. White fill means True, grey "
                "False, hatching unfrozen. Solid edges are positive literals, "
                "dashed edges negated literals. Labels repeat truth values."
            )
            root.insert(1, description)
            x, y, width, height = map(float, root.get("viewBox").split())
            x, y, width, height = x - 5, y - 5, width + 10, height + 10
            root.set("viewBox", f"{x} {y} {width} {height}")
            root.set("width", f"{width}pt")
            root.set("height", f"{height}pt")
            canvas = ET.Element(f"{{{SVG}}}rect", {
                "x": str(x), "y": str(y), "width": str(width),
                "height": str(height), "fill": "white",
            })
            root.insert(2, canvas)
            tree.write(path, encoding="utf-8", xml_declaration=True)
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
