"""Script to generate docs based on given input directory (first argument)
and output directory (second argument)"""
import os
import subprocess
import sys
import glob
import nbformat

code_path = sys.argv[1]
out_path = sys.argv[2]

for notebook_path in glob.glob(f"{code_path}/examples/PySDM_examples/*/*.ipynb"):
    with open(notebook_path, encoding="utf8") as fin:
        with open(notebook_path + ".badges.md", "w", encoding="utf8") as fout:
            fout.write(nbformat.read(fin, nbformat.NO_CONVERT).cells[0].source)
subprocess.run(
    [
        sys.executable,
        "-We",
        "-m",
        "pdoc",
        "-o",
        f"{out_path}/html",
        "PySDM",
        "examples/PySDM_examples",
        "-t",
        "docs/templates",
        "--math",
        "--mermaid",
    ],
    env={**os.environ, "PDOC_ALLOW_EXEC": "1"},
    check=True,
)
