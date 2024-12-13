import os
import subprocess
import sys
import glob, nbformat

code_path = sys.argv[1]
out_path = sys.argv[2]

os.chdir(code_path)
os.environ["PDOC_ALLOW_EXEC"] = "1"

for notebook_path in glob.glob("examples/PySDM_examples/*/*.ipynb"):
    with open(notebook_path, encoding="utf8") as fin:
        with open(notebook_path + ".badges.md", 'w') as fout:
            fout.write(nbformat.read(fin, nbformat.NO_CONVERT).cells[0].source)
subprocess.run(["python", "-We", "-m", "pdoc", "-o", "html", "PySDM","examples/PySDM_examples", "-t", "docs/templates", "--math", "--mermaid"],env=os.environ,check=True)


