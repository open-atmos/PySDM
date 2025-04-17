"""
Script to generate docs based on given input directory (first argument)
and output directory (second argument), includes bibliography logic
"""

import json
import os
import re
import subprocess
import sys

import nbformat

# <TEMPORARILY COPIED FROM DEVOPS_TESTS>
from git.cmd import Git


def find_files(path_to_folder_from_project_root=".", file_extension=None):
    """
    Returns all files in a current git repo.
    The list of returned files may be filtered with `file_extension` param.
    """
    all_files = [
        path
        for path in Git(
            Git(path_to_folder_from_project_root).rev_parse("--show-toplevel")
        )
        .ls_files()
        .split("\n")
        if os.path.isfile(path)
    ]
    if file_extension is not None:
        return list(filter(lambda path: path.endswith(file_extension), all_files))

    return all_files


# </TEMPORARILY COPIED FROM DEVOPS_TESTS>


def generate_badges_md_files():
    """extracts badge-containing cell from each notebook and writes it to an .md file"""
    for notebook_path in find_files("examples/PySDM_examples", ".ipynb"):
        with open(notebook_path, encoding="utf8") as fin:
            with open(notebook_path + ".badges.md", "w", encoding="utf8") as fout:
                fout.write(nbformat.read(fin, nbformat.NO_CONVERT).cells[0].source)


def read_urls_from_json(code_path):
    """loads bibliography data from .json file"""
    bibliography_json_path = f"{code_path}/docs/bibliography.json"
    urls_from_json = {}
    if os.path.exists(bibliography_json_path):
        with open(bibliography_json_path, "r", encoding="utf8") as fin:
            urls_from_json = json.load(fin)
    return urls_from_json


def check_urls(urls_from_json):
    """checks if all bib entries are referenced from code and vice versa"""
    found_urls = []
    for extension in (".md", ".ipynb", ".py"):
        for full_path in find_files(file_extension=extension):
            with open(full_path, "r", encoding="utf-8") as fin:
                text = fin.read()
            for pattern in (
                r"\b(https://doi\.org/10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?![\"&\'<>^\\])\S)+)\b",
                r"\b(https://digitallibrary\.un\.org/record/(?:[0-9])+)\b",
                r"\b(http://mi\.mathnet\.ru/dan(?:[0-9])+)\b",
                r"\b(https://archive.org/details/(?:[0-9a-z_\.-])+)\b",
                r"\b(https://web.archive.org/web/(?:[0-9])+/https://(?:[0-9a-zA-Z_\.-/])+)\b",
            ):
                urls = re.findall(pattern, text)
                if urls:
                    found_urls.extend((full_path, url) for url in urls)

    unique_urls_found = {url for _, url in found_urls}
    unique_urls_read = set(urls_from_json.keys())

    for url in unique_urls_found:
        assert url in unique_urls_read, f"{url} not found in the json file"
    for url in unique_urls_read:
        assert url in unique_urls_found, f"{url} not referenced in the code"

    url_usages_found = {
        url: sorted({path for path, d in found_urls if d == url})
        for url in unique_urls_found
    }
    for url in unique_urls_read:
        assert set(url_usages_found[url]) == set(
            sorted(urls_from_json[url]["usages"])
        ), (
            f"{url} usages mismatch (please fix docs/bibliography.json):\n"
            f"\texpected: {url_usages_found[url]}\n"
            f"\tactual:   {urls_from_json[url]['usages']}"
        )


def create_references_html(urls_from_json, code_path):
    """writes HTML file with the reference list"""
    with open(
        f"{code_path}/docs/templates/bibliography.html", "w", encoding="utf8"
    ) as fout:
        fout.write("<ol>\n")
        for url, data in sorted(
            urls_from_json.items(), key=lambda item: item[1]["label"].lower()
        ):
            fout.write("<li>\n")
            fout.write(
                f'<a href="{url}">{data["label"]}: "<em>{data["title"]}</em>"</a>\n'
            )
            fout.write('<ul style="list-style-type:square;font-size:smaller;">')
            for path in sorted(data["usages"]):
                fout.write(
                    f'<li><a href="https://github.com/open-atmos/PySDM/tree/main/{path}">'
                )
                fout.write(f"{path}</a></li>")
            fout.write("</ul>\n")
        fout.write("</ol>\n")


def run_pdoc(code_path, out_path):
    """generates docs in HTML formay by executing pdoc"""
    subprocess.run(
        [
            sys.executable,
            "-We",
            "-m",
            "pdoc",
            "-o",
            f"{out_path}/html",
            f"{code_path}/PySDM",
            f"{code_path}/examples/PySDM_examples",
            "-t",
            f"{code_path}/docs/templates",
            "--math",
            "--mermaid",
        ],
        env={**os.environ, "PDOC_ALLOW_EXEC": "1"},
        check=True,
    )


def _main():
    assert len(sys.argv) == 3, f"usage: {sys.argv[0]} code_path out_path"
    code_path, out_path = sys.argv[1], sys.argv[2]
    generate_badges_md_files()
    urls_from_json = read_urls_from_json(code_path)
    check_urls(urls_from_json)
    create_references_html(urls_from_json, code_path)
    run_pdoc(code_path, out_path)


if __name__ == "__main__":
    _main()
