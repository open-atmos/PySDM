import re
import os
import json

bibliography_json_path = "bibliography.json"
bibliography_html_path = "./templates/bibliography.html"

# find all referenced dois
found_dois = []
included = [".md", ".ipynb", ".tex", ".txt", ".html", ".py"]
for root, subdirs, files in os.walk(".."):
    for file in files:
        full_path = os.path.join(root, file)
        if any([full_path.endswith(x) for x in included]):
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
                dois = re.findall(r"\b(https://doi\.org/10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?![\"&\'<>^\\])\S)+)\b", text)
                if dois:
                    found_dois.extend((full_path, doi) for doi in dois)

# get all unique dois and their usages
unique_dois_found = set([doi for _, doi in found_dois])
doi_usages_found = {doi: [path for path, d in found_dois if d == doi] for doi in unique_dois_found}


# read dois from .json and comparte with dois found as a test
dois_from_json = {}
if os.path.exists(bibliography_json_path):
    with open(bibliography_json_path, "r") as f:
        dois_from_json = json.load(f)

unique_dois_read = set(dois_from_json.keys())


for doi in unique_dois_found:
    if doi not in unique_dois_read:
        print(f"{doi} not found in the json file")
        raise AssertionError

for doi in unique_dois_read:
    if doi not in unique_dois_found:
        print(f"{doi} not referenced in the code")
        raise AssertionError

# creating a table with references in a html format, that will be included in
with open(bibliography_html_path, "w") as f:
    f.write("<table>\n")
    f.write("<tr>\n")
    f.write("<th>Title</th>\n")
    f.write("<th>Doi</th>\n")
    f.write("<th>Usages</th>\n")
    f.write("</tr>\n")
    for doi, data in dois_from_json.items():
        f.write("<tr>\n")
        f.write(f"<td>{data['title']}</td>\n")
        f.write(f"<td>{doi}</td>\n")
        f.write(f"<td>{', '.join(data['usages'])}</td>\n")
        f.write("</tr>\n")
    f.write("</table>\n")