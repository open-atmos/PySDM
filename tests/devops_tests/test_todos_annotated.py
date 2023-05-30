# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os
import pathlib
import re
import sys

import pytest
from fastcore.net import ExceptionsHTTP
from ghapi.all import GhApi, paged


# https://stackoverflow.com/questions/7012921/recursive-grep-using-python
def findfiles(path, regex):
    reg_obj = re.compile(regex)
    res = []
    for root, _, fnames in os.walk(path):
        for fname in fnames:
            if reg_obj.match(fname):
                res.append(os.path.join(root, fname))
    return res


def grep(filepath, regex):
    reg_obj = re.compile(regex)
    res = []
    with open(filepath, encoding="utf8") as f:
        for line in f:
            if reg_obj.match(line):
                res.append(line)
    return res


@pytest.fixture(
    params=findfiles(
        pathlib.Path(__file__).parent.parent.parent.absolute(),
        r".*\.(ipynb|py|txt|yml|m|jl|md)$",
    )
)
def file(request):
    return request.param


@pytest.fixture(scope="session")
def gh_issues():
    res = {}
    if "CI" not in os.environ or (
        "GITHUB_ACTIONS" in os.environ and sys.version_info.minor >= 8
    ):
        try:
            api = GhApi(owner="open-atmos", repo="PySDM")
            pages = paged(
                api.issues.list_for_repo,
                owner="open-atmos",
                repo="PySDM",
                state="all",
                per_page=100,
            )
            for page in pages:
                for item in page.items:
                    res[item.number] = item.state
        except ExceptionsHTTP[403]:
            pass
    return res


# pylint: disable=redefined-outer-name
def test_todos_annotated(file, gh_issues):
    if (
        os.path.basename(file) == "test_todos_annotated.py"
        or file.endswith("-checkpoint.ipynb")
        or ".eggs" in file
    ):
        return
    for line in grep(file, r".*TODO.*"):
        match = re.search(r"TODO #(\d+)", line)
        if match is None:
            raise AssertionError(f"TODO not annotated with issue id ({line})")
        giving_up_with_hope_other_builds_did_it = len(gh_issues) == 0
        if not giving_up_with_hope_other_builds_did_it:
            number = int(match.group(1))
            if number not in gh_issues.keys():
                raise AssertionError(f"TODO annotated with non-existent id ({line})")
            if gh_issues[number] != "open":
                raise AssertionError(f"TODO remains for a non-open issue ({line})")
