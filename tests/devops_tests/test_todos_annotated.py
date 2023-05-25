""" utilities to ensure all TO-DO comments in the code are annotated
    with an id of an open GitHub issue """

import os
import re

import pytest
from binaryornot.check import is_binary
from ghapi.all import GhApi, paged
from git.cmd import Git


def _grep(filepath, regex):
    reg_obj = re.compile(regex)
    res = []
    with open(filepath, encoding="utf8") as file_lines:
        for line in file_lines:
            if reg_obj.match(line):
                res.append(line)
    return res


@pytest.fixture(
    params=(
        path
        for path in Git(Git(".").rev_parse("--show-toplevel")).ls_files().split("\n")
        if os.path.isfile(path)
    ),
    name="git_tracked_file",
)
def _git_tracked_file(request):
    return request.param


@pytest.fixture(scope="session", name="gh_issues")
def _gh_issues():
    res = {}
    repo = os.path.basename(Git(".").rev_parse("--show-toplevel"))
    api = GhApi(owner="open-atmos", repo=repo)
    pages = paged(
        api.issues.list_for_repo,
        owner="open-atmos",
        repo=repo,
        state="all",
        per_page=100,
    )
    for page in pages:
        for item in page.items:
            res[item.number] = item.state
    return res


def test_todos_annotated(git_tracked_file, gh_issues):
    """raises assertion errors if a (TODO|FIXME) is not annotated or if the annotation
    does not point to an open issue"""
    if is_binary(git_tracked_file):
        pytest.skip("binary file")
    for line in _grep(git_tracked_file, r".*(TODO|FIXME).*"):
        if "(TODO|FIXME)" in line:
            continue
        match = re.search(r"(TODO|FIXME) #(\d+)", line)
        if match is None:
            raise AssertionError(f"(TODO|FIXME) not annotated with issue id ({line})")
        giving_up_with_hope_other_builds_did_it = len(gh_issues) == 0
        if not giving_up_with_hope_other_builds_did_it:
            number = int(match.group(2))
            if number not in gh_issues.keys():
                raise AssertionError(
                    f"(TODO|FIXME) annotated with non-existent id ({line})"
                )
            if gh_issues[number] != "open":
                raise AssertionError(
                    f"(TODO|FIXME) remains for a non-open issue ({line})"
                )
