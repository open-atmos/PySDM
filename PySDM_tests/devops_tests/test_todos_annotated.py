import pytest
import os
import re
import sys
from ghapi.all import GhApi, paged, github_token


# https://stackoverflow.com/questions/7012921/recursive-grep-using-python
def findfiles(path, regex):
    regObj = re.compile(regex)
    res = []
    for root, dirs, fnames in os.walk(path):
        for fname in fnames:
            if regObj.match(fname):
                res.append(os.path.join(root, fname))
    return res


def grep(filepath, regex):
    regObj = re.compile(regex)
    res = []
    with open(filepath, encoding="utf8") as f:
        for line in f:
            if regObj.match(line):
                res.append(line)
    return res


@pytest.fixture(params=findfiles('../..', r'.*\.(ipynb|py|txt|yml|m|jl|md)$'))
def file(request):
    return request.param


@pytest.fixture(scope='session')
def gh_issues():
    args = {'owner': 'atmos-cloud-sim-uj', 'repo': 'PySDM'}
    if 'CI' in os.environ:
        args['token'] = github_token()
    api = GhApi(**args)
    pages = paged(api.issues.list_for_repo, owner='atmos-cloud-sim-uj', repo='PySDM', state='all')
    all = {}
    for page in pages:
        for item in page.items:
            all[item.number] = item.state
    return all


@pytest.mark.skipif("'CI' in os.environ and ('GITHUB_ACTIONS' not in os.environ or sys.version_info.minor < 8)")
def test_todos_annotated(file, gh_issues):
    if os.path.basename(file) == 'test_todos_annotated.py':
        return
    for line in grep(file, r'.*TODO.*'):
        match = re.search(r'TODO #(\d+)', line)
        if match is None:
            raise Exception(f"TODO not annotated with issue id ({line})")
        number = int(match.group(1))
        if number not in gh_issues.keys():
            raise Exception(f"TODO annotated with non-existent id ({line})")
        if gh_issues[number] != 'open':
            raise Exception(f"TODO remains for a non-open issue ({line})")
