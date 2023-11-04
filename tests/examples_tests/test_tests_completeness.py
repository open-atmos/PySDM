# pylint: disable=missing-module-docstring
import pathlib

import yaml

from .conftest import TEST_SUITES, findfiles, get_selected_test_paths


def test_all_cases_in_testsuites():
    """raise error, e.g., if a newly added example is not within TEST_SUITES dict"""
    tmp = findfiles(
        pathlib.Path(__file__)
        .parent.parent.parent.absolute()
        .joinpath("examples")
        .joinpath("PySDM_examples"),
        r".*\.(py|ipynb)$",
    )
    all_files = list(
        filter(
            lambda x: pathlib.Path(x).name != "__init__.py",
            tmp,
        )
    )

    selected_paths_set = set()
    for suite_name in TEST_SUITES:
        selected_paths_set.update(
            map(str, get_selected_test_paths(suite_name, all_files))
        )

    assert len(all_files) > 0
    assert selected_paths_set == set(all_files)


def test_no_cases_in_multiple_testsuites():
    """raise an error if an example is featured in multiple TEST_SUITES"""
    flattened_suites = sum(list(TEST_SUITES.values()), [])

    assert len(set(flattened_suites)) == len(flattened_suites)


def test_all_test_suites_are_on_ci():
    workflow_file_path = (
        pathlib.Path(__file__)
        .parent.parent.parent.absolute()
        .joinpath(".github")
        .joinpath("workflows")
        .joinpath("tests+artifacts+pypi.yml")
    )
    with open(workflow_file_path, "r", encoding="utf8") as workflow_file:
        d = yaml.safe_load(workflow_file)
        ci_test_suites_lst = d["jobs"]["examples"]["strategy"]["matrix"]["test-suite"]
        ci_test_suites_set = set(ci_test_suites_lst)

        assert len(ci_test_suites_set) > 0
        assert len(ci_test_suites_lst) == len(ci_test_suites_set)
        assert ci_test_suites_set == set(TEST_SUITES.keys())
