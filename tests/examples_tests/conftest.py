# pylint: disable=missing-module-docstring
import os
import pathlib
import re
import ast
import importlib
import inspect
import pkgutil


from .. import smoke_tests


class NotebookVarExtractor(ast.NodeVisitor):
    def __init__(self):
        self.paths = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "notebook_vars":
            for arg in node.args:
                self._maybe_add_path_expr(arg)
            for kw in node.keywords:
                self._maybe_add_path_expr(kw.value)
        self.generic_visit(node)

    def _maybe_add_path_expr(self, expr):
        """Extract a path-like expression string from AST."""
        expr_str = ast.unparse(expr)
        if "Path(" in expr_str:
            self.paths.append(expr_str)


def extract_path_expressions_from_module(mod):
    try:
        source = inspect.getsource(mod)
    except (OSError, TypeError):
        return []

    tree = ast.parse(source)
    extractor = NotebookVarExtractor()
    extractor.visit(tree)

    return extractor.paths


def evaluate_path_expr(expr_str, module_globals):
    """Evaluate the path expression using the module's globals."""
    local_ctx = {"Path": pathlib.Path}
    value = eval(expr_str, {**module_globals, **local_ctx})  # pylint: disable=eval-used
    return value.resolve() if isinstance(value, pathlib.Path) else None


def iter_submodule_names(module):
    """Yield names of all submodules recursively without importing."""
    if not hasattr(module, "__path__"):
        return
    for _, name, _ in pkgutil.walk_packages(module.__path__, module.__name__ + "."):
        yield name


def find_modules_using_notebook_vars(module):
    names = []
    for name in iter_submodule_names(module):
        filepath = importlib.util.find_spec(name).origin
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        if "notebook_vars" in source:
            names.append(name)
    return names


SMOKE_TEST_COVERED_PATHS = []
for mod_name in find_modules_using_notebook_vars(smoke_tests):
    submodule = importlib.import_module(mod_name)
    exprs = extract_path_expressions_from_module(submodule)
    for path_expr in exprs:
        submodule_path = evaluate_path_expr(path_expr, vars(submodule))
        if submodule_path:
            SMOKE_TEST_COVERED_PATHS.append(submodule_path)


# https://stackoverflow.com/questions/7012921/recursive-grep-using-python
def findfiles(path, regex):
    reg_obj = re.compile(regex)
    res = []
    for root, _, fnames in os.walk(path):
        for fname in fnames:
            if reg_obj.match(fname):
                res.append(os.path.join(root, fname))
    return res


TEST_SUITES = {
    "isotopes_chemistry_extraterrestrial": [
        "Bolot_et_al_2013",
        "Merlivat_and_Nief_1967",
        "Van_Hook_1968",
        "Pierchala_et_al_2022",
        "Gedzelman_and_Arnold_1994",
        "Graf_et_al_2019",
        "Lamb_et_al_2017",
        "Miyake_et_al_1968",
        "Rozanski_and_Sonntag_1982",
        "Bolin_1958",
        "Stewart_1975",
        "Kinzer_And_Gunn_1951",
        "Pruppacher_and_Rasmussen_1979",
        "Fisher_1991",
        "Jouzel_and_Merlivat_1984",
        "Jaruga_and_Pawlowska_2018",
        "Kreidenweis_et_al_2003",
        "Toon_et_al_1980",
    ],
    "condensation_a": [
        "Lowe_et_al_2019",
        "Singer_Ward",
        "Rogers_1975",
    ],
    "condensation_b": [
        "Abdul_Razzak_Ghan_2000",
        "Arabas_and_Shima_2017",
        "Pyrcel",
        "Yang_et_al_2018",
    ],
    "condensation_c": [
        "Grabowski_and_Pawlowska_2023",
        "Jensen_and_Nugent_2017",
        "Abade_and_Albuquerque_2024",
    ],
    "coagulation_freezing": [
        "Berry_1967",
        "Shima_et_al_2009",
        "Alpert_and_Knopf_2016",
        "Ervens_and_Feingold_2012",
        "Niedermeier_et_al_2014",
        "Spichtinger_et_al_2023",
        "Ware_et_al_2025",
        "Matsushima_et_al_2023",
    ],
    "multi-process_a": [
        "Arabas_et_al_2015",
        "_HOWTOs",
        "Strzabala_2025_BEng",
    ],
    "multi-process_b": [
        "Arabas_et_al_2025",
    ],
    "multi-process_c_breakup": [
        "Bartman_2020_MasterThesis",
        "Bieli_et_al_2022",
        "deJong_Mackay_et_al_2023",
        "Srivastava_1982",
    ],
    "multi-process_d": [
        "Bartman_et_al_2021",
    ],
    "multi-process_e": [
        "deJong_Azimi",
        "Bulenok_2023_MasterThesis",
        "Morrison_and_Grabowski_2007",
        "Shipway_and_Hill_2012",
        "seeding",
        "utils",
        "Zaba_et_al",
        "Gonfiantini_1986",
    ],
}


def get_selected_test_paths(suite_name, paths):
    if suite_name is None:
        return paths

    cases = TEST_SUITES[suite_name]

    result = []
    for path in paths:
        for case in cases:
            path = pathlib.Path(path)
            if case in path.parts:
                result.append(path)

    return result


def pytest_addoption(parser):
    parser.addoption("--suite", action="append")


def pytest_generate_tests(metafunc):
    suite_args = metafunc.config.option.suite or []

    # Support both --suite name1 --suite name2 and --suite name1,name2
    suite_names = []
    for arg in suite_args:
        suite_names.extend([s.strip() for s in arg.split(",") if s.strip()])

    pysdm_examples_abs_path = (
        pathlib.Path(__file__)
        .parent.parent.parent.absolute()
        .joinpath("examples")
        .joinpath("PySDM_examples")
    )
    if "notebook_filename" in metafunc.fixturenames:
        notebook_paths = [
            path
            for path in findfiles(pysdm_examples_abs_path, r".*\.ipynb$")
            if ".ipynb_checkpoints" not in str(path)
        ]
        selected_paths = set()
        for suite_name in suite_names:
            selected_paths.update(
                set(get_selected_test_paths(suite_name, notebook_paths))
            )

        # Remove duplicates and any smoke-tested paths
        selected_paths = selected_paths - set(SMOKE_TEST_COVERED_PATHS)
        metafunc.parametrize(
            "notebook_filename",
            selected_paths,
            ids=[str(path) for path in selected_paths],
        )

    if "example_filename" in metafunc.fixturenames:
        examples_paths = findfiles(
            pysdm_examples_abs_path,
            r".*\.(py)$",
        )
        selected_paths = set()
        for suite_name in suite_names:
            selected_paths.update(
                set(get_selected_test_paths(suite_name, examples_paths))
            )
        metafunc.parametrize(
            "example_filename",
            selected_paths,
            ids=[str(path) for path in selected_paths],
        )
