import exdown
import pathlib


test_readme = exdown.pytests(
    pathlib.Path(__file__).parent.parent.parent.joinpath("README.md").absolute(),
    syntax_filter="Python",
    encoding='utf8'
)
