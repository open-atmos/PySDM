import re

from setuptools import setup


def get_long_description():
    """returns contents of the pdoc landing site with pdoc links converted into URLs"""
    with open("docs/pysdm_examples_landing.md", "r", encoding="utf8") as file:
        pdoc_links = re.compile(
            r"(`)([\w\d_-]*).([\w\d_-]*)(`)", re.MULTILINE | re.UNICODE
        )
        return pdoc_links.sub(
            r'<a href="https://open-atmos.github.io/PySDM/\2/\3.html">\3</a>',
            file.read(),
        )


setup(
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
)
