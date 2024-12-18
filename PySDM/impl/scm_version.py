""" overriding defaults for setuptools_scm (see pyproject.toml) """


def scheme(_):
    """removes git hash from version string to avoid clash with PyPI rules"""
    return ""
