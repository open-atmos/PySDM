"""
utility routine converting "CamelCase" strings into space-separated ones (i.e., "camel case")
"""

import re

CAMEL_CASE_PATTERN = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![^A-Z])")


def camel_case_to_words(string: str):
    words = CAMEL_CASE_PATTERN.findall(string)
    words = (word if word.isupper() else word.lower() for word in words)
    return " ".join(words)
