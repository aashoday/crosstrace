
# ruff: noqa: FLY002

import pytest

from crosstrace.traceback_parser import (
    canonical_name,
    package_from_path,
    parse_issue_text,
    parse_tracebacks,
    parse_versions,
)

PLAIN = "\n".join(
    [
        "Traceback (most recent call last):",
        '  File "test.py", line 4, in <module>',
        "    result = optimize.minimize(f, x0)",
        '  File "/env/site-packages/scipy/optimize/_minimize.py", line 700, in minimize',
        "    res = _minimize_bfgs(fun, x0)",
        "          ^^^^^^^^^^^^^^^^^^^^^^^",
        '  File "/env/site-packages/numpy/linalg/linalg.py", line 100, in inv',
        "    return wrap(ainv)",
        "numpy.linalg.LinAlgError: Singular matrix",
    ]
)

CHAINED = "\n".join(
    [
        "Traceback (most recent call last):",
        '  File "a.py", line 1, in <module>',
        "KeyError: 'a'",
        "",
        "During handling of the above exception, another exception occurred:",
        "",
        "Traceback (most recent call last):",
        '  File "b.py", line 2, in <module>',
        "ValueError: bad key",
    ]
)

IPYTHON = "\n".join(
    [
        "-" * 75,
        "LinAlgError" + " " * 20 + "Traceback (most recent call last)",
        "Cell In[1], line 3",
        "      1 import numpy as np",
        "      2 a = np.ones(3)",
        "----> 3 np.linalg.inv(a)",
        "",
        "File ~/env/site-packages/numpy/linalg/linalg.py:545, in inv(a)",
        "    543 a, wrap = _makearray(a)",
        "--> 545 _assert_stacked_square(a)",
        "",
        (
            "File ~/env/site-packages/numpy/linalg/linalg.py:190, "
            "in _assert_stacked_square(*arrays)"
        ),
        "    188 raise LinAlgError('bad shape')",
        "",
        "LinAlgError: Last 2 dimensions of the array must be square",
    ]
)

NO_PACKAGE = "\n".join(
    [
        "Traceback (most recent call last):",
        '  File "<stdin>", line 1, in <module>',
        '  File "<__array_function__ internals>", line 5, in dot',
        "ValueError: shapes (2,3) and (2,3) not aligned: 3 (dim 1) != 2 (dim 0)",
    ]
)


def test_plain_traceback():
    result = parse_tracebacks(PLAIN)
    assert len(result) == 1
    tb = result[0]
    assert [f.package for f in tb.frames] == [None, "scipy", "numpy"]
    assert [f.line for f in tb.frames] == [4, 700, 100]
    assert tb.frames[0].function == "<module>"
    assert tb.frames[2].function == "inv"
    assert tb.error_type == "numpy.linalg.LinAlgError"
    assert tb.error_message == "Singular matrix"
    assert tb.packages == ("scipy", "numpy")


def test_packages_appear_once_in_frame_order():
    text = "\n".join(
        [
            "Traceback (most recent call last):",
            '  File "/e/site-packages/numpy/a.py", line 1, in f',
            '  File "/e/site-packages/numpy/b.py", line 2, in g',
            '  File "/e/site-packages/scipy/c.py", line 3, in h',
            "ValueError: x",
        ]
    )
    assert parse_tracebacks(text)[0].packages == ("numpy", "scipy")


def test_chained_tracebacks_are_separate():
    result = parse_tracebacks(CHAINED)
    assert [tb.error_type for tb in result] == ["KeyError", "ValueError"]
    assert [tb.error_message for tb in result] == ["'a'", "bad key"]


def test_ipython_traceback():
    result = parse_tracebacks(IPYTHON)
    assert len(result) == 1
    tb = result[0]
    assert [f.function for f in tb.frames] == ["inv", "_assert_stacked_square"]
    assert [f.line for f in tb.frames] == [545, 190]
    assert tb.packages == ("numpy",)
    assert tb.error_type == "LinAlgError"
    assert tb.error_message == "Last 2 dimensions of the array must be square"


def test_traceback_without_package_frames():
    tb = parse_tracebacks(NO_PACKAGE)[0]
    assert len(tb.frames) == 2
    assert tb.packages == ()
    assert tb.error_type == "ValueError"
    assert tb.error_message.startswith("shapes (2,3)")


def test_truncated_traceback_has_no_error():
    text = "\n".join(
        [
            "Traceback (most recent call last):",
            '  File "x.py", line 3, in f',
            "",
            "Any idea what is wrong?",
        ]
    )
    tb = parse_tracebacks(text)[0]
    assert len(tb.frames) == 1
    assert tb.error_type is None
    assert tb.error_message is None


def test_error_without_message():
    text = "\n".join(
        [
            "Traceback (most recent call last):",
            '  File "x.py", line 3, in f',
            "KeyboardInterrupt",
        ]
    )
    tb = parse_tracebacks(text)[0]
    assert tb.error_type == "KeyboardInterrupt"
    assert tb.error_message is None


def test_traceback_inside_code_fence():
    text = "\n".join(
        [
            "```python",
            "Traceback (most recent call last):",
            '  File "x.py", line 3, in f',
            "ValueError: x",
            "```",
        ]
    )
    result = parse_tracebacks(text)
    assert len(result) == 1
    assert result[0].error_type == "ValueError"


def test_text_without_traceback():
    text = "I get a wrong result when I call np.mean.\nThere is no error."
    assert parse_tracebacks(text) == []


def test_empty_text():
    assert parse_tracebacks("") == []
    assert parse_versions("") == {}
    result = parse_issue_text("")
    assert result.tracebacks == ()
    assert result.versions == {}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/env/lib/python3.11/site-packages/numpy/core/_methods.py", "numpy"),
        ("/usr/lib/python3/dist-packages/scipy/x.py", "scipy"),
        ("C:\\Users\\me\\site-packages\\pandas\\core\\frame.py", "pandas"),
        ("/env/site-packages/sklearn/base.py", "scikit-learn"),
        ("/env/site-packages/six.py", "six"),
        ("/env/site-packages/foo-1.0-py3.8.egg/foo/bar.py", "foo"),
        ("/usr/lib/python3.11/subprocess.py", None),
        ("test.py", None),
        ("<stdin>", None),
        ("/env/site-packages/", None),
    ],
)
def test_package_from_path(path, expected):
    assert package_from_path(path) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Scikit_Learn", "scikit-learn"),
        ("sklearn", "scikit-learn"),
        ("skimage", "scikit-image"),
        ("NumPy", "numpy"),
        ("ruamel.yaml", "ruamel-yaml"),
    ],
)
def test_canonical_name(name, expected):
    assert canonical_name(name) == expected


def test_versions_from_pip_freeze():
    text = "numpy==1.26.4\nscipy==1.11.4\nscikit-learn==1.3.2"
    assert parse_versions(text) == {
        "numpy": "1.26.4",
        "scipy": "1.11.4",
        "scikit-learn": "1.3.2",
    }


def test_versions_from_show_versions_style():
    text = "\n".join(
        [
            "INSTALLED VERSIONS",
            "------------------",
            "python           : 3.11.5.final.0",
            "pandas           : 2.1.1",
            "numpy            : 1.26.0",
            "Cython           : None",
        ]
    )
    assert parse_versions(text) == {
        "python": "3.11.5.final.0",
        "pandas": "2.1.1",
        "numpy": "1.26.0",
    }


def test_versions_from_pip_list_table():
    text = "Package    Version\n---------- -------\nnumpy      1.26.4\nastropy    5.3.4"
    assert parse_versions(text) == {"numpy": "1.26.4", "astropy": "5.3.4"}


def test_versions_use_alias_and_bullets():
    assert parse_versions("sklearn: 1.3.2") == {"scikit-learn": "1.3.2"}
    assert parse_versions("- numpy: 1.26.4") == {"numpy": "1.26.4"}


def test_first_version_wins():
    assert parse_versions("numpy==1.0.0\nnumpy==2.0.0") == {"numpy": "1.0.0"}


def test_prose_gives_no_versions():
    assert parse_versions("numpy is great and I use numpy 1.26 daily") == {}


def test_parse_issue_text_combines_both():
    result = parse_issue_text("Versions:\nnumpy==1.26.4\n\n" + PLAIN)
    assert len(result.tracebacks) == 1
    assert result.versions == {"numpy": "1.26.4"}
