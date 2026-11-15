import pytest

from crosstrace.projects import load_projects


def test_default_list_is_valid():
    projects = load_projects()
    assert len(projects) > 0
    assert any(p.role == "upstream" for p in projects)
    assert any(p.role == "downstream" for p in projects)


def test_bad_role_is_rejected(tmp_path):
    bad = tmp_path / "projects.toml"
    bad.write_text('[[project]]\nrepo = "a/b"\nrole = "middle"\n')
    with pytest.raises(ValueError):
        load_projects(bad)


def test_duplicate_is_rejected(tmp_path):
    bad = tmp_path / "projects.toml"
    text = '[[project]]\nrepo = "a/b"\nrole = "upstream"\n'
    bad.write_text(text * 2)
    with pytest.raises(ValueError):
        load_projects(bad)