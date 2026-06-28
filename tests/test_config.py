import pytest

from src.config import Config, validate_gee_project


@pytest.mark.parametrize("value", [None, "", "your-gee-project-id", "UPPER CASE",
                                   "-starts-wrong", "ends-wrong-"])
def test_invalid_gee_project_has_actionable_error(value):
    with pytest.raises(ValueError, match="config.yaml"):
        validate_gee_project(value)


def test_valid_gee_project():
    assert validate_gee_project("procurementai-472604") == "procurementai-472604"


def test_config_requires_gee_section():
    cfg = Config(raw={})
    with pytest.raises(ValueError, match="not configured"):
        _ = cfg.gee_project
