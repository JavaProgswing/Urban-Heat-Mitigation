from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_sidebar_is_production_only_and_hides_project_id():
    app = AppTest.from_file("dashboard/app.py", default_timeout=60).run()
    assert not app.exception
    assert [widget.label for widget in app.radio] == []
    assert [widget.label for widget in app.selectbox] == []
    text_labels = [widget.label for widget in app.text_input]
    assert text_labels == ["Search a place"]
    assert "GEE project id" not in text_labels
    assert "Analyze urban heat" in [button.label for button in app.button]
    rendered = "\n".join(str(block.value) for block in app.markdown)
    assert "AI/ML decision support" not in rendered
    assert "Earth Engine ready" not in rendered
    assert "config.yaml validated" not in rendered
    assert "runtime-dot" not in rendered
    assert "linear-gradient" not in rendered


def test_plan_copy_and_water_labels_are_unambiguous():
    source = Path("dashboard/app.py").read_text(encoding="utf-8")
    assert "Switch the map layer above" not in source
    assert 'st.markdown("**Legend** "' not in source
    assert "New water features" in source
    assert "Existing water — reference only" in source
    assert "Intervention register" in source
