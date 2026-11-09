import pytest


@pytest.mark.slow
def test_app_renders():
    st = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("dashboard/app.py", default_timeout=1800)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 5
