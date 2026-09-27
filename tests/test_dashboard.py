from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_dashboard_and_controls():
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py").run(timeout=30)
    assert not app.exception
    app.selectbox[0].select("circulation").run(timeout=30)
    assert not app.exception
    app.slider[2].set_value(.2).run(timeout=30)
    assert not app.exception
