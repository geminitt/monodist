import pytest

from monodist import report


def test_render_fills_placeholders_and_leaves_latex_alone(monkeypatch):
    monkeypatch.setattr(report, "values", lambda: {"x.mean": "8.1%", "l": "WRONG", "H": "WRONG", "cam": "WRONG"})
    monkeypatch.setattr(report, "tables", lambda: {"t": "| a |"})
    text = r"error {x.mean}; $\frac{l}{2z} \cdot \frac{2y - H}{H}$, $h_\mathrm{cam}$, {table:t}"
    assert report.render(text) == r"error 8.1%; $\frac{l}{2z} \cdot \frac{2y - H}{H}$, $h_\mathrm{cam}$, | a |"


def test_render_fails_on_an_unknown_placeholder(monkeypatch):
    monkeypatch.setattr(report, "values", lambda: {})
    monkeypatch.setattr(report, "tables", lambda: {})
    with pytest.raises(KeyError):
        report.render("value {missing.key}")
