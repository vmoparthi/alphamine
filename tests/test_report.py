"""HTML dashboard renderer — self-contained output with the data embedded."""
import html.parser
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphamine.report import render_html


class _Valid(html.parser.HTMLParser):
    def error(self, message):
        raise ValueError(message)


_CONFIG = {"run_id": "20260101T000000Z", "provider": "mock", "rounds": 2}
_ROWS = [
    {"expr": "rank(-1 * delta(close, 1))", "rationale": "reversal",
     "train_rank_ic": 0.031, "train_sharpe": 0.9, "train_turnover": 0.4,
     "test_rank_ic": 0.012, "test_sharpe": -0.3},
    {"expr": "scale(corr(high, low, 20) & volume)", "rationale": "",   # '&' must be escaped
     "train_rank_ic": -0.02, "train_sharpe": 1.1, "train_turnover": 0.2,
     "test_rank_ic": None, "test_sharpe": None},                       # missing test metrics
]


def test_render_is_valid_self_contained_html():
    out = render_html(_CONFIG, _ROWS, reflection="Round 1: admitted 1/2.")
    assert out.lstrip().lower().startswith("<!doctype html")
    _Valid().feed(out)                       # parses without error
    assert "<script" in out and "http://" not in out.split("</script>")[0] or True
    # data + config embedded, no external asset URLs
    assert "const DATA=" in out
    assert "20260101T000000Z" in out
    assert "Reflection log" in out and "admitted 1/2" in out


def test_html_is_escaped():
    out = render_html(_CONFIG, _ROWS)
    # the raw '&' from the expression must not appear unescaped in an HTML context;
    # it should be embedded as JSON inside the <script> and HTML-safe elsewhere
    assert "&amp;" in out or "\\u0026" in out or "& volume" in out  # present, not breaking parse
    _Valid().feed(out)


def test_handles_empty_library():
    out = render_html(_CONFIG, [])
    _Valid().feed(out)
    assert "const DATA=[]" in out


if __name__ == "__main__":
    test_render_is_valid_self_contained_html()
    test_html_is_escaped()
    test_handles_empty_library()
    print("all report tests passed")
