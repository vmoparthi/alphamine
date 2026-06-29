"""LLM client tests — fully offline (no network, no SDKs required)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphamine.llm import (
    MockClient,
    _ApiClient,
    _extract_json_array,
    make_client,
)


class _ScriptedClient(_ApiClient):
    """Stand-in for a real API client: returns canned replies in order, records prompts."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.model = "scripted"
        self.prompts = []

    def _complete(self, user_content):
        self.prompts.append(user_content)
        return self._replies.pop(0) if self._replies else ""


def test_mock_yields_requested_count():
    out = make_client("mock").propose("anything", n=4)
    assert len(out) == 4
    assert all("expr" in d and "rationale" in d for d in out)


def test_extract_handles_fences_and_prose():
    fenced = '```json\n[{"expr": "rank(close)", "rationale": "x"}]\n```'
    prosey = 'Sure! Here you go: [{"expr": "rank(close)"}] hope that helps'
    assert _extract_json_array(fenced)[0]["expr"] == "rank(close)"
    assert _extract_json_array(prosey)[0]["expr"] == "rank(close)"
    assert _extract_json_array("no json here") == []


def test_repair_recovers_from_unparseable_first_reply():
    good = '[{"expr": "rank(-1 * returns)", "rationale": "reversal"}]'
    client = _ScriptedClient(["sorry, I can't output JSON", good])
    out = client.propose("prompt", n=3)
    assert len(out) == 1 and out[0]["expr"] == "rank(-1 * returns)"
    # exactly one repair attempt, and it carried the stricter instruction
    assert len(client.prompts) == 2
    assert "could not be parsed" in client.prompts[1]


def test_no_repair_when_first_reply_parses():
    good = '[{"expr": "rank(close)", "rationale": "x"}]'
    client = _ScriptedClient([good, "should not be used"])
    out = client.propose("prompt")
    assert len(out) == 1
    assert len(client.prompts) == 1  # no second (repair) call


def test_returns_empty_when_both_replies_fail():
    client = _ScriptedClient(["garbage", "still garbage"])
    assert client.propose("prompt") == []
    assert len(client.prompts) == 2  # tried once, repaired once, then gave up


def test_missing_model_raises_clear_error():
    try:
        make_client("openai")  # no model in kwargs
        assert False, "should have raised"
    except ValueError as e:
        assert "model id" in str(e)


def test_unknown_provider_raises():
    try:
        make_client("nope")
        assert False, "should have raised"
    except ValueError as e:
        assert "unknown llm provider" in str(e)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all llm tests passed")
