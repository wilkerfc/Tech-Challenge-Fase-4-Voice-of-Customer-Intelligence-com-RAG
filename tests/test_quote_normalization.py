import pytest
from voc.pipeline import validate_generation


def test_added_terminal_punctuation_is_auditable_and_published_quote_is_literal():
    source = "A entrega está em atraso"
    result = validate_generation({"findings": [{"theme": "atraso", "citation": "E1", "quote": source + "."}], "insufficient": False}, [{"citation": "E1", "text": source}])
    finding = result["findings"][0]
    assert finding["quote"] == source
    assert finding["quote_normalization"] == "removed_terminal_punctuation"


def test_content_edits_remain_rejected_despite_terminal_punctuation():
    with pytest.raises(ValueError):
        validate_generation({"findings": [{"theme": "atraso", "citation": "E1", "quote": "A entrega atrasou 99 dias."}], "insufficient": False}, [{"citation": "E1", "text": "A entrega atrasou 2 dias"}])
