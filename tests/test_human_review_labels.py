import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_human_review import validate_labels


def test_empty_labels_are_not_human_review(tmp_path):
    path=tmp_path/"labels.jsonl"
    path.write_text(json.dumps({"item_id":"x","reviewer_id":"person","task":"loop_assessment","decision":None})+"\n")
    with pytest.raises(ValueError,match="missing/invalid"):
        validate_labels({"x":{"task":"loop_assessment"}},path)


def test_valid_labels_preserve_uncertainty(tmp_path):
    path=tmp_path/"labels.jsonl"
    path.write_text(json.dumps({"item_id":"x","reviewer_id":"person","task":"loop_assessment","decision":"uncertain"})+"\n")
    name,rows=validate_labels({"x":{"task":"loop_assessment"}},path)
    assert name=="person" and rows["x"]["decision"]=="uncertain"
