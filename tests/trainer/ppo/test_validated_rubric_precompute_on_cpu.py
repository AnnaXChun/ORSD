from examples.token_credit_grpo.precompute_validated_rubrics import (
    parse_judge_response,
    split_rubric_items,
)


def test_split_rubric_items_deduplicates_numbered_criteria():
    rubric = "1. Check the sign constraint.\n2. Reduce the expression exactly.\n3. Check the sign constraint."
    assert split_rubric_items(rubric) == [
        "Check the sign constraint.",
        "Reduce the expression exactly.",
    ]


def test_parse_judge_response_accepts_only_all_hard_constraints():
    candidates = ["Use the defining constraint.", "The answer is 42."]
    response = """
    {
      "items": [
        {"index": 1, "correct": true, "relevant": true, "actionable": true,
         "no_answer_leak": true, "self_contained": true, "score": 0.9,
         "reason": "valid"},
        {"index": 2, "correct": true, "relevant": true, "actionable": true,
         "no_answer_leak": false, "self_contained": true, "score": 0.95,
         "reason": "leaks the answer"}
      ]
    }
    """
    accepted, reasons = parse_judge_response(response, candidates, threshold=0.75)
    assert accepted == [(candidates[0], 0.9)]
    assert reasons == ["2: leaks the answer"]


def test_parse_judge_response_rejects_missing_candidate_result():
    candidates = ["First valid step.", "Second unreviewed step."]
    response = """
    {"items": [{"index": 1, "correct": true, "relevant": true,
    "actionable": true, "no_answer_leak": true, "self_contained": true,
    "score": 0.8, "reason": "valid"}]}
    """
    accepted, reasons = parse_judge_response(response, candidates, threshold=0.75)
    assert accepted == [(candidates[0], 0.8)]
    assert reasons == ["2: missing or malformed validation result"]
