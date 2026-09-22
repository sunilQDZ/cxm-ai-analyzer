from unittest.mock import patch, MagicMock

from services.ai_service import parse_llm_json, call_openai_completion, call_ai_completion
from services.gibberish_service import is_gibrish_comment
from services.keyword_service import extract_keywords, normalize_text
from services.rule_service import clean_for_match, similarity_score, comment_match_score, detect_category_from_db_text
from services.db_service import clear_category_cache
from services.dashboard_analysis import validate_payload_data
from schemas.dashboard import DashboardAnalyzeRequest


def test_parse_llm_json_valid():
    raw = '{"key": "value", "status": "ok"}'
    parsed = parse_llm_json(raw)
    assert parsed == {"key": "value", "status": "ok"}


def test_parse_llm_json_markdown_wrapped():
    raw = '```json\n{"drivers": [{"title": "Service"}]}\n```'
    parsed = parse_llm_json(raw)
    assert parsed == {"drivers": [{"title": "Service"}]}


def test_parse_llm_json_substring_fallback():
    raw = 'Here is your response:\n{"result": "success"}\nHope this helps!'
    parsed = parse_llm_json(raw)
    assert parsed == {"result": "success"}


def test_parse_llm_json_invalid():
    assert parse_llm_json(None) is None
    assert parse_llm_json("") is None
    assert parse_llm_json("This is plain text with no json") is None


@patch("services.ai_service.requests.post")
@patch("os.getenv", return_value="fake-openai-key")
def test_call_openai_completion_success(mock_env, mock_post):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"drivers": [{"title": "Staff Behavior", "sentiment_type": "positive"}]}'
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    result = call_openai_completion("Analyze feedback prompt")
    assert result is not None
    assert "drivers" in result
    assert result["drivers"][0]["title"] == "Staff Behavior"


def test_call_ai_completion():
    with patch("services.ai_service.call_openai_completion") as mock_call:
        mock_call.return_value = {"key_insights": ["Insight 1"]}
        res, provider = call_ai_completion("Test prompt")
        assert res == {"key_insights": ["Insight 1"]}
        assert provider == "openai"


def test_gibberish_service():
    # Gibberish text
    assert is_gibrish_comment("asdfghjklqwerty") == 1
    assert is_gibrish_comment("1234567890!!!!!") == 1
    assert is_gibrish_comment(None) == 1

    # Legitimate customer comments
    assert is_gibrish_comment("The delivery was super fast and customer support was very helpful.") == 0
    assert is_gibrish_comment("Product was damaged during shipping.") == 0


def test_keyword_service():
    norm = normalize_text("Don't worry, the service was GREAT!")
    assert "dont" in norm
    assert "great" in norm

    text = "The delivery driver was extremely polite and arrived early."
    kw = extract_keywords(text)
    assert isinstance(kw, str)
    assert len(kw) > 0


def test_rule_service():
    cleaned = clean_for_match("Delivery & Dispatch Service!")
    assert cleaned == "delivery dispatch service"

    sim = similarity_score("delivery service", "delivery service")
    assert sim == 1.0

    score = comment_match_score("The agent was very polite and helpful", "agent behavior")
    assert score > 0

    category_mapping = {
        "Delivery Experience": ["Delivery Time", "Driver Behavior"],
        "Customer Support": ["Agent Behavior", "Response Time"]
    }
    cat = detect_category_from_db_text("The delivery driver was polite", category_mapping)
    assert cat == "Delivery Experience"


def test_unmatched_voc_fallback_to_generic():
    from services.rule_service import fix_category_subcategory_from_db, is_category_relevant_to_comment

    # Dynamic taxonomy setup for client/survey
    domain_taxonomy = {
        "Customer Service": ["Agent Behavior", "Response Delay"],
        "Technical Support": ["System Crash", "Bug Resolution"],
        "Billing Services": ["Payment Processing", "Refund Status"]
    }

    # 1. Unmatched VOC should NOT force-fit into unrelated categories
    unmatched_comment = "I would like to know the weather forecast for tomorrow."
    cat, sub = fix_category_subcategory_from_db(
        category="Billing Services",
        sub_category="Payment Processing",
        category_mapping=domain_taxonomy,
        comment=unmatched_comment
    )
    assert cat == "Generic"
    assert sub == "Generic"

    # 2. Matched VOC should map correctly
    matched_comment = "The customer service agent was very helpful during my call."
    m_cat, m_sub = fix_category_subcategory_from_db(
        category="Customer Service",
        sub_category="Agent Behavior",
        category_mapping=domain_taxonomy,
        comment=matched_comment
    )
    assert m_cat == "Customer Service"
    assert m_sub == "Agent Behavior"



def test_handle_out_of_domain_generic():
    from services.rule_service import handle_out_of_domain_generic

    # Case A: Out-of-domain flagged feedback
    obs_out, rec_out = handle_out_of_domain_generic(
        comment="unrelated text",
        category="Generic",
        sub_category="Generic",
        observation="The comment does not belong to configured domain",
        recommendations="This feedback is outside operational domain"
    )
    assert "does not belong" in obs_out or "domain" in obs_out
    assert "outside" in rec_out or "domain" in rec_out

    # Case B: In-domain feedback assigned Generic category (must preserve observation)
    obs_in, rec_in = handle_out_of_domain_generic(
        comment="my loan is not approved yet",
        category="Generic",
        sub_category="Generic",
        observation="Customer expressed dissatisfaction with loan approval turnaround.",
        recommendations="Review loan application processing times."
    )
    assert "Customer expressed dissatisfaction with loan approval turnaround." in obs_in
    assert "does not belong" not in obs_in



def test_db_service_cache():
    cleared = clear_category_cache()
    assert isinstance(cleared, int)


def test_validate_payload_data_snapshot():
    req_valid = DashboardAnalyzeRequest(
        dashboard_type="snapshot",
        category="Support",
        data={"customer_responses": ["Good service"]}
    )
    is_valid, msg = validate_payload_data("snapshot", req_valid)
    assert is_valid is True

    req_invalid = DashboardAnalyzeRequest(
        dashboard_type="snapshot",
        category="Support",
        data={}
    )
    is_valid, msg = validate_payload_data("snapshot", req_invalid)
    assert is_valid is False
    assert "No survey feedback" in msg


def test_validate_payload_data_sentiment():
    req_valid = DashboardAnalyzeRequest(
        dashboard_type="sentiment_analysis",
        category="Support",
        data={"total_mentions": 10, "responses": ["Great"]}
    )
    is_valid, msg = validate_payload_data("sentiment_analysis", req_valid)
    assert is_valid is True

    req_invalid = DashboardAnalyzeRequest(
        dashboard_type="sentiment_analysis",
        category="Support",
        data={"total_mentions": 0}
    )
    is_valid, msg = validate_payload_data("sentiment_analysis", req_invalid)
    assert is_valid is False


def test_validate_payload_data_trend():
    req_valid = DashboardAnalyzeRequest(
        dashboard_type="trend_analysis",
        category="Support",
        month_1={"data": {"nps": 50}},
        month_2={"data": {"nps": 60}}
    )
    is_valid, msg = validate_payload_data("trend_analysis", req_valid)
    assert is_valid is True

    req_invalid = DashboardAnalyzeRequest(
        dashboard_type="trend_analysis",
        category="Support",
        month_1={},
        month_2={}
    )
    is_valid, msg = validate_payload_data("trend_analysis", req_invalid)
    assert is_valid is False


def test_load_categories_from_db_with_client_id_and_survey_id():
    from services.db_service import load_categories_from_db_with_status, _normalize_int_id
    assert _normalize_int_id("123") == 123
    assert _normalize_int_id(456) == 456
    assert _normalize_int_id(None) is None
    assert _normalize_int_id("invalid") is None

    mapping, status = load_categories_from_db_with_status(client_id="10", survey_id=5)
    assert isinstance(mapping, dict)
    assert "Generic" in mapping


def test_generate_insight_null_and_numeric_edge_cases():
    from services.insight_service import generate_insight
    # Integer comment_id, None comment
    res1 = generate_insight(comment=None, comment_id=12345, client_id="101", survey_id="202")
    assert res1["id"] == "12345"
    assert res1["is_gibberish"] == 1
    assert res1["category"] == "Generic"

    # Numeric comment (e.g. 99999)
    res2 = generate_insight(comment=99999, comment_id="c-999")
    assert res2["id"] == "c-999"
    assert "category" in res2

