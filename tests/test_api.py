import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.session_store import session_store
from app.llm_service import llm_service

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_sessions():
    """Ensure session store is cleared between tests."""
    session_store.clear()
    yield
    session_store.clear()


def test_health_check():
    """Verify health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "llm_mode" in data
    assert data["max_questions"] == 3


def test_web_ui_endpoint():
    """Verify web UI homepage renders correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Viva Examination Portal" in response.text


def test_start_viva_success():
    """Test initiating a new viva session."""
    payload = {
        "student_name": "Ada Lovelace",
        "subject": "Operating Systems",
        "difficulty": "Medium"
    }
    response = client.post("/api/viva/start", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["student_name"] == "Ada Lovelace"
    assert data["subject"] == "Operating Systems"
    assert data["difficulty"] == "Medium"
    assert data["question_number"] == 1
    assert data["total_questions"] == 3
    assert len(data["question"]) > 10


def test_start_viva_validation_errors():
    """Test validation errors on missing or empty fields."""
    # Blank name
    res1 = client.post("/api/viva/start", json={"student_name": "   ", "subject": "DBMS", "difficulty": "Easy"})
    assert res1.status_code in [400, 422]

    # Blank subject
    res2 = client.post("/api/viva/start", json={"student_name": "Alan Turing", "subject": "", "difficulty": "Easy"})
    assert res2.status_code in [400, 422]


def test_complete_three_question_viva_flow():
    """
    Test the full interactive viva flow:
    1. Start Viva -> Question 1
    2. Answer Q1 -> Score, Feedback, Question 2
    3. Answer Q2 -> Score, Feedback, Question 3
    4. Answer Q3 -> Score, Feedback, Concluded Viva & Performance Report
    5. Verify scores, average marks, overall assessment, and areas for improvement.
    """
    # 1. Start viva
    start_resp = client.post("/api/viva/start", json={
        "student_name": "Grace Hopper",
        "subject": "Python Programming",
        "difficulty": "Medium"
    })
    assert start_resp.status_code == 201
    start_data = start_resp.json()
    session_id = start_data["session_id"]
    assert start_data["question_number"] == 1

    # 2. Answer Question 1
    ans1_resp = client.post("/api/viva/answer", json={
        "session_id": session_id,
        "answer": "Python uses reference counting along with cyclic garbage collection to detect cycles of isolated objects and free memory automatically."
    })
    assert ans1_resp.status_code == 200
    ans1_data = ans1_resp.json()
    assert ans1_data["question_number_evaluated"] == 1
    assert 0.0 <= ans1_data["score"] <= 10.0
    assert len(ans1_data["feedback"]) > 5
    assert ans1_data["is_completed"] is False
    assert ans1_data["next_question_number"] == 2
    assert ans1_data["next_question"] is not None
    assert ans1_data["performance_report"] is None

    # Verify report is not accessible yet (session in progress)
    pre_rep = client.get(f"/api/viva/{session_id}/report")
    assert pre_rep.status_code == 400
    assert "Only 1 of 3 questions completed" in pre_rep.json()["detail"]

    # 3. Answer Question 2
    ans2_resp = client.post("/api/viva/answer", json={
        "session_id": session_id,
        "answer": "In production, performance bottlenecks can be addressed using asynchronous I/O, caching layers like Redis, and profiling with cProfile."
    })
    assert ans2_resp.status_code == 200
    ans2_data = ans2_resp.json()
    assert ans2_data["question_number_evaluated"] == 2
    assert 0.0 <= ans2_data["score"] <= 10.0
    assert ans2_data["is_completed"] is False
    assert ans2_data["next_question_number"] == 3
    assert ans2_data["next_question"] is not None
    assert ans2_data["performance_report"] is None

    # 4. Answer Question 3 (Final question)
    ans3_resp = client.post("/api/viva/answer", json={
        "session_id": session_id,
        "answer": "When debugging memory leaks in long-running services, I inspect circular references using the objgraph and gc modules, and write regression unit tests."
    })
    assert ans3_resp.status_code == 200
    ans3_data = ans3_resp.json()
    assert ans3_data["question_number_evaluated"] == 3
    assert 0.0 <= ans3_data["score"] <= 10.0
    assert ans3_data["is_completed"] is True
    assert ans3_data["next_question_number"] is None
    assert ans3_data["next_question"] is None

    # 5. Verify Performance Report inside answer response
    report = ans3_data["performance_report"]
    assert report is not None
    assert report["session_id"] == session_id
    assert report["student_name"] == "Grace Hopper"
    assert report["subject"] == "Python Programming"
    assert len(report["scores"]) == 3

    # Verify average marks calculation
    expected_avg = round(sum(report["scores"]) / 3, 2)
    assert report["average_marks"] == expected_avg
    assert 0.0 <= report["average_marks"] <= 10.0

    # Verify overall assessment and areas for improvement
    assert len(report["overall_assessment"]) > 10
    assert isinstance(report["areas_for_improvement"], list)
    assert len(report["areas_for_improvement"]) >= 1

    # 6. Verify dedicated Report Endpoint
    rep_resp = client.get(f"/api/viva/{session_id}/report")
    assert rep_resp.status_code == 200
    rep_data = rep_resp.json()
    assert rep_data["session_id"] == session_id
    assert rep_data["average_marks"] == expected_avg

    # 7. Submitting another answer after completion should fail (400)
    extra_ans = client.post("/api/viva/answer", json={
        "session_id": session_id,
        "answer": "Another extra answer"
    })
    assert extra_ans.status_code == 400


def test_submit_blank_answer():
    """Verify validation error when submitting a blank answer."""
    start_resp = client.post("/api/viva/start", json={
        "student_name": "Nikola Tesla",
        "subject": "Electrical Systems",
        "difficulty": "Easy"
    })
    session_id = start_resp.json()["session_id"]

    blank_ans = client.post("/api/viva/answer", json={
        "session_id": session_id,
        "answer": "   "
    })
    assert blank_ans.status_code in [400, 422]


def test_invalid_session_id():
    """Verify appropriate error on invalid or non-existent session ID."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    resp = client.post("/api/viva/answer", json={
        "session_id": fake_id,
        "answer": "My answer"
    })
    assert resp.status_code == 404

    resp_rep = client.get(f"/api/viva/{fake_id}/report")
    assert resp_rep.status_code == 404

    resp_hist = client.get(f"/api/viva/{fake_id}/history")
    assert resp_hist.status_code == 404


def test_viva_history_endpoint():
    """Verify history endpoint tracks answers and status."""
    start_resp = client.post("/api/viva/start", json={
        "student_name": "Claude Shannon",
        "subject": "Information Theory",
        "difficulty": "Hard"
    })
    session_id = start_resp.json()["session_id"]

    # Check history before any answer
    hist_resp = client.get(f"/api/viva/{session_id}/history")
    assert hist_resp.status_code == 200
    hist = hist_resp.json()
    assert hist["student_name"] == "Claude Shannon"
    assert hist["status"] == "in_progress"
    assert len(hist["evaluations"]) == 0

    # Submit answer
    client.post("/api/viva/answer", json={
        "session_id": session_id,
        "answer": "Entropy measures the average uncertainty or information content produced by an information source."
    })

    # Check history after 1 answer
    hist_resp2 = client.get(f"/api/viva/{session_id}/history")
    hist2 = hist_resp2.json()
    assert len(hist2["evaluations"]) == 1
    assert hist2["evaluations"][0]["question_number"] == 1


def test_llm_custom_mock_integration():
    """Test LLM service with mocked external API returning custom score and question."""
    with patch.object(llm_service, "generate_first_question", return_value="What is Bayes Theorem?"):
        start_resp = client.post("/api/viva/start", json={
            "student_name": "Thomas Bayes",
            "subject": "Probability",
            "difficulty": "Easy"
        })
        assert start_resp.status_code == 201
        assert start_resp.json()["question"] == "What is Bayes Theorem?"

    session_id = start_resp.json()["session_id"]

    with patch.object(llm_service, "evaluate_and_generate_next", return_value={
        "score": 9.5,
        "feedback": "Flawless explanation of conditional probability.",
        "next_question": "How is it applied in Naive Bayes classification?"
    }):
        ans_resp = client.post("/api/viva/answer", json={
            "session_id": session_id,
            "answer": "Bayes theorem describes the probability of an event based on prior knowledge of conditions that might be related to the event."
        })
        assert ans_resp.status_code == 200
        ans_data = ans_resp.json()
        assert ans_data["score"] == 9.5
        assert ans_data["feedback"] == "Flawless explanation of conditional probability."
        assert ans_data["next_question"] == "How is it applied in Naive Bayes classification?"
