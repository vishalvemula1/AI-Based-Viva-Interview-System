import os
from typing import Optional
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import (
    VivaStartRequest,
    VivaStartResponse,
    AnswerSubmitRequest,
    AnswerSubmitResponse,
    QuestionEvaluation,
    PerformanceReport,
    VivaHistoryResponse
)
from app.session_store import session_store
from app.llm_service import llm_service

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Automated Technical Viva Examination System powered by FastAPI and Generative AI"
)

# Enable CORS for frontend clients or API consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Template engine for interactive web UI
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir) if os.path.exists(templates_dir) else None


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint for Docker and monitoring."""
    llm_mode = "Live Gemini API" if llm_service.gemini_model else "Intelligent Viva Engine"
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "llm_mode": llm_mode,
        "max_questions": settings.max_questions
    }


@app.post(
    "/api/viva/start",
    response_model=VivaStartResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Viva Interview"],
    summary="Start a new technical viva examination"
)
def start_viva(req: VivaStartRequest):
    """
    Accepts student's name, subject, and difficulty level.
    Uses LLM API to generate the first viva question and initiates a session.
    """
    if not req.student_name.strip():
        raise HTTPException(status_code=400, detail="Student name cannot be empty.")
    if not req.subject.strip():
        raise HTTPException(status_code=400, detail="Subject cannot be empty.")

    # 1. Generate Question 1 via LLM
    first_question = llm_service.generate_first_question(
        student_name=req.student_name.strip(),
        subject=req.subject.strip(),
        difficulty=req.difficulty.strip()
    )

    # 2. Initialize Session
    session = session_store.create_session(
        student_name=req.student_name.strip(),
        subject=req.subject.strip(),
        difficulty=req.difficulty.strip(),
        first_question=first_question
    )

    return VivaStartResponse(
        session_id=session.session_id,
        student_name=session.student_name,
        subject=session.subject,
        difficulty=session.difficulty,
        question_number=1,
        total_questions=settings.max_questions,
        question=first_question,
        message=f"Viva session initialized for {session.student_name}. Question 1 is ready."
    )


@app.post(
    "/api/viva/answer",
    response_model=AnswerSubmitResponse,
    tags=["Viva Interview"],
    summary="Submit student response, evaluate score and feedback, and generate next question or final report"
)
def submit_answer(req: AnswerSubmitRequest):
    """
    Submits student's response for the current question:
    1. Evaluates answer (score out of 10 and constructive feedback).
    2. Maintains conversation history in the session.
    3. If question < 3: generates next question adaptively.
    4. If question == 3: concludes viva and compiles the final Performance Report.
    """
    session = session_store.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Viva session '{req.session_id}' not found.")

    if session.is_completed:
        raise HTTPException(
            status_code=400,
            detail="Viva session has already completed all 3 questions. Please view the final report."
        )

    current_q_num = session.current_question_number
    current_q_text = session.current_question

    if not req.answer.strip():
        raise HTTPException(status_code=400, detail="Answer cannot be blank.")

    # Determine if next question is needed (Total = 3 questions)
    next_q_num = current_q_num + 1 if current_q_num < settings.max_questions else None

    # Evaluate via LLM and optionally generate next question
    eval_result = llm_service.evaluate_and_generate_next(
        student_name=session.student_name,
        subject=session.subject,
        difficulty=session.difficulty,
        current_question_number=current_q_num,
        current_question=current_q_text,
        student_answer=req.answer.strip(),
        history=session.evaluations,
        next_question_number=next_q_num
    )

    evaluation = QuestionEvaluation(
        question_number=current_q_num,
        question=current_q_text,
        student_answer=req.answer.strip(),
        score=eval_result["score"],
        feedback=eval_result["feedback"]
    )

    # Record evaluation and next question
    next_question_text = eval_result.get("next_question")
    session_store.record_answer_and_evaluation(
        session_id=session.session_id,
        evaluation=evaluation,
        next_question=next_question_text
    )

    # Check if viva completed exactly 3 questions
    performance_report: Optional[PerformanceReport] = None
    if len(session.evaluations) >= settings.max_questions:
        scores = [e.score for e in session.evaluations]
        avg_marks = round(sum(scores) / len(scores), 2)

        # Generate comprehensive overall assessment and areas for improvement
        rep_llm = llm_service.generate_performance_report(
            student_name=session.student_name,
            subject=session.subject,
            difficulty=session.difficulty,
            evaluations=session.evaluations,
            average_marks=avg_marks
        )

        performance_report = PerformanceReport(
            session_id=session.session_id,
            student_name=session.student_name,
            subject=session.subject,
            difficulty=session.difficulty,
            scores=scores,
            average_marks=avg_marks,
            overall_assessment=rep_llm["overall_assessment"],
            areas_for_improvement=rep_llm["areas_for_improvement"],
            evaluations=session.evaluations
        )

        session_store.attach_performance_report(session.session_id, performance_report)

    return AnswerSubmitResponse(
        session_id=session.session_id,
        question_number_evaluated=current_q_num,
        score=evaluation.score,
        feedback=evaluation.feedback,
        is_completed=session.is_completed,
        next_question_number=next_q_num,
        next_question=next_question_text,
        performance_report=performance_report
    )


@app.get(
    "/api/viva/{session_id}/report",
    response_model=PerformanceReport,
    tags=["Viva Interview"],
    summary="Get final performance report"
)
def get_performance_report(session_id: str):
    """Retrieves the final performance report containing scores, average marks, overall assessment, and improvement areas."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Viva session '{session_id}' not found.")

    if not session.is_completed or not session.performance_report:
        completed_count = len(session.evaluations)
        raise HTTPException(
            status_code=400,
            detail=f"Performance report not ready. Only {completed_count} of 3 questions completed."
        )

    return session.performance_report


@app.get(
    "/api/viva/{session_id}/history",
    response_model=VivaHistoryResponse,
    tags=["Viva Interview"],
    summary="Get full conversation history and evaluation status"
)
def get_viva_history(session_id: str):
    """Retrieves conversation history, current question, and all past evaluations."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Viva session '{session_id}' not found.")

    return VivaHistoryResponse(
        session_id=session.session_id,
        student_name=session.student_name,
        subject=session.subject,
        difficulty=session.difficulty,
        status=session.status,
        current_question_number=session.current_question_number,
        current_question=session.current_question,
        evaluations=session.evaluations,
        performance_report=session.performance_report
    )


@app.get("/", response_class=HTMLResponse, tags=["Web UI"], include_in_schema=False)
def get_web_ui(request: Request):
    """Serves the interactive web interface for students and examiners."""
    if templates and os.path.exists(os.path.join(templates_dir, "index.html")):
        return templates.TemplateResponse(request=request, name="index.html")
    return HTMLResponse("<h2>AI-Based Viva Interview System is Running! Visit <a href='/docs'>/docs</a> for Swagger API documentation.</h2>")
