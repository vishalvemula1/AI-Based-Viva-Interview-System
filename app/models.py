from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class DifficultyLevel(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class VivaStartRequest(BaseModel):
    student_name: str = Field(..., min_length=1, max_length=100, description="Full name of the student")
    subject: str = Field(..., min_length=1, max_length=100, description="Technical subject for the viva (e.g., Python, DBMS, Operating Systems, Machine Learning)")
    difficulty: str = Field(default="Medium", description="Difficulty level: Easy, Medium, or Hard")


class VivaStartResponse(BaseModel):
    session_id: str = Field(..., description="Unique session ID to track viva interview history")
    student_name: str
    subject: str
    difficulty: str
    question_number: int = Field(default=1, description="Current question number")
    total_questions: int = Field(default=3, description="Total questions in this viva examination")
    question: str = Field(..., description="First AI-generated viva question")
    message: str = Field(default="Viva session started successfully. Please submit your answer.")


class AnswerSubmitRequest(BaseModel):
    session_id: str = Field(..., description="Unique session ID for the active viva")
    answer: str = Field(..., min_length=1, description="Student's response to the current viva question")


class QuestionEvaluation(BaseModel):
    question_number: int
    question: str
    student_answer: str
    score: float = Field(..., ge=0.0, le=10.0, description="Score assigned out of 10")
    feedback: str = Field(..., description="Constructive feedback evaluating the answer")


class PerformanceReport(BaseModel):
    session_id: str
    student_name: str
    subject: str
    difficulty: str
    scores: List[float] = Field(..., description="List of scores obtained for each question (out of 10)")
    average_marks: float = Field(..., ge=0.0, le=10.0, description="Average marks across all 3 questions")
    overall_assessment: str = Field(..., description="Comprehensive qualitative assessment of student performance")
    areas_for_improvement: List[str] = Field(..., description="Actionable technical areas where the student can improve")
    evaluations: List[QuestionEvaluation] = Field(..., description="Detailed per-question breakdown")


class AnswerSubmitResponse(BaseModel):
    session_id: str
    question_number_evaluated: int = Field(..., description="The question number that was just evaluated")
    score: float = Field(..., ge=0.0, le=10.0, description="Score out of 10 for the submitted answer")
    feedback: str = Field(..., description="Evaluator feedback on the submitted answer")
    is_completed: bool = Field(..., description="True if all 3 viva questions have been completed")
    next_question_number: Optional[int] = Field(None, description="Next question number (e.g., 2 or 3), or None if completed")
    next_question: Optional[str] = Field(None, description="Next viva question generated based on previous answers, or None if completed")
    performance_report: Optional[PerformanceReport] = Field(None, description="Final viva performance report generated upon completing 3 questions")


class VivaHistoryResponse(BaseModel):
    session_id: str
    student_name: str
    subject: str
    difficulty: str
    status: str  # "in_progress" or "completed"
    current_question_number: int
    current_question: Optional[str] = None
    evaluations: List[QuestionEvaluation]
    performance_report: Optional[PerformanceReport] = None
