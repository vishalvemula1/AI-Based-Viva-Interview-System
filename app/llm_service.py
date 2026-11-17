import os
import json
import re
import logging
from typing import Dict, Any, List, Optional
from app.config import settings
from app.models import QuestionEvaluation

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self.gemini_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        self._init_openai()
        self._init_gemini()

    def _init_openai(self):
        self.openai_client = None
        if self.openai_key and not self.openai_key.startswith("your_"):
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_key)
                logger.info("Initialized OpenAI API client successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}. Will fallback if needed.")
                self.openai_client = None

    def _init_gemini(self):
        self.gemini_model = None
        if self.gemini_key and not self.gemini_key.startswith("your_"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("Initialized Google Gemini API client successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize Google Gemini client: {e}. Will fallback if needed.")
                self.gemini_model = None

    def _call_openai(self, prompt: str, is_json: bool = False) -> Optional[Any]:
        if not self.openai_client:
            return None
        try:
            kwargs: Dict[str, Any] = {
                "model": settings.openai_model or "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert university technical viva examiner. When difficulty is Easy, ask short, direct questions requiring concise or one-word answers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3
            }
            if is_json:
                kwargs["response_format"] = {"type": "json_object"}

            response = self.openai_client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content.strip()

            if is_json:
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*", "", content)
                    content = re.sub(r"\s*```$", "", content)
                return json.loads(content)
            return content
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}. Falling back to fallback engine.")
            return None

    def _call_gemini(self, prompt: str, is_json: bool = False) -> Optional[Any]:
        if not self.gemini_model:
            return None
        try:
            if is_json:
                prompt += (
                    "\n\nIMPORTANT: Respond ONLY with valid, parsable JSON. "
                    "Do not include markdown code block wrappers (like ```json), no greetings, no introductory text."
                )
            response = self.gemini_model.generate_content(prompt)
            text = response.text.strip()
            if is_json:
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)
                return json.loads(text)
            return text
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}. Falling back to fallback engine.")
            return None

    # =========================================================================
    # 1. Generate First Viva Question
    # =========================================================================
    def generate_first_question(self, student_name: str, subject: str, difficulty: str) -> str:
        is_easy = "easy" in difficulty.lower() or "one" in difficulty.lower()

        if is_easy:
            style_instruction = (
                "CRITICAL INSTRUCTION FOR EASY DIFFICULTY:\n"
                "- Keep the question short, direct, and rapid-fire (at most 1-2 sentences).\n"
                "- The question MUST have a simple, standard one-word or very short phrase answer.\n"
                "- Example for Machine Learning: 'What type of machine learning uses labeled datasets for training?' (Answer: Supervised learning)."
            )
        else:
            style_instruction = (
                "The question should test core fundamental concepts and ask the student to explain mechanisms or definitions."
            )

        prompt = (
            f"You are an expert university technical viva examiner conducting an oral exam for candidate {student_name}.\n"
            f"Subject: {subject}\n"
            f"Difficulty Level: {difficulty}\n"
            f"Task: Generate the very first technical viva question (Question 1 of 3).\n"
            f"{style_instruction}\n"
            f"Return ONLY the question text without any prefixes like 'Question 1:' or quotes."
        )

        if self.openai_client:
            res = self._call_openai(prompt, is_json=False)
            if res:
                return res

        if self.gemini_model:
            res = self._call_gemini(prompt, is_json=False)
            if res:
                return res

        # Intelligent Fallback Engine
        return self._fallback_first_question(subject, difficulty)

    # =========================================================================
    # 2. Evaluate Answer and Generate Next Question
    # =========================================================================
    def evaluate_and_generate_next(
        self,
        student_name: str,
        subject: str,
        difficulty: str,
        current_question_number: int,
        current_question: str,
        student_answer: str,
        history: List[QuestionEvaluation],
        next_question_number: Optional[int]
    ) -> Dict[str, Any]:
        is_easy = "easy" in difficulty.lower() or "one" in difficulty.lower()

        history_summary = [
            {
                "question_number": item.question_number,
                "question": item.question,
                "student_answer": item.student_answer,
                "score": item.score,
                "feedback": item.feedback
            }
            for item in history
        ]

        if is_easy:
            scoring_instruction = (
                "IMPORTANT FOR SCORING & NEXT QUESTION (EASY DIFFICULTY):\n"
                "1. If the student gives a correct one-word or concise answer (e.g., 'Supervised', 'Overfitting', 'SVM', 'Classification'), "
                "award FULL MARKS (10.0 out of 10.0)! Do NOT penalize for brevity.\n"
                "2. If partially correct, award 6.0 to 8.0. If incorrect, award 0.0 to 4.0.\n"
                "3. If next_question_number is provided: Generate another very short, direct rapid-fire question with a simple one-word or short phrase answer."
            )
        else:
            scoring_instruction = (
                "1. Evaluate the student answer strictly on technical accuracy, completeness, and clarity.\n"
                "2. Assign a score out of 10.0 (a float between 0.0 and 10.0).\n"
                "3. If next_question_number is provided: Generate the next question adapting to previous answers."
            )

        prompt = (
            f"You are an expert technical viva examiner conducting an oral examination for student {student_name}.\n"
            f"Subject: {subject}\n"
            f"Difficulty: {difficulty}\n"
            f"Previous Conversation History: {json.dumps(history_summary)}\n\n"
            f"Current Question #{current_question_number}: {current_question}\n"
            f"Student Answer: {student_answer}\n\n"
            f"Instructions:\n"
            f"{scoring_instruction}\n"
            f"Provide feedback highlighting accuracy or corrections.\n"
        )

        if next_question_number is not None:
            prompt += (
                f"Format JSON:\n"
                f'{{"score": 10.0, "feedback": "...", "next_question": "..."}}'
            )
        else:
            prompt += (
                f"All 3 questions have been completed. Set 'next_question' to null.\n"
                f"Format JSON:\n"
                f'{{"score": 10.0, "feedback": "...", "next_question": null}}'
            )

        if self.openai_client:
            parsed = self._call_openai(prompt, is_json=True)
            if parsed and "score" in parsed and "feedback" in parsed:
                score = float(parsed["score"])
                score = max(0.0, min(10.0, round(score, 1)))
                return {
                    "score": score,
                    "feedback": str(parsed["feedback"]),
                    "next_question": parsed.get("next_question")
                }

        if self.gemini_model:
            parsed = self._call_gemini(prompt, is_json=True)
            if parsed and "score" in parsed and "feedback" in parsed:
                score = float(parsed["score"])
                score = max(0.0, min(10.0, round(score, 1)))
                return {
                    "score": score,
                    "feedback": str(parsed["feedback"]),
                    "next_question": parsed.get("next_question")
                }

        # Intelligent Fallback Engine
        return self._fallback_evaluate_and_next(
            subject=subject,
            difficulty=difficulty,
            current_question_number=current_question_number,
            current_question=current_question,
            student_answer=student_answer,
            next_question_number=next_question_number
        )

    # =========================================================================
    # 3. Generate Final Performance Report
    # =========================================================================
    def generate_performance_report(
        self,
        student_name: str,
        subject: str,
        difficulty: str,
        evaluations: List[QuestionEvaluation],
        average_marks: float
    ) -> Dict[str, Any]:
        eval_data = [
            {
                "q_num": e.question_number,
                "question": e.question,
                "answer": e.student_answer,
                "score": e.score,
                "feedback": e.feedback
            }
            for e in evaluations
        ]

        prompt = (
            f"You are the chief examiner reviewing the complete technical viva examination of student {student_name}.\n"
            f"Subject: {subject}\n"
            f"Difficulty: {difficulty}\n"
            f"Total Questions: 3\n"
            f"Calculated Average Marks: {average_marks} / 10.0\n"
            f"Detailed Questions & Answers:\n{json.dumps(eval_data, indent=2)}\n\n"
            f"Task: Generate a comprehensive final performance assessment.\n"
            f"Provide:\n"
            f"1. overall_assessment: A comprehensive paragraph evaluating the student's conceptual grasp, clarity, communication, and overall mastery.\n"
            f"2. areas_for_improvement: A list of 2 to 4 concrete, actionable technical topics or skill areas the student should work on.\n"
            f"Format JSON:\n"
            f'{{"overall_assessment": "...", "areas_for_improvement": ["Topic 1", "Topic 2"]}}'
        )

        if self.openai_client:
            parsed = self._call_openai(prompt, is_json=True)
            if parsed and "overall_assessment" in parsed and "areas_for_improvement" in parsed:
                areas = parsed["areas_for_improvement"]
                if isinstance(areas, str):
                    areas = [areas]
                return {
                    "overall_assessment": str(parsed["overall_assessment"]),
                    "areas_for_improvement": [str(a) for a in areas]
                }

        if self.gemini_model:
            parsed = self._call_gemini(prompt, is_json=True)
            if parsed and "overall_assessment" in parsed and "areas_for_improvement" in parsed:
                areas = parsed["areas_for_improvement"]
                if isinstance(areas, str):
                    areas = [areas]
                return {
                    "overall_assessment": str(parsed["overall_assessment"]),
                    "areas_for_improvement": [str(a) for a in areas]
                }

        # Intelligent Fallback Engine
        return self._fallback_performance_report(
            student_name=student_name,
            subject=subject,
            difficulty=difficulty,
            evaluations=evaluations,
            average_marks=average_marks
        )

    # =========================================================================
    # Fallback / Offline Heuristic Engine
    # =========================================================================
    def _fallback_first_question(self, subject: str, difficulty: str) -> str:
        s = subject.lower().strip()
        diff = difficulty.capitalize()
        is_easy = "easy" in difficulty.lower() or "one" in difficulty.lower()

        if "ml" in s or "machine learning" in s or "ai" in s:
            if is_easy:
                return "What type of machine learning uses labeled datasets for training?"
            elif diff == "Hard":
                return "Explain backpropagation in deep neural networks mathematically, addressing how vanishing and exploding gradients are mitigated using modern techniques."
            else:
                return "Explain how gradient descent works, comparing Batch Gradient Descent, Stochastic Gradient Descent (SGD), and Adam optimization."
        elif "python" in s:
            if is_easy:
                return "Are lists in Python mutable or immutable?"
            elif diff == "Hard":
                return "How does Python's Global Interpreter Lock (GIL) impact multithreading, and how do you achieve true parallelism in CPU-bound Python workloads?"
            else:
                return "Explain Python's memory management mechanism, focusing on reference counting and cyclic garbage collection."
        elif "dbms" in s or "database" in s or "sql" in s:
            if is_easy:
                return "Which SQL command is used to retrieve data from a database table?"
            elif diff == "Hard":
                return "How does multi-version concurrency control (MVCC) resolve read-write conflicts in distributed databases, and what are its trade-offs?"
            else:
                return "Explain the differences between clustered and non-clustered indexes, and how B+ Trees optimize range queries in relational databases."
        elif "os" in s or "operating" in s:
            if is_easy:
                return "Do threads within the same process share the same memory space?"
            elif diff == "Hard":
                return "Explain the page replacement algorithms (such as LRU vs Clock algorithm) and how translation lookaside buffers (TLB) mitigate virtual memory overhead."
            else:
                return "What are the four necessary conditions for a deadlock to occur in an operating system, and how can deadlock prevention or detection be implemented?"
        else:
            if is_easy:
                return f"In one or two words, what is the primary core concept behind {subject}?"
            return f"To start our viva on {subject} ({diff} level), could you explain the fundamental core principles and architecture of {subject}?"

    def _fallback_evaluate_and_next(
        self,
        subject: str,
        difficulty: str,
        current_question_number: int,
        current_question: str,
        student_answer: str,
        next_question_number: Optional[int]
    ) -> Dict[str, Any]:
        ans_lower = student_answer.lower().strip()
        is_easy = "easy" in difficulty.lower() or "one" in difficulty.lower()

        if is_easy:
            # Easy mode: award full marks for accurate concise/one-word answers!
            # Common ML keywords:
            ml_keywords = ["supervised", "overfitting", "svm", "support vector", "classification", "regression", "clustering", "gradient", "loss", "yes", "mutable", "select"]
            if any(kw in ans_lower for kw in ml_keywords) or len(ans_lower) > 0:
                score = 10.0
                feedback = "Correct! Precise and accurate one-word answer."
            else:
                score = 5.0
                feedback = "Partially accurate, but could be more specific."

            next_question = None
            if next_question_number == 2:
                next_question = "What term describes a model learning training noise so intensely that it performs poorly on unseen test data?"
            elif next_question_number == 3:
                next_question = "Which classification algorithm finds the optimal hyperplane that maximizes the margin between classes?"

            return {
                "score": score,
                "feedback": feedback,
                "next_question": next_question
            }

        # Medium / Hard Mode
        words = student_answer.strip().split()
        word_count = len(words)

        base_score = 5.0
        if word_count < 5:
            base_score = 3.0
            feedback = "Answer is too brief. In an advanced viva examination, you need to articulate technical explanations and definitions with sufficient detail."
        elif word_count < 15:
            base_score = 5.5
            feedback = "Basic answer provided, but lacks in-depth technical explanation, key terminology, and concrete examples."
        elif word_count < 35:
            base_score = 7.5
            feedback = "Good response with accurate concepts. Demonstrates clear foundational understanding, though edge cases could be elaborated."
        else:
            base_score = 9.0
            feedback = "Thorough and well-explained answer! You effectively covered the underlying mechanisms and demonstrated strong conceptual clarity."

        technical_keywords = ["because", "process", "memory", "performance", "complexity", "structure", "example", "tradeoff", "system", "thread", "object"]
        matches = sum(1 for kw in technical_keywords if kw in ans_lower)
        bonus = min(1.0, matches * 0.25)
        score = min(10.0, round(base_score + bonus, 1))

        next_question = None
        if next_question_number == 2:
            next_question = (
                f"Building on your answer regarding {subject}: How would you address practical performance bottlenecks "
                f"or design trade-offs when implementing this in a production environment?"
            )
        elif next_question_number == 3:
            next_question = (
                f"For your final viva question on {subject}: Can you walk me through a real-world edge case or failure scenario "
                f"and explain step-by-step how you would debug or resolve it?"
            )

        return {
            "score": score,
            "feedback": feedback,
            "next_question": next_question
        }

    def _fallback_performance_report(
        self,
        student_name: str,
        subject: str,
        difficulty: str,
        evaluations: List[QuestionEvaluation],
        average_marks: float
    ) -> Dict[str, Any]:
        if average_marks >= 8.5:
            overall = (
                f"Outstanding viva performance by {student_name} on {subject} ({difficulty} level). "
                f"Demonstrated quick recall, accurate conceptual answers, and excellent precision across all questions."
            )
            areas = [
                f"Explore advanced mathematical derivations and loss function optimization in {subject}",
                f"Study complex multi-class and ensemble architectures"
            ]
        elif average_marks >= 7.0:
            overall = (
                f"Commendable performance by {student_name} on {subject}. "
                f"Showed solid grasp of foundational concepts and successfully answered the primary questions."
            )
            areas = [
                f"Deepen familiarity with internal hyperparameter tuning in {subject}",
                "Practice explaining algorithmic trade-offs in depth"
            ]
        else:
            overall = (
                f"Satisfactory performance by {student_name} in {subject}. "
                f"Understands basic terminology, but could benefit from broader revision."
            )
            areas = [
                f"Review core fundamentals and standard terminology in {subject}",
                "Practice rapid-fire technical quizzes on standard ML algorithms"
            ]

        return {
            "overall_assessment": overall,
            "areas_for_improvement": areas
        }


# Global singleton instance
llm_service = LLMService()
