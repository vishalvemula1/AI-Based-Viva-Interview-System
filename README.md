# AI-Based Technical Viva Interview System 🎓🤖

An intelligent, automated oral technical viva examination platform built with **FastAPI** and **Generative AI**. The system conducts an adaptive technical viva on any subject selected by the candidate, dynamically generates questions, grades answers with score (out of 10) and feedback, maintains conversation history via session IDs, and compiles an end-of-viva performance report across exactly 3 questions.

---

## 🌟 Key Features

1. **Candidate & Subject Configuration**: Accepts student's name, technical subject (e.g., Python, DBMS, Operating Systems, Machine Learning), and difficulty level (*Easy*, *Medium*, *Hard*).
2. **Generative AI Question Generation**: Uses LLM API to generate Question 1 tailored to the subject and difficulty level.
3. **Adaptive Evaluation & Follow-up**:
   - Scores student responses out of **10.0**.
   - Generates constructive oral feedback highlighting strengths, inaccuracies, and nuances.
   - Generates follow-up questions dynamically adapted to the candidate's prior responses.
4. **Session State & History**: Tracks conversation history and question-answer pairs using a unique UUID session ID.
5. **Final Performance Report**:
   - Concludes after **exactly three questions**.
   - Calculates individual question scores and the **average marks**.
   - Generates an **overall assessment** and actionable **areas for improvement**.
6. **Pytest Suite**: Complete test coverage of all endpoints, workflows, validation errors, and report calculations.
7. **CI/CD with GitHub Actions**: Automated continuous integration executing pytest across Python versions on code push.
8. **Containerization**: Production-ready lightweight `Dockerfile` with healthcheck.
9. **Interactive Web UI**: Built-in responsive portal accessible directly at `http://localhost:8080/`.

---

## 🏛️ Architecture & Project Structure

```text
MLOPS EndSem Lab 2/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI automated testing
├── app/
│   ├── __init__.py
│   ├── config.py                # App & LLM configuration
│   ├── models.py                # Pydantic schemas (requests, responses, reports)
│   ├── session_store.py         # Thread-safe session & history storage
│   ├── llm_service.py           # LLM integration (Gemini / fallback engine)
│   ├── main.py                  # FastAPI application & API endpoints
│   └── templates/
│       └── index.html           # Modern interactive Viva Web UI
├── tests/
│   ├── __init__.py
│   └── test_api.py              # Comprehensive Pytest test cases
├── Dockerfile                   # Docker containerization
├── .dockerignore
├── .gitignore
├── .env.example
├── pytest.ini                   # Pytest test configuration
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.11 or 3.12
- Git
- (Optional) Docker

### 2. Setup Virtual Environment
```bash
# Clone the repository and navigate to the project directory
cd "MLOPS EndSem Lab 2"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. (Optional) Configure Gemini API Key
Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```
Add your Google Gemini API key:
```env
GEMINI_API_KEY="AIzaSy..."
```
*(Note: If no API key is set, the system automatically runs using its intelligent built-in viva heuristic engine, ensuring zero downtime and fully offline execution for testing).*

### 4. Run Application
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```
- **Interactive Web Interface**: Open [http://localhost:8080/](http://localhost:8080/)
- **Interactive API Documentation (Swagger)**: Open [http://localhost:8080/docs](http://localhost:8080/docs)
- **ReDoc Documentation**: Open [http://localhost:8080/redoc](http://localhost:8080/redoc)

---

## 🧪 Running Pytest Test Suite

Execute the test suite with verbose output:
```bash
# Run tests
pytest -v tests/
```

### Test Coverage Highlights:
- ✅ Health check verification (`GET /health`)
- ✅ Web portal rendering (`GET /`)
- ✅ Starting viva session & validation errors (`POST /api/viva/start`)
- ✅ Full 3-question evaluation lifecycle (`POST /api/viva/answer`)
- ✅ Score validation (0–10 scale) & average marks calculation
- ✅ Performance report verification (scores, overall assessment, areas for improvement)
- ✅ Prevention of extra answers after 3 questions
- ✅ Session history tracking (`GET /api/viva/{session_id}/history`)
- ✅ Report status guards (`GET /api/viva/{session_id}/report`)
- ✅ Error handling for invalid session IDs and blank answers

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interactive Web UI Portal |
| `GET` | `/health` | System health and LLM provider status |
| `POST` | `/api/viva/start` | Accepts name, subject, difficulty; returns Question 1 & `session_id` |
| `POST` | `/api/viva/answer` | Evaluates answer (score/feedback) & returns next question or final report |
| `GET` | `/api/viva/{session_id}/report` | Retrieves the final Performance Report |
| `GET` | `/api/viva/{session_id}/history` | Retrieves full question-answer conversation history |

### Example Request / Response Flow

#### 1. Start Viva:
```bash
curl -X POST "http://localhost:8080/api/viva/start" \
     -H "Content-Type: application/json" \
     -d '{"student_name": "Sarah Connor", "subject": "Operating Systems", "difficulty": "Medium"}'
```
**Response:**
```json
{
  "session_id": "9f7b49d1-8178-43d9-9ec6-8d5f30ecda51",
  "student_name": "Sarah Connor",
  "subject": "Operating Systems",
  "difficulty": "Medium",
  "question_number": 1,
  "total_questions": 3,
  "question": "What are the four necessary conditions for a deadlock to occur in an operating system, and how can deadlock prevention or detection be implemented?",
  "message": "Viva session initialized for Sarah Connor. Question 1 is ready."
}
```

#### 2. Submit Answer:
```bash
curl -X POST "http://localhost:8080/api/viva/answer" \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "9f7b49d1-8178-43d9-9ec6-8d5f30ecda51",
       "answer": "The four conditions are mutual exclusion, hold and wait, no preemption, and circular wait. Prevention works by invalidating at least one of these conditions."
     }'
```

#### 3. Final Question Response includes Performance Report:
```json
{
  "session_id": "9f7b49d1-8178-43d9-9ec6-8d5f30ecda51",
  "question_number_evaluated": 3,
  "score": 8.5,
  "feedback": "Thorough and well-explained answer!",
  "is_completed": true,
  "next_question_number": null,
  "next_question": null,
  "performance_report": {
    "session_id": "9f7b49d1-8178-43d9-9ec6-8d5f30ecda51",
    "student_name": "Sarah Connor",
    "subject": "Operating Systems",
    "difficulty": "Medium",
    "scores": [8.0, 7.5, 8.5],
    "average_marks": 8.0,
    "overall_assessment": "Commendable performance on Operating Systems...",
    "areas_for_improvement": [
      "Deepen familiarity with internal execution models",
      "Provide concrete real-world code/architecture examples"
    ]
  }
}
```

---

## 🐳 Docker Deployment

### Build the Docker Image:
```bash
docker build -t ai-viva-system .
```

### Run the Container:
```bash
docker run -p 8080:8080 -e OPENAI_API_KEY="your_key" ai-viva-system
```

Access the system at `http://localhost:8080`.

---

## 🔄 GitHub Actions CI/CD Pipeline

The repository includes `.github/workflows/ci.yml`. On every `push` or `pull_request` to `main` or `master`, the workflow:
1. Checks out the code.
2. Sets up Python 3.11 and 3.12 environments.
3. Installs dependencies from `requirements.txt`.
4. Executes the full `pytest` suite.
