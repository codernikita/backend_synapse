import os
import uuid
import numpy as np
from datetime import date, datetime
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, DateTime, ForeignKey, ARRAY
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# BKT Engine Imports
from bkt.engine import BKTEngine
from bkt.parameters import from_irt
from bkt.state import BKTState

# 1. Load Environment Variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is missing from your .env file!")

# 2. Database Engine & Session Setup
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ------------------------------------------------------------------
# 3. SQLAlchemy ORM Models
# ------------------------------------------------------------------

class UserDB(Base):
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default='student')
    created_at = Column(DateTime, default=datetime.utcnow)

class StudentProfileDB(Base):
    __tablename__ = "student_profiles"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    class_grade = Column(String(20), nullable=False)
    date_of_birth = Column(Date)
    school_name = Column(String(200))
    previous_class_percentage = Column(Float)
    aptitude_score = Column(Float, default=0.0)
    diagnostic_completed = Column(Boolean, default=False)
    theta_ability = Column(Float, default=0.0)
    mastery_state = Column(JSONB, default=dict)
    updated_at = Column(DateTime, default=datetime.utcnow)

class ConceptDB(Base):
    __tablename__ = "concepts"
    __table_args__ = {'extend_existing': True}

    concept_id = Column(String(50), primary_key=True)
    concept_name = Column(String(300), nullable=False)
    chapter = Column(String(100), nullable=False)
    ncert_section = Column(Float, nullable=True)
    description = Column(String, nullable=True)
    formula = Column(String(200), nullable=True)
    unit = Column(String(100), nullable=True)
    bloom_level = Column(String(50), nullable=True)
    difficulty_level = Column(Integer, nullable=True)
    concept_type = Column(String(50), nullable=True)
    misconception = Column(String, nullable=True)
    quiz_in_scope = Column(String(10), nullable=True)

class QuestionDB(Base):
    __tablename__ = "question_bank"
    __table_args__ = {'extend_existing': True}

    question_id = Column(Integer, primary_key=True, index=True)
    chapter = Column(String(100), nullable=False)
    concept_id = Column(String(50), nullable=True)
    bloom = Column(String(50), nullable=True)
    difficulty = Column(Integer, nullable=True)
    question_type = Column(String(50), nullable=True)
    question_text = Column(String, nullable=False)
    option_a = Column(String, nullable=False)
    option_b = Column(String, nullable=False)
    option_c = Column(String, nullable=False)
    option_d = Column(String, nullable=False)
    correct_option = Column(String(2), nullable=False)
    correct_reasoning = Column(String, nullable=True)
    wrong_a_confusion = Column(String, nullable=True)
    wrong_b_confusion = Column(String, nullable=True)
    wrong_c_confusion = Column(String, nullable=True)
    wrong_d_confusion = Column(String, nullable=True)

class QuizSessionDB(Base):
    __tablename__ = "quiz_sessions"
    __table_args__ = {'extend_existing': True}

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    quiz_type = Column(String(50), nullable=False)
    concept_id = Column(String(50), nullable=True)
    assigned_question_ids = Column(ARRAY(Integer), nullable=False)
    current_index = Column(Integer, default=0)
    is_completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class QuizAttemptDB(Base):
    __tablename__ = "quiz_attempts"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("quiz_sessions.session_id", ondelete="CASCADE"))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    question_id = Column(Integer, ForeignKey("question_bank.question_id"))
    selected_option = Column(String(2), nullable=True)
    is_correct = Column(Boolean, nullable=False)
    response_time_ms = Column(Integer, nullable=False)
    mastery_before = Column(Float, nullable=True)
    mastery_after = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class AptitudeQuestionDB(Base):
    __tablename__ = "aptitude_bank"
    __table_args__ = {'extend_existing': True}

    question_id = Column(Integer, primary_key=True, index=True)
    topic = Column(String(50))
    difficulty = Column(Integer)
    question_text = Column(String, nullable=False)
    option_a = Column(String, nullable=False)
    option_b = Column(String, nullable=False)
    option_c = Column(String, nullable=False)
    option_d = Column(String, nullable=False)
    correct_option = Column(String(2), nullable=False)
    correct_reasoning = Column(String, nullable=True)


# Automatically create tables in Postgres if they do not exist
Base.metadata.create_all(bind=engine)


# ------------------------------------------------------------------
# 4. Pydantic Schemas
# ------------------------------------------------------------------

class SignupReq(BaseModel):
    username: str
    password: str

class LoginReq(BaseModel):
    username: str
    password: str

class ProfileFormReq(BaseModel):
    user_id: int
    class_grade: str
    date_of_birth: str  # YYYY-MM-DD
    school_name: str
    previous_class_percentage: float

class QuizSubmitReq(BaseModel):
    user_id: int
    session_id: str
    question_id: int
    selected_option: str
    response_time_ms: int


# ------------------------------------------------------------------
# 5. FastAPI App Initialization & CORS
# ------------------------------------------------------------------

app = FastAPI(
    title="Synapse Adaptive Learning Engine",
    description="Backend API supporting Knowledge Graph concepts and Diagnostic Quizzes.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------
# 6. Endpoints & User Onboarding Flow
# ------------------------------------------------------------------

@app.get("/", response_class=FileResponse)
def read_index():
    return FileResponse("index.html")


# --- STEP 1: AUTHENTICATION ---

@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
def register(req: SignupReq, db: Session = Depends(get_db)):
    try:
        existing = db.query(UserDB).filter(UserDB.username == req.username).first()
        if existing:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        user = UserDB(username=req.username, hashed_password=req.password)
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"message": "Account created successfully", "user_id": user.id, "username": user.username}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error during registration: {str(e)}")


@app.post("/api/auth/login")
def login(req: LoginReq, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == req.username).first()
    if not user or user.hashed_password != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    profile = db.query(StudentProfileDB).filter(StudentProfileDB.user_id == user.id).first()
    has_profile = profile is not None
    
    return {
        "message": "Login successful",
        "user_id": user.id,
        "username": user.username,
        "has_completed_profile": has_profile,
        "next_step": "/api/dashboard/" + str(user.id) if has_profile else "/api/profile/details"
    }


# --- STEP 2: PROFILE DETAILS FORM ---

@app.post("/api/profile/details")
def save_profile(req: ProfileFormReq, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.id == req.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        dob = date.fromisoformat(req.date_of_birth)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    existing_profile = db.query(StudentProfileDB).filter(StudentProfileDB.user_id == req.user_id).first()
    if existing_profile:
        existing_profile.class_grade = req.class_grade
        existing_profile.date_of_birth = dob
        existing_profile.school_name = req.school_name
        existing_profile.previous_class_percentage = req.previous_class_percentage
    else:
        profile = StudentProfileDB(
            user_id=req.user_id,
            class_grade=req.class_grade,
            date_of_birth=dob,
            school_name=req.school_name,
            previous_class_percentage=req.previous_class_percentage
        )
        db.add(profile)
    
    db.commit()
    return {"status": "Profile saved successfully", "next_step": "/api/quiz/aptitude"}


@app.get("/api/quiz/aptitude")
def get_aptitude_quiz(db: Session = Depends(get_db)):
    questions = db.query(AptitudeQuestionDB).all()
    return {
        "quiz_type": "APTITUDE_TEST",
        "total_questions": len(questions),
        "next_step": "/api/quiz/diagnostic",
        "questions": [
            {
                "question_id": q.question_id,
                "topic": q.topic,
                "difficulty": q.difficulty,
                "question_text": q.question_text,
                "options": {
                    "A": q.option_a,
                    "B": q.option_b,
                    "C": q.option_c,
                    "D": q.option_d
                }
            } for q in questions
        ]
    }


# --- STEP 3 & 4: DIAGNOSTIC QUIZ ---

@app.get("/api/quiz/diagnostic")
def get_diagnostic_quiz(db: Session = Depends(get_db)):
    questions = db.query(QuestionDB).all()
    if not questions:
        raise HTTPException(status_code=404, detail="No quiz questions found in database. Did you run load_data.py?")

    return {
        "total_questions": len(questions),
        "questions": [
            {
                "question_id": q.question_id,
                "chapter": q.chapter,
                "concept_id": q.concept_id,
                "bloom": q.bloom,
                "difficulty": q.difficulty,
                "question_text": q.question_text,
                "options": {
                    "A": q.option_a,
                    "B": q.option_b,
                    "C": q.option_c,
                    "D": q.option_d
                }
            } for q in questions
        ]
    }


# --- STEP 5: FINAL DASHBOARD WITH CHAPTERS & CONCEPT LINKS ---

@app.get("/api/dashboard/{user_id}")
def get_dashboard(user_id: int, db: Session = Depends(get_db)):
    profile = db.query(StudentProfileDB).filter(StudentProfileDB.user_id == user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Student profile not found. Please complete profile form first.")

    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    concepts = db.query(ConceptDB).all()
    
    dashboard_data: Dict[str, List[Any]] = {}
    mastery_map = profile.mastery_state or {}

    for c in concepts:
        if c.chapter not in dashboard_data:
            dashboard_data[c.chapter] = []
        
        mastery = mastery_map.get(c.concept_id, 0.0)
        
        dashboard_data[c.chapter].append({
            "concept_id": c.concept_id,
            "concept_name": c.concept_name,
            "formula": c.formula or "N/A",
            "unit": c.unit or "N/A",
            "bloom_level": c.bloom_level,
            "mastery_percentage": round(float(mastery) * 100, 1),
            "quiz_link": f"/api/quiz/generate/{c.concept_id}/{user_id}"
        })

    return {
        "student_info": {
            "user_id": user_id,
            "username": user.username if user else "Student",
            "class_grade": profile.class_grade,
            "school_name": profile.school_name,
            "previous_percentage": profile.previous_class_percentage
        },
        "chapters": [
            {
                "chapter_name": ch,
                "total_concepts": len(subtopics),
                "subtopics": subtopics
            }
            for ch, subtopics in dashboard_data.items()
        ]
    }


# --- ADAPTIVE CONCEPT QUIZ GENERATION & ANSWER SUBMISSION ---

@app.get("/api/quiz/generate/{concept_id}/{user_id}")
def generate_concept_quiz(concept_id: str, user_id: int, db: Session = Depends(get_db)):
    matching_q = db.query(QuestionDB).filter(QuestionDB.concept_id.like(f"%{concept_id}%")).all()
    
    if not matching_q:
        matching_q = db.query(QuestionDB).limit(5).all()

    assigned_ids = [q.question_id for q in matching_q]
    
    new_session = QuizSessionDB(
        user_id=user_id,
        quiz_type="CONCEPT_QUIZ",
        concept_id=concept_id,
        assigned_question_ids=assigned_ids
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    first_q = matching_q[0]

    return {
        "session_id": str(new_session.session_id),
        "total_questions": len(assigned_ids),
        "question": {
            "question_id": first_q.question_id,
            "question_text": first_q.question_text,
            "options": {
                "A": first_q.option_a,
                "B": first_q.option_b,
                "C": first_q.option_c,
                "D": first_q.option_d
            }
        }
    }


@app.post("/api/quiz/submit")
@app.post("/api/quiz/submit")
def submit_answer(data: QuizSubmitReq, db: Session = Depends(get_db)):
    question = db.query(QuestionDB).filter(QuestionDB.question_id == data.question_id).first()
    profile = db.query(StudentProfileDB).filter(StudentProfileDB.user_id == data.user_id).first()

    if not question or not profile:
        raise HTTPException(status_code=404, detail="Question or Profile not found")

    is_correct = (data.selected_option.upper() == question.correct_option.upper())
    
    # Clean and extract primary concept_id (e.g. "E01, E02" -> "E01")
    raw_concept = question.concept_id or "GENERAL"
    concept_key = raw_concept.split(",")[0].strip()

    mastery_map = dict(profile.mastery_state or {})
    current_pl = float(mastery_map.get(concept_key, 0.20))

    # 1. Read IRT Student Ability & Item Difficulty
    theta = profile.theta_ability or 0.0
    b_diff = float(question.difficulty or 0.0)
    a_disc = 1.0

    # 2. Derive IRT-bridged BKT parameters
    skill_params = from_irt(
        theta=theta,
        b=b_diff,
        a=a_disc,
        p_l0=current_pl,
        skill_id=concept_key
    )

    # 3. Create BKT State
    current_state = BKTState(
        student_id=str(data.user_id),
        skill_id=concept_key,
        p_l=current_pl
    )

    # 4. Run BKT update calculation
    bkt_result = BKTEngine.update(
        state=current_state,
        is_correct=is_correct,
        params=skill_params,
        response_time_ms=data.response_time_ms,
        theta=theta,
        difficulty=b_diff,
        discrimination=a_disc
    )

    new_pl = bkt_result.new_state.p_l

    # 5. Save Quiz Attempt Log
    attempt = QuizAttemptDB(
        session_id=uuid.UUID(data.session_id),
        user_id=data.user_id,
        question_id=data.question_id,
        selected_option=data.selected_option,
        is_correct=is_correct,
        response_time_ms=data.response_time_ms,
        mastery_before=current_pl,
        mastery_after=new_pl
    )
    db.add(attempt)

    # 6. Store updated mastery in StudentProfile
    mastery_map[concept_key] = round(float(new_pl), 4)
    profile.mastery_state = mastery_map
    db.commit()

    return {
        "is_correct": is_correct,
        "correct_option": question.correct_option,
        "explanation": question.correct_reasoning,
        "flagged_as_guess": bkt_result.was_flagged_guess,
        "old_mastery_percent": round(current_pl * 100, 1),
        "new_mastery_percent": round(new_pl * 100, 1),
        "is_concept_mastered": bkt_result.new_state.is_mastered
    }