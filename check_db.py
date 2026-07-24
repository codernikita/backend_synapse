# check_db.py
from app import SessionLocal, UserDB, StudentProfileDB, QuizAttemptDB, QuizSessionDB

db = SessionLocal()

print("--- 1. USERS TABLE ---")
users = db.query(UserDB).all()
for u in users:
    print(f"ID: {u.id} | Username: {u.username} | Role: {u.role}")

print("\n--- 2. STUDENT PROFILES (IRT & BKT State) ---")
profiles = db.query(StudentProfileDB).all()
for p in profiles:
    print(f"User ID: {p.user_id} | Class: {p.class_grade} | Percentage: {p.previous_class_percentage}% | Theta: {p.theta_ability}")
    print(f"Mastery State JSON: {p.mastery_state}\n")

print("--- 3. RECENT QUIZ ATTEMPTS (BKT Updates) ---")
attempts = db.query(QuizAttemptDB).order_by(QuizAttemptDB.timestamp.desc()).limit(10).all()
for a in attempts:
    print(f"Attempt ID: {a.id} | User ID: {a.user_id} | Q_ID: {a.question_id} | Correct: {a.is_correct} | Time: {a.response_time_ms}ms | Before: {a.mastery_before} -> After: {a.mastery_after}")

db.close()