import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# 1. Load Concepts
print("⏳ Loading Knowledge Graph Concepts...")
df_kg = pd.read_excel("Synapse_KG_Concepts.xlsx", sheet_name="02_KG_Concepts", header=1)
df_kg_clean = df_kg[[
    'concept_id', 'concept_name', 'chapter', 'ncert_section', 
    'description', 'formula', 'unit', 'bloom_level', 
    'difficulty_level', 'concept_type', 'misconception', 'quiz_in_scope'
]]
df_kg_clean.to_sql('concepts', engine, if_exists='append', index=False)
print("✅ Loaded Concepts!")

# 2. Load Diagnostic Science Quiz
print("⏳ Loading Diagnostic Science Quiz...")
df_quiz = pd.read_excel("Synapse_Quiz_30Q.xlsx", sheet_name="Quiz_Questions", header=1)
df_quiz_clean = pd.DataFrame({
    'chapter': df_quiz['Chapter'],
    'concept_id': df_quiz['concept_id'],
    'bloom': df_quiz['Bloom'],
    'difficulty': df_quiz['Difficulty'],
    'question_type': df_quiz['Type'],
    'question_text': df_quiz['Question Stem'],
    'option_a': df_quiz['Option A'],
    'option_b': df_quiz['Option B'],
    'option_c': df_quiz['Option C'],
    'option_d': df_quiz['Option D'],
    'correct_option': df_quiz['Correct Answer'],
    'correct_reasoning': df_quiz['Correct Reasoning'],
    'wrong_a_confusion': df_quiz['Wrong A — Confusion / Weak Topic'],
    'wrong_b_confusion': df_quiz['Wrong B — Confusion / Weak Topic'],
    'wrong_c_confusion': df_quiz['Wrong C — Confusion / Weak Topic'],
    'wrong_d_confusion': df_quiz['Wrong D — Confusion / Weak Topic']
})
df_quiz_clean.to_sql('question_bank', engine, if_exists='append', index=False)
print("✅ Loaded Science Quiz!")

# 3. Load Aptitude Questions
print("⏳ Loading Aptitude Test Questions...")
df_apt = pd.read_excel("Synapse_Aptitude_10Q.xlsx", sheet_name="Aptitude_Questions")
df_apt_clean = pd.DataFrame({
    'topic': df_apt['Topic'],
    'difficulty': df_apt['Difficulty'],
    'question_text': df_apt['Question Stem'],
    'option_a': df_apt['Option A'],
    'option_b': df_apt['Option B'],
    'option_c': df_apt['Option C'],
    'option_d': df_apt['Option D'],
    'correct_option': df_apt['Correct Answer'],
    'correct_reasoning': df_apt['Correct Reasoning']
})
df_apt_clean.to_sql('aptitude_bank', engine, if_exists='append', index=False)
print("✅ Loaded Aptitude Questions!")