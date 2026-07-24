# 🧠 Synapse Assessment Engine

Quickstart guide to set up and run the portal locally.

---

## ⚡ Quickstart (4 Steps)

### 1. Set Up Virtual Environment
Open your terminal in the project folder and run:

```bash
python -m venv venv

Activate it:

Windows: .\venv\Scripts\activate

Install dependencies:

Bash
pip install -r requirements.txt

2. Configure Database
Make sure PostgreSQL is running on your machine.

Create a database named synapse_db_1:

SQL
CREATE DATABASE synapse_db_1;
Create a .env file in the root directory and add your credentials:

Code snippet
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/synapse_db_1
SECRET_KEY=SYNAPSE_SUPER_SECRET_KEY_2026
(Note: If your password has @ in it, replace it with %40, e.g., psg%40123)

3. Load Excel Data
Import concepts and question banks into PostgreSQL:

Bash
python load_data.py
4. Run the Portal
Start the server:

Bash
uvicorn app:app --reload --port 8000
Open your browser and go to:

👉 http://localhost:8000

🛠️ Verify Data
To check if student data and BKT mastery updates are saving to Postgres:

Bash
python check_db.py
