"""
repository.py — Data Source Abstraction Layer for the Synapse Hybrid IRT Engine

Responsibility
--------------
Every module upstream of this one (bloom_mapper.py, feature_builder.py,
clustering.py, segregation.py, item_parameters.py, theta.py,
mastery_initializer.py) is pure: it takes and returns plain dataclasses
and never imports a database driver, a CSV parser, or an HTTP client.
This module is the seam that produces those dataclasses from something
real — a CSV folder for development/testing or a Postgres database in
production.

Design
------
    IRTRepository (ABC)
        - defines the read contract every data source must satisfy
        - returns ONLY dataclasses defined elsewhere in this package

    CSVRepository(IRTRepository)
        - reads sample_data/students.csv, questions.csv, responses.csv

    PostgresRepository(IRTRepository)
        - backed by live Postgres connection (synapse_db)
        - lazily imports psycopg2
        - maps database tables:
            student_profiles (user_id, previous_class_percentage, aptitude_score)
            question_bank    (question_id, concept_id, bloom)
            quiz_attempts    (user_id, question_id, is_correct)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, TYPE_CHECKING

from .config import load_database_url
from .feature_builder import ResponseRow, StudentProfileRow
from .mastery_initializer import ConceptAttempt
from .theta import AnswerRecord

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import psycopg2  # noqa: F401


# ── Exceptions ───────────────────────────────────────────────────────────


class RepositoryError(Exception):
    """Base class for every exception raised by this module."""


class DataSourceUnavailableError(RepositoryError):
    """Raised when the underlying data source cannot be reached."""


class RecordNotFoundError(RepositoryError):
    """Raised when a requested record does not exist in the data source."""


class MissingDependencyError(RepositoryError):
    """Raised when PostgresRepository needs psycopg2 but it is missing."""


# ── Shared Helpers ───────────────────────────────────────────────────────


def _split_concept_ids(raw: Any) -> List[str]:
    """Parses comma-separated concept IDs (e.g., 'E07,E10') into a list."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _primary_concept_id(raw: Any) -> Optional[str]:
    """Returns the primary (first) concept ID to prevent multi-concept duplicate attempt errors."""
    ids = _split_concept_ids(raw)
    return ids[0] if ids else None


def _to_bool(value: Any) -> bool:
    """Normalizes is_correct values across CSV and SQL types into booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in ("1", "true", "t", "yes", "y"):
        return True
    if text in ("0", "false", "f", "no", "n"):
        return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean is_correct value.")


def _to_optional_float(value: Any) -> Optional[float]:
    """Normalizes missing numerical values (None/NaN) to Optional[float]."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN check
        return None
    return f


# ── Abstract Interface ──────────────────────────────────────────────────


class IRTRepository(ABC):
    """The read contract every data source implementation must satisfy."""

    @abstractmethod
    def get_student_profiles(self) -> List[StudentProfileRow]: ...

    @abstractmethod
    def get_all_student_ids(self) -> List[str]: ...

    @abstractmethod
    def get_responses(
        self, student_ids: Optional[Iterable[str]] = None
    ) -> List[ResponseRow]: ...

    @abstractmethod
    def get_question_bloom_levels(self) -> Dict[str, str]: ...

    @abstractmethod
    def get_concept_attempts(self, student_id: str) -> List[ConceptAttempt]: ...

    @abstractmethod
    def get_answer_records(self, student_id: str) -> List[AnswerRecord]: ...

    def close(self) -> None:
        return None

    def __enter__(self) -> IRTRepository:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ── CSVRepository ───────────────────────────────────────────────────────


class CSVRepository(IRTRepository):
    """Development/testing data source using CSV files in sample_data/."""

    STUDENTS_FILENAME = "students.csv"
    QUESTIONS_FILENAME = "questions.csv"
    RESPONSES_FILENAME = "responses.csv"

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        if not self._data_dir.is_dir():
            raise DataSourceUnavailableError(
                f"CSVRepository data directory not found: {self._data_dir}"
            )

        self._students_by_id: Dict[str, StudentProfileRow] = self._load_students()
        self._bloom_by_question: Dict[str, str] = {}
        self._concept_ids_by_question: Dict[str, List[str]] = {}
        self._load_questions()
        self._responses: List[ResponseRow] = self._load_responses()
        self._responses_by_student: Dict[str, List[ResponseRow]] = {}
        for r in self._responses:
            self._responses_by_student.setdefault(r.student_id, []).append(r)

    @classmethod
    def from_default_sample_data(cls) -> CSVRepository:
        here = Path(__file__).resolve().parent
        return cls(here.parent / "sample_data")

    def _csv_path(self, filename: str) -> Path:
        path = self._data_dir / filename
        if not path.is_file():
            raise DataSourceUnavailableError(f"Required CSV file not found: {path}")
        return path

    def _load_students(self) -> Dict[str, StudentProfileRow]:
        import pandas as pd

        path = self._csv_path(self.STUDENTS_FILENAME)
        df = pd.read_csv(path)
        required = {"student_id", "previous_percentage"}
        missing = required - set(df.columns)
        if missing:
            raise DataSourceUnavailableError(
                f"{path} is missing required column(s): {sorted(missing)}"
            )

        students: Dict[str, StudentProfileRow] = {}
        for row in df.itertuples(index=False):
            sid = str(row.student_id)
            iq = _to_optional_float(getattr(row, "iq_score", None))
            pct = _to_optional_float(row.previous_class_percentage) if hasattr(
                row, "previous_class_percentage"
            ) else _to_optional_float(row.previous_percentage)
            students[sid] = StudentProfileRow(
                student_id=sid,
                previous_class_percentage=pct,
                iq_score=iq,
            )
        return students

    def _load_questions(self) -> None:
        import pandas as pd

        path = self._csv_path(self.QUESTIONS_FILENAME)
        df = pd.read_csv(path)
        required = {"question_id", "bloom_level"}
        missing = required - set(df.columns)
        if missing:
            raise DataSourceUnavailableError(
                f"{path} is missing required column(s): {sorted(missing)}"
            )

        for row in df.itertuples(index=False):
            qid = str(row.question_id)
            self._bloom_by_question[qid] = str(row.bloom_level).strip()
            concept_raw = getattr(row, "concept_id", None)
            self._concept_ids_by_question[qid] = _split_concept_ids(concept_raw)

    def _load_responses(self) -> List[ResponseRow]:
        import pandas as pd

        path = self._csv_path(self.RESPONSES_FILENAME)
        df = pd.read_csv(path)
        required = {"student_id", "question_id", "is_correct"}
        missing = required - set(df.columns)
        if missing:
            raise DataSourceUnavailableError(
                f"{path} is missing required column(s): {sorted(missing)}"
            )

        rows: List[ResponseRow] = []
        for row in df.itertuples(index=False):
            qid = str(row.question_id)
            bloom = self._bloom_by_question.get(qid)
            if bloom is None:
                raise DataSourceUnavailableError(
                    f"{path} references question_id {qid!r} which does not appear in {self.QUESTIONS_FILENAME}."
                )
            rows.append(
                ResponseRow(
                    student_id=str(row.student_id),
                    question_id=qid,
                    is_correct=_to_bool(row.is_correct),
                    bloom_level=bloom,
                )
            )
        return rows

    def get_student_profiles(self) -> List[StudentProfileRow]:
        return list(self._students_by_id.values())

    def get_all_student_ids(self) -> List[str]:
        return list(self._students_by_id.keys())

    def get_responses(
        self, student_ids: Optional[Iterable[str]] = None
    ) -> List[ResponseRow]:
        if student_ids is None:
            return list(self._responses)
        wanted = {str(s) for s in student_ids}
        return [r for r in self._responses if r.student_id in wanted]

    def get_question_bloom_levels(self) -> Dict[str, str]:
        return dict(self._bloom_by_question)

    def get_concept_attempts(self, student_id: str) -> List[ConceptAttempt]:
        student_id = str(student_id)
        if student_id not in self._students_by_id:
            raise RecordNotFoundError(f"Unknown student_id: {student_id!r}")

        attempts: List[ConceptAttempt] = []
        for r in self._responses_by_student.get(student_id, []):
            concept_id = _primary_concept_id(
                ",".join(self._concept_ids_by_question.get(r.question_id, []))
            )
            if concept_id is None:
                continue
            attempts.append(
                ConceptAttempt(
                    concept_id=concept_id,
                    question_id=r.question_id,
                    is_correct=r.is_correct,
                    bloom_level=r.bloom_level,
                )
            )
        return attempts

    def get_answer_records(self, student_id: str) -> List[AnswerRecord]:
        student_id = str(student_id)
        if student_id not in self._students_by_id:
            raise RecordNotFoundError(f"Unknown student_id: {student_id!r}")
        return [
            AnswerRecord(question_id=r.question_id, is_correct=r.is_correct)
            for r in self._responses_by_student.get(student_id, [])
        ]


# ── PostgresRepository (Updated for Synapse Database Schema) ─────────────


class PostgresRepository(IRTRepository):
    """Production data source connected to PostgreSQL (synapse_db).

    Maps SQL tables:
        student_profiles(user_id, previous_class_percentage, aptitude_score)
        question_bank(question_id, concept_id, bloom)
        quiz_attempts(user_id, question_id, is_correct)
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        connection: Optional[Any] = None,
    ) -> None:
        self._database_url = database_url
        self._connection = connection
        self._owns_connection = connection is None

    def _connect(self) -> Any:
        if self._connection is not None:
            return self._connection

        try:
            import psycopg2
        except ImportError as exc:
            raise MissingDependencyError(
                "PostgresRepository needs psycopg2 to open its connection. "
                "Install via `pip install psycopg2-binary`."
            ) from exc

        url = load_database_url(self._database_url)
        try:
            self._connection = psycopg2.connect(url)
        except Exception as exc:
            raise DataSourceUnavailableError(
                f"Could not connect to Postgres: {exc}"
            ) from exc
        return self._connection

    def close(self) -> None:
        if self._connection is not None and self._owns_connection:
            self._connection.close()
        self._connection = None

    def _query(self, sql: str, params: tuple = ()) -> List[tuple]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())
        except DataSourceUnavailableError:
            raise
        except Exception as exc:
            raise DataSourceUnavailableError(f"Postgres query failed: {exc}") from exc

    def get_student_profiles(self) -> List[StudentProfileRow]:
        rows = self._query(
            "SELECT user_id, previous_class_percentage, aptitude_score "
            "FROM student_profiles ORDER BY user_id"
        )
        return [
            StudentProfileRow(
                student_id=str(uid),
                previous_class_percentage=_to_optional_float(pct),
                iq_score=_to_optional_float(apt),
            )
            for uid, pct, apt in rows
        ]

    def get_all_student_ids(self) -> List[str]:
        rows = self._query("SELECT user_id FROM student_profiles ORDER BY user_id")
        return [str(r[0]) for r in rows]

    def get_responses(
        self, student_ids: Optional[Iterable[str]] = None
    ) -> List[ResponseRow]:
        base_sql = (
            "SELECT r.user_id, r.question_id, r.is_correct, q.bloom "
            "FROM quiz_attempts r JOIN question_bank q ON q.question_id = r.question_id"
        )
        if student_ids is not None:
            wanted = [int(s) for s in student_ids if s.isdigit()]
            sql = base_sql + " WHERE r.user_id = ANY(%s) ORDER BY r.user_id, r.question_id"
            rows = self._query(sql, (wanted,))
        else:
            sql = base_sql + " ORDER BY r.user_id, r.question_id"
            rows = self._query(sql)

        return [
            ResponseRow(
                student_id=str(uid),
                question_id=str(qid),
                is_correct=_to_bool(is_correct),
                bloom_level=str(bloom or "understand").strip(),
            )
            for uid, qid, is_correct, bloom in rows
        ]

    def get_question_bloom_levels(self) -> Dict[str, str]:
        rows = self._query("SELECT question_id, bloom FROM question_bank")
        return {str(qid): str(bloom or "understand").strip() for qid, bloom in rows}

    def _ensure_known_student(self, student_id: str) -> None:
        if not student_id.isdigit():
            raise RecordNotFoundError(f"Unknown student_id: {student_id!r}")
        rows = self._query(
            "SELECT 1 FROM student_profiles WHERE user_id = %s", (int(student_id),)
        )
        if not rows:
            raise RecordNotFoundError(f"Unknown student_id: {student_id!r}")

    def get_concept_attempts(self, student_id: str) -> List[ConceptAttempt]:
        student_id = str(student_id)
        self._ensure_known_student(student_id)

        rows = self._query(
            "SELECT r.question_id, r.is_correct, q.bloom, q.concept_id "
            "FROM quiz_attempts r JOIN question_bank q ON q.question_id = r.question_id "
            "WHERE r.user_id = %s ORDER BY r.question_id",
            (int(student_id),),
        )

        attempts: List[ConceptAttempt] = []
        for qid, is_correct, bloom, concept_raw in rows:
            concept_id = _primary_concept_id(concept_raw)
            if concept_id is None:
                continue
            attempts.append(
                ConceptAttempt(
                    concept_id=concept_id,
                    question_id=str(qid),
                    is_correct=_to_bool(is_correct),
                    bloom_level=str(bloom or "understand").strip(),
                )
            )
        return attempts

    def get_answer_records(self, student_id: str) -> List[AnswerRecord]:
        student_id = str(student_id)
        self._ensure_known_student(student_id)

        rows = self._query(
            "SELECT question_id, is_correct FROM quiz_attempts "
            "WHERE user_id = %s ORDER BY question_id",
            (int(student_id),),
        )
        return [
            AnswerRecord(question_id=str(qid), is_correct=_to_bool(is_correct))
            for qid, is_correct in rows
        ]


__all__ = [
    "IRTRepository",
    "CSVRepository",
    "PostgresRepository",
    "RepositoryError",
    "DataSourceUnavailableError",
    "RecordNotFoundError",
    "MissingDependencyError",
]