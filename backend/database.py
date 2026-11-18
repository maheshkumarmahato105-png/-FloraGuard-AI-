"""SQLAlchemy database engine and session management.

Uses SQLite by default and supports PostgreSQL seamlessly by setting the
DATABASE_URL environment variable (e.g., postgresql+psycopg2://user:pass@host/db).
"""

from collections.abc import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.config import DATABASE_URL

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db() -> None:
    """Create database tables if they do not exist."""
    from backend.models import db_models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def check_db_health() -> bool:
    """Verify database connectivity."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
