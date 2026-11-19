"""SQLAlchemy database engine, session factory, and health check utilities."""

from __future__ import annotations

from collections.abc import Generator
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

Base = declarative_base()


class DatabaseManager:
    """Encapsulates SQLAlchemy engine and session lifecycle for clean testing & isolation."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        kwargs: dict = {"connect_args": connect_args, "pool_pre_ping": True}
        if database_url == "sqlite:///:memory:":
            kwargs["poolclass"] = StaticPool
        self.engine: Engine = create_engine(database_url, **kwargs)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def init_db(self) -> None:
        from . import models  # noqa: F401

        Base.metadata.create_all(bind=self.engine)

    def check_health(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def get_session(self) -> Generator[Session, None, None]:
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def dispose(self) -> None:
        self.engine.dispose()
