"""SQLAlchemy persistence compatible with SQLite and PostgreSQL."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from campus_forecast.config import get_settings
from campus_forecast.schema import ExperimentRecord


class Base(DeclarativeBase):
    pass


JSONType = JSON().with_variant(JSONB(), "postgresql")


class DatasetRow(Base):
    __tablename__ = "datasets"

    dataset_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    identifier: Mapped[str] = mapped_column(String(256), nullable=False)
    version: Mapped[str] = mapped_column(String(128), nullable=False)
    root_path: Mapped[str] = mapped_column(Text, nullable=False)
    audit: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class BuildingRow(Base):
    __tablename__ = "buildings"

    building_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    category: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)


class ExperimentRow(Base):
    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    building_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    feature_configuration: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class ForecastRow(Base):
    __tablename__ = "forecast_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    building_id: Mapped[str] = mapped_column(String(128), nullable=False)
    issue_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    horizon_step: Mapped[int] = mapped_column(Integer, nullable=False)
    p10: Mapped[float | None] = mapped_column(Float, nullable=True)
    p50: Mapped[float | None] = mapped_column(Float, nullable=True)
    p90: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual: Mapped[float | None] = mapped_column(Float, nullable=True)


def _engine():
    settings = get_settings()
    if settings.database_url.startswith("sqlite"):
        return create_engine(settings.database_url, connect_args={"check_same_thread": False})
    return create_engine(settings.database_url)


engine = _engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    get_settings().ensure_directories()
    Base.metadata.create_all(engine)


def db_session() -> Session:
    return SessionLocal()


def upsert_dataset(
    session: Session,
    *,
    dataset_id: str,
    identifier: str,
    version: str,
    root_path: str,
    audit: dict[str, Any],
) -> DatasetRow:
    row = session.get(DatasetRow, dataset_id)
    if row is None:
        row = DatasetRow(
            dataset_id=dataset_id,
            identifier=identifier,
            version=version,
            root_path=root_path,
            audit=audit,
        )
        session.add(row)
    else:
        row.identifier = identifier
        row.version = version
        row.root_path = root_path
        row.audit = audit
    session.commit()
    return row


def save_experiment(session: Session, record: ExperimentRecord) -> ExperimentRow:
    row = session.get(ExperimentRow, record.experiment_id)
    metadata = record.to_dict()
    if row is None:
        row = ExperimentRow(
            experiment_id=record.experiment_id,
            dataset_id=record.dataset_id,
            model_name=record.model_name,
            status=record.status,
            building_id=record.building_id,
            feature_configuration=record.feature_configuration,
            metadata_json=metadata,
            metrics=record.metrics,
        )
        session.add(row)
    else:
        row.status = record.status
        row.metadata_json = metadata
        row.metrics = record.metrics
    session.commit()
    return row


def list_experiment_rows(session: Session) -> list[ExperimentRow]:
    return list(session.scalars(select(ExperimentRow).order_by(ExperimentRow.created_at.desc())))


def save_forecast_preview(session: Session, result: dict[str, Any]) -> int:
    """Persist the compact preview while keeping large forecast arrays on disk."""

    if str(result.get("model_name", "")).endswith("_occupancy"):
        return 0
    architecture = result.get("model_architecture") or {}
    preview = architecture.get("forecast_preview") or []
    building_id = str(result.get("building_id") or "campus")
    experiment_id = str(result["experiment_id"])
    session.query(ForecastRow).filter(ForecastRow.experiment_id == experiment_id).delete()
    inserted = 0
    for issue in preview:
        issue_timestamp = datetime.fromisoformat(
            str(issue["issue_timestamp"]).replace("Z", "+00:00")
        )
        for horizon in issue.get("horizons", []):
            session.add(
                ForecastRow(
                    experiment_id=experiment_id,
                    building_id=building_id,
                    issue_timestamp=issue_timestamp.replace(tzinfo=None),
                    horizon_step=int(horizon["horizon_step"]),
                    p10=horizon.get("p10"),
                    p50=horizon.get("p50"),
                    p90=horizon.get("p90"),
                    actual=horizon.get("actual"),
                )
            )
            inserted += 1
    session.commit()
    return inserted
