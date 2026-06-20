from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.infrastructure.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="user", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    simulations: Mapped[list[Simulation]] = relationship(
        "Simulation", back_populates="owner", cascade="all, delete-orphan"
    )
    training_runs: Mapped[list[TrainingRun]] = relationship(
        "TrainingRun", back_populates="owner", cascade="all, delete-orphan"
    )
    refresh_tokens: Mapped[list[RefreshToken]] = relationship(
        "RefreshToken", back_populates="user", cascade="all, delete-orphan"
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), index=True, nullable=False
    )
    token_jti: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    replaced_by_jti: Mapped[str | None] = mapped_column(String(64))

    user: Mapped[User] = relationship("User", back_populates="refresh_tokens")


class Simulation(Base):
    __tablename__ = "simulations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), index=True
    )
    name: Mapped[str] = mapped_column(String(255), default="Unnamed", nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="created", nullable=False)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    owner: Mapped[User | None] = relationship("User", back_populates="simulations")
    episodes: Mapped[list[Episode]] = relationship(
        "Episode", back_populates="simulation", cascade="all, delete-orphan"
    )
    households: Mapped[list[Household]] = relationship(
        "Household", back_populates="simulation", cascade="all, delete-orphan"
    )
    metrics: Mapped[list[Metric]] = relationship(
        "Metric", back_populates="simulation", cascade="all, delete-orphan"
    )


class Episode(Base):
    __tablename__ = "episodes"
    __table_args__ = (
        UniqueConstraint("simulation_id", "episode_index", name="uq_episode_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    simulation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("simulations.id"), index=True, nullable=False
    )
    episode_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    total_reward: Mapped[float | None] = mapped_column(Float)
    step_count: Mapped[int | None] = mapped_column(Integer)

    simulation: Mapped[Simulation] = relationship(
        "Simulation", back_populates="episodes"
    )
    metrics: Mapped[list[Metric]] = relationship(
        "Metric", back_populates="episode", cascade="all, delete-orphan"
    )


class Household(Base):
    __tablename__ = "households"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    simulation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("simulations.id"), index=True, nullable=False
    )
    household_index: Mapped[int] = mapped_column(Integer, nullable=False)
    initial_energy: Mapped[float | None] = mapped_column(Float)
    max_battery: Mapped[float | None] = mapped_column(Float)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    simulation: Mapped[Simulation] = relationship(
        "Simulation", back_populates="households"
    )


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), index=True
    )
    algorithm: Mapped[str] = mapped_column(String(100), default="ppo", nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="created", nullable=False)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    total_reward: Mapped[float | None] = mapped_column(Float)

    owner: Mapped[User | None] = relationship("User", back_populates="training_runs")
    metrics: Mapped[list[Metric]] = relationship(
        "Metric", back_populates="training_run", cascade="all, delete-orphan"
    )
    models: Mapped[list[ModelArtifact]] = relationship(
        "ModelArtifact", back_populates="training_run", cascade="all, delete-orphan"
    )


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    simulation_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("simulations.id"), index=True
    )
    episode_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("episodes.id"), index=True
    )
    training_run_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("training_runs.id"), index=True
    )
    step: Mapped[int | None] = mapped_column(Integer)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    simulation: Mapped[Simulation | None] = relationship(
        "Simulation", back_populates="metrics"
    )
    episode: Mapped[Episode | None] = relationship("Episode", back_populates="metrics")
    training_run: Mapped[TrainingRun | None] = relationship(
        "TrainingRun", back_populates="metrics"
    )


class ModelArtifact(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("training_runs.id"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    artifact_path: Mapped[str | None] = mapped_column(Text)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    training_run: Mapped[TrainingRun] = relationship(
        "TrainingRun", back_populates="models"
    )
