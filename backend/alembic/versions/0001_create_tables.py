from alembic import op
import sqlalchemy as sa

revision = "0001_create_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")
        ),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "simulations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id")),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("config_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")
        ),
    )
    op.create_index("ix_simulations_user_id", "simulations", ["user_id"])

    op.create_table(
        "episodes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "simulation_id", sa.Integer(), sa.ForeignKey("simulations.id"), nullable=False
        ),
        sa.Column("episode_index", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column("total_reward", sa.Float()),
        sa.Column("step_count", sa.Integer()),
        sa.UniqueConstraint("simulation_id", "episode_index", name="uq_episode_index"),
    )
    op.create_index("ix_episodes_simulation_id", "episodes", ["simulation_id"])

    op.create_table(
        "households",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "simulation_id", sa.Integer(), sa.ForeignKey("simulations.id"), nullable=False
        ),
        sa.Column("household_index", sa.Integer(), nullable=False),
        sa.Column("initial_energy", sa.Float()),
        sa.Column("max_battery", sa.Float()),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
    )
    op.create_index("ix_households_simulation_id", "households", ["simulation_id"])

    op.create_table(
        "training_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id")),
        sa.Column("algorithm", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("config_json", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column("total_reward", sa.Float()),
    )
    op.create_index("ix_training_runs_user_id", "training_runs", ["user_id"])

    op.create_table(
        "models",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "training_run_id", sa.Integer(), sa.ForeignKey("training_runs.id"), nullable=False
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("artifact_path", sa.Text()),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")
        ),
    )
    op.create_index("ix_models_training_run_id", "models", ["training_run_id"])

    op.create_table(
        "metrics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("simulation_id", sa.Integer(), sa.ForeignKey("simulations.id")),
        sa.Column("episode_id", sa.Integer(), sa.ForeignKey("episodes.id")),
        sa.Column("training_run_id", sa.Integer(), sa.ForeignKey("training_runs.id")),
        sa.Column("step", sa.Integer()),
        sa.Column("metric_name", sa.String(length=100), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column(
            "recorded_at", sa.DateTime(timezone=True), server_default=sa.text("now()")
        ),
    )
    op.create_index("ix_metrics_simulation_id", "metrics", ["simulation_id"])
    op.create_index("ix_metrics_episode_id", "metrics", ["episode_id"])
    op.create_index("ix_metrics_training_run_id", "metrics", ["training_run_id"])


def downgrade() -> None:
    op.drop_index("ix_metrics_training_run_id", table_name="metrics")
    op.drop_index("ix_metrics_episode_id", table_name="metrics")
    op.drop_index("ix_metrics_simulation_id", table_name="metrics")
    op.drop_table("metrics")

    op.drop_index("ix_models_training_run_id", table_name="models")
    op.drop_table("models")

    op.drop_index("ix_training_runs_user_id", table_name="training_runs")
    op.drop_table("training_runs")

    op.drop_index("ix_households_simulation_id", table_name="households")
    op.drop_table("households")

    op.drop_index("ix_episodes_simulation_id", table_name="episodes")
    op.drop_table("episodes")

    op.drop_index("ix_simulations_user_id", table_name="simulations")
    op.drop_table("simulations")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
