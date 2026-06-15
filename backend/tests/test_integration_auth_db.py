from __future__ import annotations

from uuid import uuid4

from app.repositories.db_models import Episode, Metric, ModelArtifact, Simulation, TrainingRun, User
from app.services.auth_service import create_user, issue_token_pair


def _unique_email(prefix: str = "user") -> str:
    return f"{prefix}-{uuid4().hex[:12]}@example.com"


def test_auth_api_roundtrip_and_refresh_rotation(client, db_session):
    email = _unique_email("auth")
    password = "test-pass-123"

    register = client.post("/auth/register", json={"email": email, "password": password})
    assert register.status_code in (200, 409)

    if register.status_code == 200:
        register_payload = register.json()
    else:
        login_response = client.post("/auth/login", json={"email": email, "password": password})
        assert login_response.status_code == 200
        register_payload = login_response.json()

    assert register_payload["user"]["email"] == email
    assert register_payload["token_type"] == "bearer"

    access_headers = {"Authorization": f"Bearer {register_payload['access_token']}"}
    me_response = client.get("/auth/me", headers=access_headers)
    assert me_response.status_code == 200
    assert me_response.json()["email"] == email

    refresh_response = client.post(
        "/auth/refresh",
        json={"refresh_token": register_payload["refresh_token"]},
    )
    assert refresh_response.status_code == 200
    refresh_payload = refresh_response.json()
    assert refresh_payload["user"]["email"] == email
    assert refresh_payload["refresh_token"] != register_payload["refresh_token"]

    logout_response = client.post(
        "/auth/logout",
        json={"refresh_token": refresh_payload["refresh_token"]},
    )
    assert logout_response.status_code == 204

    user = db_session.query(User).filter(User.email == email).one()
    assert len(user.refresh_tokens) == 2
    assert user.refresh_tokens[0].revoked_at is not None
    assert user.refresh_tokens[1].revoked_at is not None


def test_admin_can_list_users_and_persist_domain_models(client, db_session):
    admin_email = _unique_email("admin")
    admin = create_user(db_session, email=admin_email, password="admin-pass-123", role="admin")
    issue_token_pair(db_session, admin)

    other_email = _unique_email("member")
    other = create_user(db_session, email=other_email, password="member-pass-123")

    users_response = client.post("/auth/login", json={"email": admin_email, "password": "admin-pass-123"})
    assert users_response.status_code == 200
    admin_access = users_response.json()["access_token"]
    list_response = client.get(
        "/auth/users",
        headers={"Authorization": f"Bearer {admin_access}"},
    )
    assert list_response.status_code == 200
    listed_emails = {entry["email"] for entry in list_response.json()}
    assert admin_email in listed_emails
    assert other_email in listed_emails

    simulation = Simulation(
        user_id=admin.id,
        name="integration-run",
        status="running",
        config_json={"seed": 42, "num_households": 3},
    )
    training_run = TrainingRun(
        user_id=admin.id,
        algorithm="ppo",
        status="complete",
        config_json={"episodes": 2},
        total_reward=12.5,
    )
    db_session.add_all([simulation, training_run])
    db_session.flush()

    episode = Episode(simulation_id=simulation.id, episode_index=0, step_count=4, total_reward=3.5)
    db_session.add(episode)
    db_session.flush()
    metric = Metric(
        simulation_id=simulation.id,
        episode_id=episode.id,
        training_run_id=training_run.id,
        step=1,
        metric_name="reward",
        metric_value=3.5,
    )
    artifact = ModelArtifact(
        training_run_id=training_run.id,
        name="checkpoint",
        artifact_path="s3://bucket/models/checkpoint.pt",
        metrics_json={"accuracy": 0.9},
    )
    db_session.add_all([episode, metric, artifact])
    db_session.commit()
    db_session.refresh(simulation)
    db_session.refresh(training_run)

    assert simulation.owner.email == admin_email
    assert training_run.owner.email == admin_email
    assert simulation.episodes[0].episode_index == 0
    assert training_run.models[0].name == "checkpoint"
    assert training_run.metrics[0].metric_name == "reward"
