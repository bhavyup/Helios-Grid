import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
response = client.post(
    "/simulation/reset",
    json={"seed": 123, "num_households": 64, "max_episode_steps": 8},
)
print("status_code:", response.status_code)
if response.status_code != 200:
    print(response.text)
    raise SystemExit(1)

payload = response.json()
topology = payload.get("topology", {})
nodes = topology.get("nodes", [])
households = [n for n in nodes if n.get("type") == "household"]
print(f"households: {len(households)}")
unique_positions = set()
for h in households:
    x = h.get("x")
    y = h.get("y")
    unique_positions.add((round(float(x), 3), round(float(y), 3)))

print(f"unique_positions: {len(unique_positions)}")
print("layout:", topology.get("layout"))
print("bounds:", topology.get("bounds"))
print(
    json.dumps(
        {"households": len(households), "unique": len(unique_positions)}, indent=2
    )
)
