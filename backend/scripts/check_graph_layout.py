from app.infrastructure.graph_utils import create_grid_graph

G = create_grid_graph(num_households=64, num_solar_panels=8, num_wind_turbines=4)

households = [n for n, d in G.nodes(data=True) if d.get("type") == "household"]
print("household_count:", len(households))

positions = set()
for n in households:
    d = G.nodes[n]
    positions.add((round(float(d.get("x", 0)), 3), round(float(d.get("y", 0)), 3)))

print("unique_positions:", len(positions))
print("layout:", G.graph.get("layout"))
print("bounds:", G.graph.get("bounds"))
# Print first 5 households
for n in households[:5]:
    d = G.nodes[n]
    print(n, d.get("x"), d.get("y"), d.get("lot_index"))
