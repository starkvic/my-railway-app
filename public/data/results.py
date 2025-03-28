import json

results = {
    "algorithms": [
        {"name": "Hippopotamus", "efficiency": 93.2, "runtime": 2.1},
        {"name": "PSO", "efficiency": 91.7, "runtime": 1.9},
        {"name": "GA", "efficiency": 89.5, "runtime": 2.3},
        ...
    ]
}

with open("public/data/results.json", "w") as f:
    json.dump(results, f, indent=2)
