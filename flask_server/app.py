from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import numpy as np

app = Flask(__name__)
CORS(app)

# Dummy objective function
def objective_function(V, G, T):
    return V * (G / 1000) * (1 - np.exp(-V / 100)) * (1 - 0.005 * (T - 25))

# Sample Hippopotamus algorithm
def run_hippo_algorithm(pop_size, max_iter, G, T):
    population = np.random.uniform(0, 100, (pop_size, 1))
    convergence = []
    V_history = []
    best_power = -np.inf
    best_voltage = None

    for _ in range(max_iter):
        G = G * (0.9 + 0.2 * np.random.rand())
        fitness = np.array([objective_function(v[0], G, T) for v in population])
        best_idx = np.argmax(fitness)
        best_now = fitness[best_idx]
        best_voltage = population[best_idx][0]
        convergence.append(best_now)
        V_history.append(best_voltage)
        # Update pop
        for i in range(pop_size):
            partner = population[np.random.randint(pop_size)]
            population[i] += np.random.uniform(-1, 1) * (partner - population[i])
            population[i] = np.clip(population[i], 0, 100)

    final_power = float(max(convergence))
    final_voltage = float(V_history[-1])
    threshold = 0.98 * final_power
    conv_time = next((i + 1 for i, p in enumerate(convergence) if p >= threshold), max_iter)

    return {
        "best_power": final_power,
        "final_voltage": final_voltage,
        "convergence": convergence,
        "voltage_history": V_history,
        "convergence_time": conv_time
    }

@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.json
    algorithm = data.get("algorithm", "hippo")
    pop_size = int(data.get("pop_size", 30))
    max_iter = int(data.get("max_iter", 100))
    G = float(data.get("irradiance", 800))
    T = float(data.get("temperature", 25))

    start = time.time()
    if algorithm == "hippo":
        result = run_hippo_algorithm(pop_size, max_iter, G, T)
    else:
        return jsonify({"error": "Unsupported algorithm"}), 400

    result["runtime"] = round(time.time() - start, 3)
    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5000)
