from flask import Flask, render_template, request
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

app = Flask(__name__)

# ---- Simulation Functions ----

def generate_workload(timesteps=200, seed=42):
    np.random.seed(seed)
    # Simulate CPU workload using a sine function with added noise.
    workload = np.abs(np.sin(np.linspace(0, 10, timesteps)) + np.random.normal(0, 0.1, timesteps))
    return workload

# Custom Cloud Resource Allocation Environment
class CloudEnv(gym.Env):
    def __init__(self):
        super(CloudEnv, self).__init__()
        self.state = np.array([0.5, 0.5])
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.step_idx = 0
        self.workload = generate_workload()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.state = np.array([self.workload[self.step_idx], 0.5], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        demand = self.workload[self.step_idx]
        alloc = np.clip(action[0], 0.0, 1.0)
        cost = alloc  # Cost is directly proportional to the allocation.
        penalty = abs(alloc - demand)
        reward = - (cost + penalty)
        self.step_idx += 1
        done = self.step_idx >= len(self.workload)
        next_state = np.array([self.workload[self.step_idx % len(self.workload)], alloc], dtype=np.float32)
        self.state = next_state
        return next_state, reward, done, False, {}

# Dummy Agent Using a Simple Heuristic
class DummyAgent:
    def predict(self, state):
        # The agent allocates resources aiming to match the observed demand and adds a small noise factor.
        allocation = state[0] + np.random.normal(0, 0.05)
        return [np.clip(allocation, 0.0, 1.0)], None

def run_simulation(total_steps):
    env = CloudEnv()
    agent = DummyAgent()
    state, _ = env.reset()
    rewards = []
    allocations = []
    demands = []

    # Run simulation: step through the environment for the number of given steps.
    for _ in range(total_steps):
        action, _ = agent.predict(state)
        state, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        allocations.append(action[0])
        demands.append(state[0])
        if done:
            break

    images = {}

    # Generate Plot 1: Resource Allocation vs Actual Demand
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(demands, label='Actual CPU Demand', linewidth=2, color='blue')
    ax.plot(allocations, label='Agent Allocation', linestyle='--', color='orange')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("CPU Resource Allocation")
    ax.set_title("Resource Allocation vs Actual Demand")
    ax.legend(loc='upper right')
    ax.grid(True)
    images['allocation'] = encode_plot(fig)
    plt.close(fig)

    # Generate Plot 2: Agent Reward Over Time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, label="Reward per Step", color='purple')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward")
    ax.set_title("Agent Reward Trend Over Time")
    ax.legend(loc='upper right')
    ax.grid(True)
    images['reward'] = encode_plot(fig)
    plt.close(fig)

    # Generate Plot 3: Allocation Error Over Time
    errors = [abs(a - d) for a, d in zip(allocations, demands)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(errors, label="Allocation Error", color="red")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Allocation Error Over Time")
    ax.legend(loc='upper right')
    ax.grid(True)
    images['error'] = encode_plot(fig)
    plt.close(fig)

    # Generate Plot 4: Stacked Plot for Resource Usage
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(range(len(demands)), demands, allocations,
                 labels=['CPU Demand', 'Agent Allocation'], alpha=0.7, colors=['#1f77b4', '#ff7f0e'])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("CPU Usage")
    ax.set_title("Stacked CPU Demand vs Allocation")
    ax.legend(loc='upper right')
    ax.grid(True)
    images['stacked'] = encode_plot(fig)
    plt.close(fig)

    avg_reward = np.mean(rewards)
    return images, avg_reward

def encode_plot(fig):
    """Encodes a Matplotlib figure as a base64 string for web display."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img_base64

# ---- Flask Routes ----

@app.route("/", methods=["GET"])
def index():
    # Main landing page with a detailed description and input form.
    return render_template("index.html")

@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        total_steps = int(request.form.get("total_steps", "100"))
    except ValueError:
        total_steps = 100
    images, avg_reward = run_simulation(total_steps)
    return render_template("result.html", images=images, avg_reward=avg_reward, steps=total_steps)

if __name__ == "__main__":
    # Run the app and make it available to all interfaces.
    app.run(host="0.0.0.0", port=5000, debug=True)
