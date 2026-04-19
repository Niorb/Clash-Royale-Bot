# 🏰 Clash Royale AI: Headless RL Bot

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Gymnasium](https://img.shields.io/badge/gymnasium-1.2.3-green.svg)
![RL](https://img.shields.io/badge/Reinforcement%20Learning-PPO-orange.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

An advanced Reinforcement Learning project aimed at training an AI agent to master Clash Royale. This project utilizes a **Sim-to-Real** architecture, leveraging a high-speed headless Python simulator for rapid training before deploying to a live game environment via Android emulation and Computer Vision.

---

## 🚀 The Vision

Training an AI for real-time strategy games is a massive challenge. Clash Royale involves hidden information (opponent card rotation), resource management (elixir), and spatial positioning. 

Our strategy follows the path of industry leaders like **DeepMind (AlphaStar)** and **OpenAI (OpenAI Five)**:
1.  **Phase 1: The Simulator** 🧠 - A custom, high-speed Python arena where an agent can play millions of games in hours.
2.  **Phase 2: The Perception Layer** 👁️ - YOLO-based object detection to "see" the game screen in real-time.
3.  **Phase 3: The Transfer** ⚡ - Deploying the trained "brain" to play against humans via ADB and Computer Vision.

---

## 🏗️ Architecture

### 1. Headless Simulator (`/`)
A custom-built `gymnasium` environment that replicates Clash Royale mechanics:
- **State Representation**: Vectorized game state (Elixir, Tower HP, Unit positions/health).
- **Action Space**: Discrete deployment decisions.
- **Physics**: Simplified 1D/2D combat logic with pathfinding and engagement ranges.

### 2. Reinforcement Learning Pipeline
Utilizes **PPO (Proximal Policy Optimization)** via Stable Baselines3 to train robust strategies through self-play and randomized scenarios.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/clash-royale-bot.git
cd clash-royale-bot

# Install dependencies
pip install -r requirements.txt
```

---

## 📈 Current Progress

- [x] **Core Game Objects**: `Tower` and `Troop` logic with combat math.
- [x] **Gym Environment**: Functional `ClashRoyaleEnv` wrapper.
- [x] **Verification**: Simulator successfully running with random agents.
- [ ] **Baseline Agent**: Training the first PPO agent on the 1-lane simulator.
- [ ] **Multi-Card Support**: Expanding the simulator to include spells and ranged units.
- [ ] **Vision System**: YOLOv8 implementation for unit tracking.

---

## 🎮 Usage

To run a test of the current simulator logic with a random agent:

```bash
python test_env.py
```

---

## 📜 Roadmap & Lessons Learned

Detailed documentation of our architectural decisions and "Sim-to-Real" strategies are maintained in our project knowledge base, focusing on:
- Efficient state vectorization for RL.
- Domain randomization to handle computer vision "noise."
- Reward shaping for positive elixir trades and tower pressure.

---

## 🤝 Contributing

This is an open research project. Feel free to open issues or submit PRs to improve the simulator's physics or the agent's training efficiency.

*Note: This project is for educational purposes only. Always adhere to the game's Terms of Service.*
