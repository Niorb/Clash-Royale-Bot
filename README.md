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
A custom-built `PettingZoo` and `gymnasium` environment that replicates Clash Royale mechanics:
- **Arena**: 18x30 2D grid with river, bridges, and pathfinding.
- **State Representation**: 103-dimensional vectorized game state (Elixir, Tower HP, Troop/Building positions/health).
- **Action Space**: `MultiDiscrete([9, 18, 30])` (Unit Type, X, Y).
- **Physics**: 2D combat logic with engagement ranges, flying unit support, and collision enforcement.

### 2. Reinforcement Learning Pipeline
Utilizes **PPO (Proximal Policy Optimization)** via Stable Baselines3 to train robust strategies through **Iterative Co-Evolution** (alternating training between Player 0 and Player 1).

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

- [x] **2D Arena Simulator**: 18x30 grid with river/bridge logic.
- [x] **Multi-Card Support**: Knight, Archer, Minion, Giant, PEKKA, Skeletons, etc.
- [x] **Spell & Building Support**: Fireball and Cannon logic.
- [x] **PettingZoo Environment**: Functional `ClashRoyalePZ` for multi-agent training.
- [x] **Co-Evolution Training**: `train.py` for alternating self-play training.
- [x] **Pygame Visualizer**: `visualize_env.py` for real-time 2D monitoring.
- [ ] **Advanced Reward Shaping**: Elixir efficiency and tower pressure refinement.
- [ ] **Vision System**: YOLOv8 implementation for unit tracking.

---

## 🎮 Usage

To train the agents using co-evolution:
```bash
python train.py
```

To watch the trained agents battle in the 2D arena:
```bash
python visualize_env.py
```

---

## 📜 Roadmap & Lessons Learned

Detailed documentation of our architectural decisions and "Sim-to-Real" strategies are maintained in our project knowledge base (Obsidian Vault), focusing on:
- 2D grid navigation and bridge pathfinding.
- Efficient state vectorization (103-dimensional observation).
- Iterative training to prevent strategy stagnation.


---

## 🤝 Contributing

This is an open research project. Feel free to open issues or submit PRs to improve the simulator's physics or the agent's training efficiency.

*Note: This project is for educational purposes only. Always adhere to the game's Terms of Service.*
