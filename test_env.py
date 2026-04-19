from env import ClashRoyaleEnv
import time

def main():
    env = ClashRoyaleEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    print("Starting test run with random agent...")
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        if step_count % 20 == 0:
            print(f"Step: {step_count}, Reward: {reward:.4f}, Total Reward: {total_reward:.4f}")
            print(f"  Player Elixir: {obs[0]*10:.1f}, P-Tower HP: {obs[1]*100:.0f}%, E-Tower HP: {obs[2]*100:.0f}%")
            # Number of troops on each side
            p_troops = sum(1 for i in range(3) if obs[4+i*2] > 0)
            e_troops = sum(1 for i in range(3) if obs[10+i*2] > 0)
            print(f"  Troops - Player: {p_troops}, Enemy: {e_troops}")

    print("\nTest run complete!")
    print(f"Steps: {step_count}, Total Reward: {total_reward:.4f}")
    if env.e_tower.health <= 0:
        print("Outcome: PLAYER VICTORY!")
    elif env.p_tower.health <= 0:
        print("Outcome: ENEMY VICTORY!")
    else:
        print("Outcome: DRAW (Time Out)")

if __name__ == "__main__":
    main()
