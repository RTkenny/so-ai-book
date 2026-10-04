"""Tabular Q-learning for a finite-state inventory-control problem.

The script uses only the Python standard library.  Run it with

    python3 code/inventory_q_learning.py

from the root directory of the LaTeX project.
"""

from math import exp, sqrt
from random import Random
from statistics import mean, stdev


class InventoryEnv:
    """Periodic-review inventory system with zero replenishment lead time."""

    def __init__(
        self,
        min_inventory=-20,
        max_inventory=20,
        max_order=10,
        demand_mean=4.0,
        fixed_order_cost=2.0,
        unit_order_cost=1.0,
        holding_cost=0.5,
        shortage_cost=4.0,
        horizon=30,
        seed=0,
    ):
        self.min_inventory = min_inventory
        self.max_inventory = max_inventory
        self.max_order = max_order
        self.demand_mean = demand_mean
        self.fixed_order_cost = fixed_order_cost
        self.unit_order_cost = unit_order_cost
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.horizon = horizon
        self.rng = Random(seed)
        self.inventory = 0
        self.time = 0

    @property
    def num_states(self):
        return self.max_inventory - self.min_inventory + 1

    @property
    def num_actions(self):
        return self.max_order + 1

    def state_index(self, state):
        return state - self.min_inventory

    def reset(self):
        self.inventory = 0
        self.time = 0
        return self.inventory

    def _poisson_demand(self):
        # Knuth's sampler is sufficient for the small mean used here.
        threshold = exp(-self.demand_mean)
        product = 1.0
        count = 0
        while product > threshold:
            product *= self.rng.random()
            count += 1
        return count - 1

    def step(self, action):
        if action < 0 or action > self.max_order:
            raise ValueError("action is outside the admissible range")

        demand = self._poisson_demand()
        next_inventory = self.inventory + action - demand
        next_inventory = max(
            self.min_inventory,
            min(self.max_inventory, next_inventory),
        )

        ordering = self.unit_order_cost * action
        if action > 0:
            ordering += self.fixed_order_cost
        holding = self.holding_cost * max(next_inventory, 0)
        shortage = self.shortage_cost * max(-next_inventory, 0)
        reward = -(ordering + holding + shortage)

        self.inventory = next_inventory
        self.time += 1
        done = self.time >= self.horizon
        return next_inventory, reward, done


def greedy_action(q_values):
    """Return a deterministic maximizing action (smallest one on a tie)."""
    return max(range(len(q_values)), key=q_values.__getitem__)


def train_q_learning(env_parameters, episodes=20000, gamma=0.95, seed=7):
    env = InventoryEnv(**env_parameters, seed=seed)
    action_rng = Random(seed + 1)
    q_table = [
        [0.0 for _ in range(env.num_actions)]
        for _ in range(env.num_states)
    ]
    visits = [
        [0 for _ in range(env.num_actions)]
        for _ in range(env.num_states)
    ]
    episode_costs = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        epsilon = max(0.05, 0.8 * (0.9995 ** episode))

        while not done:
            state_id = env.state_index(state)
            if action_rng.random() < epsilon:
                action = action_rng.randrange(env.num_actions)
            else:
                action = greedy_action(q_table[state_id])

            next_state, reward, done = env.step(action)
            next_state_id = env.state_index(next_state)

            visits[state_id][action] += 1
            alpha = visits[state_id][action] ** (-0.6)
            continuation = 0.0 if done else max(q_table[next_state_id])
            target = reward + gamma * continuation
            q_table[state_id][action] += alpha * (
                target - q_table[state_id][action]
            )

            state = next_state
            total_reward += reward

        episode_costs.append(-total_reward)

    return q_table, episode_costs


def evaluate_policy(env_parameters, policy, replications=5000, seed=1000):
    """Estimate expected finite-horizon cost using fresh simulation runs."""
    env = InventoryEnv(**env_parameters, seed=seed)
    costs = []
    for _ in range(replications):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            state, reward, done = env.step(policy(state))
            total_reward += reward
        costs.append(-total_reward)

    estimate = mean(costs)
    half_width = 1.96 * stdev(costs) / sqrt(replications)
    return estimate, half_width


def main():
    parameters = {
        "min_inventory": -20,
        "max_inventory": 20,
        "max_order": 10,
        "demand_mean": 4.0,
        "fixed_order_cost": 2.0,
        "unit_order_cost": 1.0,
        "holding_cost": 0.5,
        "shortage_cost": 4.0,
        "horizon": 30,
    }

    q_table, training_costs = train_q_learning(parameters)
    min_inventory = parameters["min_inventory"]
    max_order = parameters["max_order"]

    def learned_policy(state):
        return greedy_action(q_table[state - min_inventory])

    def no_order_policy(_state):
        return 0

    def base_stock_policy(state, target=5):
        return min(max_order, max(0, target - state))

    print("Mean training cost, first 500 episodes:",
          round(mean(training_costs[:500]), 2))
    print("Mean training cost, last 500 episodes: ",
          round(mean(training_costs[-500:]), 2))

    policies = {
        "Q-learning": learned_policy,
        "Base-stock": base_stock_policy,
        "No order": no_order_policy,
    }
    for name, policy in policies.items():
        estimate, half_width = evaluate_policy(parameters, policy)
        print(f"{name:11s}: {estimate:7.2f} +/- {half_width:.2f}")

    print("\nLearned policy (inventory -> order quantity)")
    for state in range(-10, 11):
        print(f"{state:3d} -> {learned_policy(state):2d}")


if __name__ == "__main__":
    main()
