"""Query-efficient, forward-only training of a small neural network.

Run from the project root with

    python3 code/forward_only_training.py

The experiment compares coordinate finite differences, a two-query random
direction estimator (SPSA), and an orthogonal-subspace estimator under the
same number of loss-function queries. Exact backpropagation is included only
as a white-box reference. The implementation uses the Python standard library.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple


HIDDEN = 3
DIMENSION = 2 * HIDDEN + HIDDEN + HIDDEN + 1
QUERY_BUDGET = 2400
REPETITIONS = 5
PERTURBATION = 0.08

Example = Tuple[float, float, int]
Vector = List[float]


def make_noisy_xor(samples_per_corner: int, seed: int) -> List[Example]:
    """Generate a balanced, nonlinear binary-classification data set."""
    rng = random.Random(seed)
    data: List[Example] = []
    for x1_center, x2_center in [(-1.0, -1.0), (-1.0, 1.0),
                                 (1.0, -1.0), (1.0, 1.0)]:
        label = int(x1_center * x2_center < 0.0)
        for _ in range(samples_per_corner):
            x1 = x1_center + rng.gauss(0.0, 0.28)
            x2 = x2_center + rng.gauss(0.0, 0.28)
            data.append((x1, x2, label))
    rng.shuffle(data)
    return data


def unpack(theta: Sequence[float]):
    """Return views of the 2-3-1 network parameters."""
    w1 = theta[0:2 * HIDDEN]
    b1 = theta[2 * HIDDEN:3 * HIDDEN]
    w2 = theta[3 * HIDDEN:4 * HIDDEN]
    b2 = theta[-1]
    return w1, b1, w2, b2


def sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def softplus(z: float) -> float:
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def network_output(theta: Sequence[float], x1: float, x2: float):
    w1, b1, w2, b2 = unpack(theta)
    hidden = [math.tanh(w1[2 * j] * x1 + w1[2 * j + 1] * x2 + b1[j])
              for j in range(HIDDEN)]
    logit = b2 + sum(w2[j] * hidden[j] for j in range(HIDDEN))
    return hidden, logit


def loss(theta: Sequence[float], data: Sequence[Example]) -> float:
    total = 0.0
    for x1, x2, label in data:
        _, logit = network_output(theta, x1, x2)
        total += softplus(logit) - label * logit
    return total / len(data)


def exact_gradient(theta: Sequence[float], data: Sequence[Example]) -> Vector:
    """White-box gradient used by the backpropagation reference."""
    gradient = [0.0] * DIMENSION
    w1, _, w2, _ = unpack(theta)
    for x1, x2, label in data:
        hidden, logit = network_output(theta, x1, x2)
        error = (sigmoid(logit) - label) / len(data)
        for j in range(HIDDEN):
            local = error * w2[j] * (1.0 - hidden[j] ** 2)
            gradient[2 * j] += local * x1
            gradient[2 * j + 1] += local * x2
            gradient[2 * HIDDEN + j] += local
            gradient[3 * HIDDEN + j] += error * hidden[j]
        gradient[-1] += error
    return gradient


def shifted(theta: Sequence[float], direction: Sequence[float],
            amount: float) -> Vector:
    return [value + amount * step for value, step in zip(theta, direction)]


def coordinate_gradient(theta: Sequence[float], data: Sequence[Example],
                        radius: float, rng: random.Random) -> Vector:
    del rng
    gradient = []
    for i in range(DIMENSION):
        direction = [0.0] * DIMENSION
        direction[i] = 1.0
        plus = loss(shifted(theta, direction, radius), data)
        minus = loss(shifted(theta, direction, -radius), data)
        gradient.append((plus - minus) / (2.0 * radius))
    return gradient


def spsa_gradient(theta: Sequence[float], data: Sequence[Example],
                  radius: float, rng: random.Random) -> Vector:
    """Two-query isotropic random-direction estimator."""
    scale = 1.0 / math.sqrt(DIMENSION)
    direction = [rng.choice((-scale, scale)) for _ in range(DIMENSION)]
    plus = loss(shifted(theta, direction, radius), data)
    minus = loss(shifted(theta, direction, -radius), data)
    directional_derivative = (plus - minus) / (2.0 * radius)
    return [DIMENSION * directional_derivative * value for value in direction]


def orthonormal_directions(count: int, rng: random.Random) -> List[Vector]:
    """Draw a uniformly oriented, orthonormal set by Gram--Schmidt."""
    basis: List[Vector] = []
    while len(basis) < count:
        vector = [rng.gauss(0.0, 1.0) for _ in range(DIMENSION)]
        for unit in basis:
            projection = sum(a * b for a, b in zip(vector, unit))
            vector = [a - projection * b for a, b in zip(vector, unit)]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 1.0e-10:
            basis.append([value / norm for value in vector])
    return basis


def subspace_gradient(theta: Sequence[float], data: Sequence[Example],
                      radius: float, rng: random.Random,
                      rank: int = 3) -> Vector:
    """Average central differences over a random rank-r subspace."""
    estimate = [0.0] * DIMENSION
    for direction in orthonormal_directions(rank, rng):
        plus = loss(shifted(theta, direction, radius), data)
        minus = loss(shifted(theta, direction, -radius), data)
        derivative = (plus - minus) / (2.0 * radius)
        for i in range(DIMENSION):
            estimate[i] += (DIMENSION / rank) * derivative * direction[i]
    return estimate


class Adam:
    def __init__(self, dimension: int, learning_rate: float):
        self.learning_rate = learning_rate
        self.first = [0.0] * dimension
        self.second = [0.0] * dimension
        self.time = 0

    def step(self, theta: Vector, gradient: Sequence[float]) -> None:
        self.time += 1
        for i, value in enumerate(gradient):
            self.first[i] = 0.9 * self.first[i] + 0.1 * value
            self.second[i] = 0.999 * self.second[i] + 0.001 * value * value
            first_hat = self.first[i] / (1.0 - 0.9 ** self.time)
            second_hat = self.second[i] / (1.0 - 0.999 ** self.time)
            theta[i] -= self.learning_rate * first_hat / (math.sqrt(second_hat) + 1e-8)


@dataclass(frozen=True)
class Method:
    name: str
    estimator: Callable[[Sequence[float], Sequence[Example], float,
                         random.Random], Vector]
    queries_per_step: int
    learning_rate: float


def train(method: Method, initial: Sequence[float], data: Sequence[Example],
          seed: int) -> Tuple[Vector, int, int]:
    theta = list(initial)
    optimizer = Adam(DIMENSION, method.learning_rate)
    rng = random.Random(seed)
    steps = QUERY_BUDGET // method.queries_per_step
    for _ in range(steps):
        gradient = method.estimator(theta, data, PERTURBATION, rng)
        optimizer.step(theta, gradient)
    return theta, steps, steps * method.queries_per_step


def backprop_estimator(theta: Sequence[float], data: Sequence[Example],
                       radius: float, rng: random.Random) -> Vector:
    del radius, rng
    return exact_gradient(theta, data)


def accuracy(theta: Sequence[float], data: Sequence[Example]) -> float:
    correct = 0
    for x1, x2, label in data:
        _, logit = network_output(theta, x1, x2)
        correct += int((logit >= 0.0) == bool(label))
    return correct / len(data)


def mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half_width


def main() -> None:
    train_data = make_noisy_xor(samples_per_corner=50, seed=2026)
    test_data = make_noisy_xor(samples_per_corner=250, seed=90210)
    methods = [
        Method("Backprop reference", backprop_estimator, 1, 0.025),
        Method("Coordinate FD", coordinate_gradient, 2 * DIMENSION, 0.025),
        Method("SPSA (rank 1)", spsa_gradient, 2, 0.012),
        Method("Orthogonal rank 3", subspace_gradient, 6, 0.014),
    ]

    results = {method.name: {"loss": [], "accuracy": []} for method in methods}
    step_counts = {}
    query_counts = {}
    for repetition in range(REPETITIONS):
        init_rng = random.Random(7000 + repetition)
        initial = [init_rng.gauss(0.0, 0.35) for _ in range(DIMENSION)]
        for method in methods:
            fitted, steps, queries = train(
                method, initial, train_data, seed=8000 + repetition
            )
            results[method.name]["loss"].append(loss(fitted, test_data))
            results[method.name]["accuracy"].append(accuracy(fitted, test_data))
            step_counts[method.name] = steps
            query_counts[method.name] = queries

    print("Noisy XOR with a 2-3-1 neural network")
    print(f"Parameters: {DIMENSION}; loss-query budget: {QUERY_BUDGET}; "
          f"repetitions: {REPETITIONS}")
    print("Backpropagation is a white-box reference; the other methods use "
          "loss queries only.\n")
    header = f"{'method':<22} {'steps':>7} {'queries':>8} {'test loss':>20} {'accuracy':>20}"
    print(header)
    print("-" * len(header))
    for method in methods:
        loss_mean, loss_half = mean_ci(results[method.name]["loss"])
        acc_mean, acc_half = mean_ci(results[method.name]["accuracy"])
        print(f"{method.name:<22} {step_counts[method.name]:7d} "
              f"{query_counts[method.name]:8d} "
              f"{loss_mean:8.4f} +/- {loss_half:7.4f} "
              f"{100.0 * acc_mean:8.2f}% +/- {100.0 * acc_half:6.2f}%")

    print("\nTry changing QUERY_BUDGET, PERTURBATION, or the subspace rank.")


if __name__ == "__main__":
    main()
