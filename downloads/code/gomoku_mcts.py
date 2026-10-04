"""Monte Carlo tree search for a small game of Gomoku (five-in-a-row).

Run from the project root with

    python3 code/gomoku_mcts.py

The implementation uses only the Python standard library.  It demonstrates
selection by UCT, expansion, random rollouts, and backpropagation,
then evaluates the search policy in color-balanced tournaments.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple


BOARD_SIZE = 7
WIN_LENGTH = 5
EXPLORATION = math.sqrt(2.0)

Board = Tuple[int, ...]
Move = int
Policy = Callable[[Board, int, random.Random], Move]


def play(board: Board, move: Move, player: int) -> Board:
    updated = list(board)
    updated[move] = player
    return tuple(updated)


def coordinates(move: Move) -> Tuple[int, int]:
    return divmod(move, BOARD_SIZE)


def outcome(board: Board, last_move: Optional[Move]) -> Optional[int]:
    """Return 1/-1 for a winner, 0 for a draw, or None if play continues."""
    if last_move is not None:
        row, column = coordinates(last_move)
        player = board[last_move]
        for row_step, column_step in ((1, 0), (0, 1), (1, 1), (1, -1)):
            length = 1
            for sign in (-1, 1):
                r = row + sign * row_step
                c = column + sign * column_step
                while (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE
                       and board[r * BOARD_SIZE + c] == player):
                    length += 1
                    r += sign * row_step
                    c += sign * column_step
            if length >= WIN_LENGTH:
                return player
    if all(value != 0 for value in board):
        return 0
    return None


def candidate_moves(board: Board) -> List[Move]:
    """Restrict search to empty cells neighboring an existing stone."""
    occupied = [index for index, value in enumerate(board) if value]
    if not occupied:
        center = BOARD_SIZE // 2
        return [center * BOARD_SIZE + center]
    candidates = set()
    for move in occupied:
        row, column = coordinates(move)
        for r in range(max(0, row - 1), min(BOARD_SIZE, row + 2)):
            for c in range(max(0, column - 1), min(BOARD_SIZE, column + 2)):
                index = r * BOARD_SIZE + c
                if board[index] == 0:
                    candidates.add(index)
    if candidates:
        return sorted(candidates)
    return [index for index, value in enumerate(board) if value == 0]


def immediate_wins(board: Board, player: int,
                   moves: Sequence[Move]) -> List[Move]:
    return [move for move in moves
            if outcome(play(board, move, player), move) == player]


def one_step_move(board: Board, player: int, rng: random.Random) -> Move:
    """Win immediately, block an immediate loss, or choose randomly."""
    moves = candidate_moves(board)
    wins = immediate_wins(board, player, moves)
    if wins:
        return rng.choice(wins)
    blocks = immediate_wins(board, -player, moves)
    if blocks:
        return rng.choice(blocks)
    return rng.choice(moves)


def random_move(board: Board, player: int, rng: random.Random) -> Move:
    del player
    return rng.choice(candidate_moves(board))


def rollout_move(board: Board, player: int, rng: random.Random) -> Move:
    """Sample a legal action for a deliberately inexpensive rollout."""
    del player
    return rng.choice([index for index, value in enumerate(board) if value == 0])


class Node:
    __slots__ = ("board", "player", "last_move", "parent", "children",
                 "untried", "visits", "value")

    def __init__(self, board: Board, player: int,
                 last_move: Optional[Move] = None,
                 parent: Optional["Node"] = None):
        self.board = board
        self.player = player
        self.last_move = last_move
        self.parent = parent
        self.children: List[Node] = []
        self.untried = (candidate_moves(board)
                        if outcome(board, last_move) is None else [])
        self.visits = 0
        self.value = 0.0

    def uct_child(self) -> "Node":
        log_parent = math.log(self.visits)
        return max(
            self.children,
            key=lambda child: (
                child.value / child.visits
                + EXPLORATION * math.sqrt(log_parent / child.visits)
            ),
        )


def rollout(board: Board, player: int, last_move: Optional[Move],
            rng: random.Random) -> int:
    result = outcome(board, last_move)
    while result is None:
        move = rollout_move(board, player, rng)
        board = play(board, move, player)
        result = outcome(board, move)
        player = -player
    return result


def mcts_move(board: Board, player: int, rng: random.Random,
              simulations: int) -> Move:
    legal = candidate_moves(board)
    wins = immediate_wins(board, player, legal)
    if wins:
        return rng.choice(wins)
    forced_blocks = immediate_wins(board, -player, legal)
    if forced_blocks:
        return rng.choice(forced_blocks)

    root = Node(board, player)
    for _ in range(simulations):
        node = root

        # Selection: follow upper-confidence bounds while fully expanded.
        while not node.untried and node.children:
            node = node.uct_child()

        # Expansion: add one previously unvisited action.
        if node.untried:
            index = rng.randrange(len(node.untried))
            move = node.untried.pop(index)
            child_board = play(node.board, move, node.player)
            node = Node(child_board, -node.player, move, node)
            node.parent.children.append(node)

        # Simulation and backpropagation.
        result = rollout(node.board, node.player, node.last_move, rng)
        while node is not None:
            node.visits += 1
            player_who_just_moved = -node.player
            if result == player_who_just_moved:
                node.value += 1.0
            elif result == 0:
                node.value += 0.5
            node = node.parent

    best = max(root.children,
               key=lambda child: (child.visits, child.value / child.visits))
    return best.last_move


def mcts_policy(simulations: int) -> Policy:
    def choose(board: Board, player: int, rng: random.Random) -> Move:
        return mcts_move(board, player, rng, simulations)
    return choose


def play_game(black: Policy, white: Policy, seed: int) -> int:
    board = (0,) * (BOARD_SIZE * BOARD_SIZE)
    rng = random.Random(seed)
    player = 1
    last_move: Optional[Move] = None
    while outcome(board, last_move) is None:
        policy = black if player == 1 else white
        move = policy(board, player, rng)
        board = play(board, move, player)
        last_move = move
        player = -player
    return outcome(board, last_move) or 0


@dataclass(frozen=True)
class MatchResult:
    wins: int
    draws: int
    losses: int

    @property
    def score(self) -> float:
        games = self.wins + self.draws + self.losses
        return (self.wins + 0.5 * self.draws) / games


def match(challenger: Policy, opponent: Policy, games: int,
          seed: int) -> MatchResult:
    wins = draws = losses = 0
    for game in range(games):
        challenger_is_black = game % 2 == 0
        black = challenger if challenger_is_black else opponent
        white = opponent if challenger_is_black else challenger
        winner = play_game(black, white, seed + game)
        challenger_color = 1 if challenger_is_black else -1
        if winner == challenger_color:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1
    return MatchResult(wins, draws, losses)


def main() -> None:
    experiments = [
        ("MCTS-30 vs random", mcts_policy(30), random_move),
        ("MCTS-100 vs random", mcts_policy(100), random_move),
        ("MCTS-100 vs one-step", mcts_policy(100), one_step_move),
    ]
    games = 6
    print(f"Gomoku: {BOARD_SIZE}x{BOARD_SIZE}, {WIN_LENGTH} in a row")
    print(f"Each match uses {games} games and alternates the challenger color.\n")
    print(f"{'match':<26} {'W-D-L':>9} {'score':>9}")
    print("-" * 46)
    for index, (name, challenger, opponent) in enumerate(experiments):
        result = match(challenger, opponent, games, seed=12000 + 100 * index)
        record = f"{result.wins}-{result.draws}-{result.losses}"
        print(f"{name:<26} {record:>9} {100.0 * result.score:8.1f}%")

    print("\nIncrease the simulation budget or change EXPLORATION to study the")
    print("trade-off between playing strength and computation.")


if __name__ == "__main__":
    main()
