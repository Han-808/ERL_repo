"""
Sokoban environment for Experiential Reinforcement Learning.

An n×n grid puzzle with n sampled from [6, 8] per episode (paper
arXiv 2602.13949, Appendix B.2).  Border cells are walls; a single box
must be pushed onto a single goal tile.  Layouts are sampled to be
solvable in ≤ MAX_STEPS actions.

Grid symbols (abstract, matches paper Appendix B.2)
---------------------------------------------------
  E   wall
  D   empty floor
  A   player on floor
  a   player on goal
  B   box on floor
  b   box on goal
  C   goal tile (empty)
"""

import random


class Sokoban:
    ACTIONS   = ["Up", "Down", "Left", "Right"]
    MAX_STEPS = 8
    MIN_N     = 6
    MAX_N     = 8

    _DELTAS = {
        "Up":    (-1,  0),
        "Down":  ( 1,  0),
        "Left":  ( 0, -1),
        "Right": ( 0,  1),
    }

    def __init__(self, seed: int = 0):
        self._master_seed  = seed
        self._episode_idx  = 0
        self._current_seed = None
        self._generate(seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, seed: int = None):
        """
        Return to the start state.

        When ``seed`` is given, regenerate the grid with that seed.
        When ``seed`` is None, reset player/box positions on the
        *current* grid (used when a caller wants to replay the current map).
        """
        if seed is not None:
            self._generate(seed)
        else:
            self.player = self._start_player
            self.boxes  = set(self._start_boxes)
            self.steps  = 0
            self.done   = False
        return self.get_observation()

    def get_observation(self):
        return (
            f"Grid:\n{self._render_grid()}\n\n"
            f"Valid actions: {self.ACTIONS}\n"
            f"Step: {self.steps}/{self.MAX_STEPS}"
        )

    def render(self):
        print(self._render_grid())

    def step(self, actions):
        if self.done:
            return self._render_grid(), "Episode already ended.", 0, True

        parts  = []
        reward = 0

        for action in actions:
            if self.done:
                break

            step_num = self.steps + 1

            if action not in self._DELTAS:
                parts.append(f"Step {step_num}: invalid action '{action}', skipped.")
                self.steps += 1
                if self.steps >= self.MAX_STEPS:
                    self.done = True
                    parts.append("Maximum steps reached.")
                continue

            dr, dc = self._DELTAS[action]
            pr, pc = self.player
            nr, nc = pr + dr, pc + dc

            if not self._in_bounds(nr, nc):
                parts.append(
                    f"Step {step_num}: moved {action}, hit the boundary, stayed in place."
                )
                self.steps += 1
                if self.steps >= self.MAX_STEPS:
                    self.done = True
                    parts.append("Maximum steps reached.")
                continue

            if self._is_wall(nr, nc):
                parts.append(
                    f"Step {step_num}: moved {action}, hit E (wall), stayed in place."
                )
                self.steps += 1
                if self.steps >= self.MAX_STEPS:
                    self.done = True
                    parts.append("Maximum steps reached.")
                continue

            # ---- box push ----------------------------------------------
            if (nr, nc) in self.boxes:
                bnr, bnc = nr + dr, nc + dc

                if (
                    not self._in_bounds(bnr, bnc)
                    or self._is_wall(bnr, bnc)
                    or (bnr, bnc) in self.boxes
                ):
                    parts.append(
                        f"Step {step_num}: moved {action}, "
                        "B (box) cannot be pushed, move failed."
                    )
                    self.steps += 1
                    if self.steps >= self.MAX_STEPS:
                        self.done = True
                        parts.append("Maximum steps reached.")
                    continue

                self.boxes.discard((nr, nc))
                self.boxes.add((bnr, bnc))
                self.player = (nr, nc)

                if (bnr, bnc) in self.targets:
                    parts.append(
                        f"Step {step_num}: moved {action}, pushed B onto C (goal)!"
                    )
                else:
                    parts.append(f"Step {step_num}: moved {action}, pushed B.")

            else:
                self.player = (nr, nc)
                parts.append(f"Step {step_num}: moved {action}.")

            self.steps += 1

            if self._all_boxes_on_targets():
                self.done = True
                reward = 1
                parts.append("All boxes are on targets. Puzzle solved!")
                break

            if self.steps >= self.MAX_STEPS and not self.done:
                self.done = True
                parts.append("Maximum steps reached.")

        feedback = " ".join(parts) if parts else "No actions executed."
        return self._render_grid(), feedback, reward, self.done

    # Backwards-compatible accessors
    @property
    def ROWS(self): return self.rows
    @property
    def COLS(self): return self.cols

    # ------------------------------------------------------------------
    # Map generation
    #
    # Construction: pick a goal, then pick a push direction and a push
    # distance k ∈ [1, 3].  Box is placed k cells opposite the push so
    # pushing in direction d, k times, lands it on the goal.  Player is
    # placed behind the box.  Every generated puzzle is therefore
    # solvable in exactly k pushes (k ≤ 3 ≤ MAX_STEPS).
    # ------------------------------------------------------------------

    def _generate(self, seed):
        self._current_seed = seed
        rng = random.Random(seed)

        for _ in range(100):
            n = rng.randint(self.MIN_N, self.MAX_N)
            layout = self._try_generate(rng, n)
            if layout is not None:
                self._install(layout)
                return

        # Fallback: a trivially solvable 6×6 (guarantees liveness)
        n = 6
        layout = {
            "n": n,
            "player": (2, 2),
            "box":    (2, 3),
            "goal":   (2, 4),
        }
        self._install(layout)

    def _try_generate(self, rng, n):
        """Try once to sample a valid (player, box, goal) triple."""
        # Interior cells (walls form the border)
        interior = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)]
        if len(interior) < 3:
            return None

        goal = rng.choice(interior)
        directions = [("Up", -1, 0), ("Down", 1, 0),
                      ("Left", 0, -1), ("Right", 0, 1)]
        rng.shuffle(directions)

        for _, dr, dc in directions:
            k = rng.randint(1, 3)
            # Box starts k cells opposite push direction
            br, bc = goal[0] - dr * k, goal[1] - dc * k
            # Player one cell further back
            pr, pc = br - dr, bc - dc
            if not (1 <= br < n - 1 and 1 <= bc < n - 1):
                continue
            if not (1 <= pr < n - 1 and 1 <= pc < n - 1):
                continue
            # All positions distinct
            if len({goal, (br, bc), (pr, pc)}) != 3:
                continue
            return {"n": n, "player": (pr, pc),
                    "box": (br, bc), "goal": goal}
        return None

    def _install(self, layout):
        n    = layout["n"]
        self.rows = n
        self.cols = n

        # Floor: walls on border, D (floor) or C (goal) inside
        self.floor = {}
        for r in range(n):
            for c in range(n):
                if r == 0 or r == n - 1 or c == 0 or c == n - 1:
                    self.floor[(r, c)] = 'E'
                else:
                    self.floor[(r, c)] = 'D'
        self.floor[layout["goal"]] = 'C'

        self.targets = {layout["goal"]}
        self._start_player = layout["player"]
        self._start_boxes  = {layout["box"]}

        self.player = self._start_player
        self.boxes  = set(self._start_boxes)
        self.steps  = 0
        self.done   = False
        self._episode_idx += 1

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, r, c):
        return self.floor.get((r, c), 'E') == 'E'

    def _all_boxes_on_targets(self):
        return all(b in self.targets for b in self.boxes)

    def _render_grid(self):
        lines = []
        for r in range(self.rows):
            cells = []
            for c in range(self.cols):
                pos  = (r, c)
                base = self.floor.get(pos, 'E')
                if base == 'E':
                    cells.append('E')
                elif pos == self.player:
                    cells.append('a' if base == 'C' else 'A')
                elif pos in self.boxes:
                    cells.append('b' if base == 'C' else 'B')
                else:
                    cells.append('C' if base == 'C' else 'D')
            lines.append(' '.join(cells))
        return '\n'.join(lines)


# ======================================================================
# Quick smoke-test
# ======================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Sokoban Environment — Self-Test")
    print("=" * 50)

    env = Sokoban(seed=0)
    print(f"\nInitial observation (n={env.rows}, seed={env._current_seed}):")
    print(env.get_observation())

    # --- Verify grid size ∈ [6, 8] ---
    for s in range(30):
        e = Sokoban(seed=s)
        assert Sokoban.MIN_N <= e.rows <= Sokoban.MAX_N, \
            f"Seed {s}: n={e.rows} out of [{Sokoban.MIN_N},{Sokoban.MAX_N}]"
    print("[PASS] Grid sizes all in [6, 8].")

    # --- Verify every generated puzzle is solvable by the planted solution ---
    # Construction guarantees the solution is "push direction, k times",
    # where direction = goal - box (normalized) and k = L1 distance.
    for s in range(20):
        e = Sokoban(seed=s)
        (br, bc) = next(iter(e._start_boxes))
        (gr, gc) = next(iter(e.targets))
        dr = 0 if gr == br else (1 if gr > br else -1)
        dc = 0 if gc == bc else (1 if gc > bc else -1)
        assert (dr == 0) ^ (dc == 0), f"Seed {s}: box-to-goal not axis-aligned"
        name = {(-1, 0): "Up", (1, 0): "Down",
                (0, -1): "Left", (0, 1): "Right"}[(dr, dc)]
        k = abs(gr - br) + abs(gc - bc)
        _, fb, r, d = e.step([name] * k)
        assert r == 1 and d is True, \
            f"Seed {s}: planted solution {[name]*k} did not solve. fb={fb}"
    print("[PASS] 20 random seeds all solvable by planted solution.")

    # --- Verify abstract symbols only ---
    e = Sokoban(seed=3)
    grid = e._render_grid()
    for sym in ('@', '$', '.', '*', '+', '#'):
        assert sym not in grid, f"Legacy symbol '{sym}' leaked"
    for sym in ('A', 'B', 'C', 'E'):
        assert sym in grid, f"Abstract symbol '{sym}' missing"
    print("[PASS] Abstract symbol encoding correct.")

    # --- Verify reset() without seed reuses same map ---
    e = Sokoban(seed=11)
    snapshot_floor = e.floor
    e.step(["Up", "Down"])
    e.reset()
    assert e.floor is snapshot_floor, "reset() without seed must keep same grid"
    assert e.player == e._start_player and e.boxes == e._start_boxes, \
        "Agent/boxes must be at start after reset()"
    print("[PASS] reset() without seed reuses current grid.")

    # --- MAX_STEPS == 8 ---
    assert Sokoban.MAX_STEPS == 8, f"Expected MAX_STEPS=8, got {Sokoban.MAX_STEPS}"
    print("[PASS] MAX_STEPS=8 confirmed.")

    print("\n[ALL PASSED]")
