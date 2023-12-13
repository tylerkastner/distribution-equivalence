import numpy as np
from typing import List, Tuple, Mapping

RoomTile = int
Pos = Tuple[int, int]


class Gridworld:

    def __init__(
            self,
            layout: List[List[RoomTile]],
            p_intended: float = 1.0,
            risky_reward: float = -1.0,
    ):
        self._p_intended = p_intended
        self._visual_layout = np.zeros_like(layout, dtype=np.float32)
        self._action_mapping = {0: (-1, 0),
                                1: (1, 0),
                                2: (0, 1),
                                3: (0, -1)}
        self._risky_reward = risky_reward
        self._final_states = []

        # organize layout information into sets.
        self._free_tiles = set()
        self._free_tile_mapping = dict()
        self._s_to_xy = []
        free_tile_counter = 0
        self._wall_tiles = set()
        self._reward_mapping = dict()
        self._risky_states = set()
        for y, row in enumerate(layout):
            for x, tile_type in enumerate(row):
                if tile_type == 0 or isinstance(tile_type, float):
                    self._visual_layout[y, x] = 0
                    self._free_tiles.add((x, y))
                    self._free_tile_mapping[(x, y)] = free_tile_counter
                    self._s_to_xy.append((x, y))
                    free_tile_counter += 1
                    if isinstance(tile_type, float):
                        self._visual_layout[y, x] = 2
                        self._reward_mapping[(x, y)] = tile_type
                        self._final_states.append(self._free_tile_mapping[(x, y)])
                elif tile_type == 2:
                    self._free_tiles.add((x, y))
                    self._free_tile_mapping[(x, y)] = free_tile_counter
                    self._s_to_xy.append((x, y))
                    free_tile_counter += 1
                    self._visual_layout[y, x] = 3
                    self._risky_states.add((x, y))
                else:
                    self._visual_layout[y, x] = 1
                    self._wall_tiles.add((x, y))
        self._p, self._r, self._r_sas = self.build_dynamics()

    def transition_agent(self, pos: Pos, a: int) -> Mapping[Pos, float]:
        transition_probs = dict()
        x, y = pos
        num_actions = len(self._action_mapping)
        for aa, (dx, dy) in self._action_mapping.items():
            new_x, new_y = x + dx, y + dy
            if (new_x, new_y) not in self._free_tiles:
                new_x, new_y = x, y
            p = self._p_intended if aa == a else (1 - self._p_intended) / (num_actions-1)
            if (new_x, new_y) in transition_probs:
                transition_probs[(new_x, new_y)] += p
            else:
                transition_probs[(new_x, new_y)] = p
        return transition_probs

    def build_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        num_states = len(self._free_tiles)
        p = np.zeros(shape=[num_states, 4, num_states])
        r = np.zeros(shape=[num_states, 4,  num_states])
        for (x, y), rr in self._reward_mapping.items():
            state = self._free_tile_mapping[(x, y)]
            r[:, :, state] = rr

        for (x, y) in self._risky_states:
            state = self._free_tile_mapping[(x, y)]

            for a in range(4):
                transition_dict = self.transition_agent((x, y), a)

                delta = self._action_mapping[a]
                next_expected_state = (x + delta[0], y + delta[1])

                su = 0
                for (new_x, new_y), pp in transition_dict.items():
                    new_state = self._free_tile_mapping[(new_x, new_y)]

                    if next_expected_state != (new_x, new_y):
                        r[state, a, new_state] = self._risky_reward
                        su += r[state, a, new_state] * pp

                    else:
                        r[state, a, new_state] = - (self._risky_reward * (1 - self._p_intended)) / pp
                        su += r[state, a, new_state] * pp

        for (x, y) in self._free_tiles:
            for a in range(4):
                # get the distribution over possible other positions the agent could be in.
                old_state = self._free_tile_mapping[(x, y)]
                for (new_x, new_y), pp in self.transition_agent((x, y), a).items():
                    new_state = self._free_tile_mapping[(new_x, new_y)]
                    p[old_state, a, new_state] = pp
        r_sas = r
        r = np.sum(r * p, axis=2)
        return p, r, r_sas

    def get_transition_tensor(self) -> np.ndarray:
        return np.copy(self._p)

    def get_reward_matrix(self) -> np.ndarray:
        return np.copy(self._r)

    def get_sas_reward_matrix(self) -> np.ndarray:
        return np.copy(self._r_sas)

    def visualize(self) -> np.ndarray:
        return np.copy(self._visual_layout)


class FourRooms(Gridworld):

    def __init__(self, p_intended=1.0):
        # 2 means puddles
        super().__init__(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1.0, 1],
             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
             [1, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             ],
            p_intended=p_intended
        )
