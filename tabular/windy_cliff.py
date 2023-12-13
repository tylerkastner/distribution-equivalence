import numpy as np
from typing import List, Tuple, Mapping

Pos = Tuple[int, int]


class WindyCliff:

    def __init__(
        self,
        layout: List[List[str]]
    ):
        self._visual_layout = np.zeros_like(layout, dtype=np.float32)
        self._action_mapping = {0: (-1, 0),
                                1: (1, 0),
                                2: (0, 1),
                                3: (0, -1)}

        # self._wind_direction = (0, -1)

        self._final_states = []

        # organize layout information into sets.
        self._free_tiles = set()
        self._free_tile_mapping = dict()
        self._s_to_xy = []
        free_tile_counter = 0

        self._holes = set()

        self._reward_mapping = dict()

        for y, row in enumerate(layout):
            for x, tile_type in enumerate(row):
                if tile_type == 'F':
                    self._visual_layout[y, x] = 0
                    self._free_tiles.add((x, y))
                    self._free_tile_mapping[(x, y)] = free_tile_counter
                    self._s_to_xy.append((x, y))
                    free_tile_counter += 1

                if tile_type == 'G':
                    self._visual_layout[y, x] = 2

                    self._free_tiles.add((x, y))
                    self._free_tile_mapping[(x, y)] = free_tile_counter
                    self._final_states.append(free_tile_counter)
                    self._s_to_xy.append((x, y))

                    self._reward_mapping[(x, y)] = 1.0

                    free_tile_counter += 1

                if tile_type == 'H':
                    self._visual_layout[y, x] = 3
                    self._holes.add((x, y))

                    self._reward_mapping[(x, y)] = -1.0

                    self._free_tiles.add((x, y))
                    self._free_tile_mapping[(x, y)] = free_tile_counter
                    self._final_states.append(free_tile_counter)
                    self._s_to_xy.append((x, y))

                    free_tile_counter += 1

        self._p, self._r, self._r_sas = self.build_dynamics()


    def transition_agent(self, pos: Pos, a: int) -> Mapping[Pos, float]:
        transition_probs = dict()
        x, y = pos

        dx, dy = self._action_mapping[a]
        new_x, new_y = x + dx, y + dy

        if (new_x, new_y) not in self._free_tiles:
            new_x, new_y = x, y

        transition_probs[(new_x, new_y)] = transition_probs.get((new_x, new_y), 0) + 3 / 4

        for aa, (dx, dy) in self._action_mapping.items():
            if aa == a:
                continue

            new_x, new_y = x + dx, y + dy

            if (new_x, new_y) not in self._free_tiles:
                new_x, new_y = x, y

            transition_probs[(new_x, new_y)] = transition_probs.get((new_x, new_y), 0) + 1 / 12

        return transition_probs


    def build_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:

        num_states = len(self._free_tiles)
        p = np.zeros(shape=[num_states, 4, num_states])
        r = np.zeros(shape=[num_states, 4,  num_states])

        for (x, y), rr in self._reward_mapping.items():
            state = self._free_tile_mapping[(x, y)]
            r[:, :, state] = rr

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


class CliffWalk(WindyCliff):

    def __init__(self):
        super().__init__(
            [['F','F','F','F','F','F','F', 'F', 'F', 'F', 'F', 'F'],
             ['F','F','F','F','F','F','F', 'F', 'F', 'F', 'F', 'F'],
             ['F','F','F','F','F','F','F', 'F', 'F', 'F', 'F', 'F'],
             ['F','H','H','H','H','H','H', 'H', 'H', 'H', 'H', 'G'],
            ]
        )
