import numpy as np

from gameState import GameState

MAX_DISTANCE = 64

COUNT_ROWS = 31
COUNT_COLS = 28
TOTAL_PELLETS = 244


def normalize(x):
    return 0 if x < 0 else (1 if x > 1 else x)


def get_corner(x, y):
    if x < COUNT_ROWS / 2:
        return 0 if y < COUNT_COLS / 2 else 3
    return 1 if y < COUNT_COLS / 2 else 2


def opposite_corner(corner):
    return (corner + 2) % 4


def corner_position(corner):
    if corner == 0:
        return (1, 1)
    if corner == 1:
        return (COUNT_ROWS - 2, 1)
    if corner == 2:
        return (COUNT_ROWS - 2, COUNT_COLS - 2)
    return (1, COUNT_COLS - 2)


def linear_index(x, y):
    return y * COUNT_ROWS + x


def delinear_index(i):
    return (i % COUNT_ROWS, i // COUNT_COLS)


class PacbotEnv:
    _game_state: GameState

    def __init__(self, game_state):
        self._game_state = game_state
        self.temp_goal = None
        self.temp_goal_steps = 0

    def _closest_pellet_predicate(self, x, y):
        return self._game_state.pelletAt(x, y)

    def _closest_frightened_ghost_predicate(self, x, y):
        return any(
            map(
                lambda ghost: ghost.isFrightened() and ghost.location.at(x, y),
                self._game_state.ghosts,
            )
        )

    def _closest_angry_ghost_predicate(self, x, y):
        return any(
            map(
                lambda ghost: not ghost.isFrightened() and ghost.location.at(x, y),
                self._game_state.ghosts,
            )
        )

    def _is_predicate(self, x, y):
        return lambda _x, _y: (_x, _y) == (x, y)

    def _closest_intersection_predicate(self, x, y):
        return (
            (not self._game_state.wallAt(x - 1, y))
            + (not self._game_state.wallAt(x + 1, y))
            + (not self._game_state.wallAt(x, y - 1))
            + (not self._game_state.wallAt(x, y + 1))
        ) > 2

    def _find_closest(
        self,
        position,
        predicate,
        origin=None,
        default=MAX_DISTANCE,
        max_distance=MAX_DISTANCE,
    ):
        if self._game_state.wallAt(*position):
            return default

        queue = [position]
        visited = np.array([-1] * COUNT_COLS * COUNT_ROWS)

        visited[linear_index(*position)] = 0
        if origin is not None:
            visited[linear_index(*origin)] = 0

        while queue:
            x, y = queue.pop(0)
            if predicate(x, y):
                return visited[linear_index(x, y)]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_x, new_y = x + dx, y + dy
                if (
                    not self._game_state.wallAt(new_x, new_y)
                    and visited[linear_index(new_x, new_y)] == -1
                ):
                    visited[linear_index(new_x, new_y)] = (
                        visited[linear_index(x, y)] + 1
                    )
                    if visited[linear_index(new_x, new_y)] < max_distance:
                        queue.append((new_x, new_y))
        return default

    def _ghosts_flood_fill(self):
        visited = np.array([-1] * COUNT_COLS * COUNT_ROWS)

        queue = [
            (ghost.location.row, ghost.location.col)
            for ghost in self._game_state.ghosts
            if not ghost.spawning
        ]

        for ghost in queue:
            visited[linear_index(*ghost)] = 0

        while queue:
            x, y = queue.pop(0)
            steps = visited[linear_index(x, y)]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_x, new_y = x + dx, y + dy
                if (
                    not self._game_state.wallAt(new_x, new_y)
                    and visited[linear_index(new_x, new_y)] == -1
                ):
                    visited[linear_index(new_x, new_y)] = steps + 1
                    queue.append((new_x, new_y))

        return visited

    def _safe_tiles(self, position, origin=None):
        if self._game_state.wallAt(*position):
            return 0

        # check if any ghosts are at the position
        if any(
            [
                ghost.location.at(*position) and not ghost.spawning
                for ghost in self._game_state.ghosts
            ]
        ):
            return 0

        # bfs while flood filling ghosts
        ghost_flood_fill = self._ghosts_flood_fill()

        queue = [position]
        visited = np.array([-1] * COUNT_COLS * COUNT_ROWS)
        visited[linear_index(position[0], position[1])] = 0
        if origin is not None:
            visited[linear_index(origin[0], origin[1])] = 0

        safe_tiles = 0

        while queue:
            x, y = queue.pop(0)
            steps = visited[linear_index(x, y)]
            safe_tiles += 1

            if steps > MAX_DISTANCE:
                continue

            # if pacman is closer than the ghosts at that time, it's not yet entrapped
            if steps < ghost_flood_fill[linear_index(x, y)]:
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_x, new_y = x + dx, y + dy
                    if (
                        not self._game_state.wallAt(new_x, new_y)
                        and visited[linear_index(new_x, new_y)] == -1
                    ):
                        visited[linear_index(new_x, new_y)] = steps + 1
                        queue.append((new_x, new_y))

        return safe_tiles

    def get_observation(self):
        level_progress = 1 - (self._game_state.numPellets() / TOTAL_PELLETS)

        power_pellet_duration = (
            max([ghost.frightSteps for ghost in self._game_state.ghosts]) / 40
        )

        pos = (self._game_state.pacmanLoc.row, self._game_state.pacmanLoc.col)
        pos_left = (pos[0], pos[1] - 1)
        pos_right = (pos[0], pos[1] + 1)
        pos_up = (pos[0] - 1, pos[1])
        pos_down = (pos[0] + 1, pos[1])

        closest_pellet_left_distance = (
            self._find_closest(pos_left, self._closest_pellet_predicate, origin=pos)
            / MAX_DISTANCE
        )
        closest_pellet_right_distance = normalize(
            self._find_closest(pos_right, self._closest_pellet_predicate, origin=pos)
            / MAX_DISTANCE
        )
        closest_pellet_up_distance = normalize(
            self._find_closest(pos_up, self._closest_pellet_predicate, origin=pos)
            / MAX_DISTANCE
        )
        closest_pellet_down_distance = normalize(
            self._find_closest(pos_down, self._closest_pellet_predicate, origin=pos)
            / MAX_DISTANCE
        )

        self.temp_goal_steps = max(self.temp_goal_steps - 1, 0)

        if self.temp_goal_steps == 0:
            self.temp_goal = None

        if (
            closest_pellet_left_distance == 1
            and closest_pellet_right_distance == 1
            and closest_pellet_up_distance == 1
            and closest_pellet_down_distance == 1
        ):
            # no pellets found, go to the opposite corner
            if self.temp_goal is None:
                corner = get_corner(*pos)
                opp_corner = opposite_corner(corner)
                to = corner_position(opp_corner)
                self.temp_goal = to
                self.temp_goal_steps = 16
            else:
                to = self.temp_goal

            left_distance = self._find_closest(
                pos_left,
                self._is_predicate(*to),
                origin=pos,
                default=255,
                max_distance=64,
            )
            right_distance = self._find_closest(
                pos_right,
                self._is_predicate(*to),
                origin=pos,
                default=255,
                max_distance=64,
            )
            up_distance = self._find_closest(
                pos_up,
                self._is_predicate(*to),
                origin=pos,
                default=255,
                max_distance=64,
            )
            down_distance = self._find_closest(
                pos_down,
                self._is_predicate(*to),
                origin=pos,
                default=255,
                max_distance=64,
            )

            is_hz_further = (
                abs(min(left_distance, right_distance))
                > abs(min(up_distance, down_distance)),
            )

            closest_pellet_left_distance = (
                0.8
                if left_distance > right_distance
                else (0.4 if is_hz_further else 0.6)
            )
            closest_pellet_right_distance = (
                0.8
                if right_distance > left_distance
                else (0.4 if is_hz_further else 0.6)
            )
            closest_pellet_up_distance = (
                0.8
                if up_distance > down_distance
                else (0.4 if not is_hz_further else 0.6)
            )
            closest_pellet_down_distance = (
                0.8
                if down_distance > up_distance
                else (0.4 if not is_hz_further else 0.6)
            )

            if left_distance == 255:
                closest_pellet_left_distance = 1
            if right_distance == 255:
                closest_pellet_right_distance = 1
            if up_distance == 255:
                closest_pellet_up_distance = 1
            if down_distance == 255:
                closest_pellet_down_distance = 1

        closest_angry_ghost_left_distance = (
            self._find_closest(
                pos_left, self._closest_angry_ghost_predicate, origin=pos, default=0
            )
            / MAX_DISTANCE
        )
        closest_angry_ghost_right_distance = (
            self._find_closest(
                pos_right, self._closest_angry_ghost_predicate, origin=pos, default=0
            )
            / MAX_DISTANCE
        )
        closest_angry_ghost_up_distance = (
            self._find_closest(
                pos_up, self._closest_angry_ghost_predicate, origin=pos, default=0
            )
            / MAX_DISTANCE
        )
        closest_angry_ghost_down_distance = (
            self._find_closest(
                pos_down, self._closest_angry_ghost_predicate, origin=pos, default=0
            )
            / MAX_DISTANCE
        )

        closest_frightened_ghost_left_distance = (
            self._find_closest(
                pos_left, self._closest_frightened_ghost_predicate, origin=pos
            )
            / MAX_DISTANCE
        )
        closest_frightened_ghost_right_distance = (
            self._find_closest(
                pos_right, self._closest_frightened_ghost_predicate, origin=pos
            )
            / MAX_DISTANCE
        )
        closest_frightened_ghost_up_distance = (
            self._find_closest(
                pos_up, self._closest_frightened_ghost_predicate, origin=pos
            )
            / MAX_DISTANCE
        )
        closest_frightened_ghost_down_distance = (
            self._find_closest(
                pos_down, self._closest_frightened_ghost_predicate, origin=pos
            )
            / MAX_DISTANCE
        )

        closest_intersection_left_distance = (
            self._find_closest(
                pos_left, self._closest_intersection_predicate, origin=pos
            )
            / MAX_DISTANCE
        )
        closest_intersection_right_distance = (
            self._find_closest(
                pos_right, self._closest_intersection_predicate, origin=pos
            )
            / MAX_DISTANCE
        )
        closest_intersection_up_distance = (
            self._find_closest(pos_up, self._closest_intersection_predicate, origin=pos)
            / MAX_DISTANCE
        )
        closest_intersection_down_distance = (
            self._find_closest(
                pos_down, self._closest_intersection_predicate, origin=pos
            )
            / MAX_DISTANCE
        )
        safe_tiles_left = self._safe_tiles(pos_left, origin=pos)
        safe_tiles_right = self._safe_tiles(pos_right, origin=pos)
        safe_tiles_up = self._safe_tiles(pos_up, origin=pos)
        safe_tiles_down = self._safe_tiles(pos_down, origin=pos)

        min_safe_tiles = min(
            safe_tiles_left, safe_tiles_right, safe_tiles_up, safe_tiles_down
        )
        entrapment_left = (safe_tiles_left - min_safe_tiles) / MAX_DISTANCE
        entrapment_right = (safe_tiles_right - min_safe_tiles) / MAX_DISTANCE
        entrapment_up = (safe_tiles_up - min_safe_tiles) / MAX_DISTANCE
        entrapment_down = (safe_tiles_down - min_safe_tiles) / MAX_DISTANCE

        loc = self._game_state.pacmanLoc

        is_direction_left = 1 if loc.colDir < 0 else 0
        is_direction_right = 1 if loc.colDir > 0 else 0
        is_direction_up = 1 if loc.rowDir < 0 else 0
        is_direction_down = 1 if loc.rowDir > 0 else 0

        return np.array(
            list(
                map(
                    normalize,
                    [
                        level_progress,
                        power_pellet_duration,
                        closest_pellet_left_distance,
                        closest_pellet_right_distance,
                        closest_pellet_up_distance,
                        closest_pellet_down_distance,
                        closest_angry_ghost_left_distance
                        - closest_intersection_left_distance,
                        closest_angry_ghost_right_distance
                        - closest_intersection_right_distance,
                        closest_angry_ghost_up_distance
                        - closest_intersection_up_distance,
                        closest_angry_ghost_down_distance
                        - closest_intersection_down_distance,
                        closest_frightened_ghost_left_distance,
                        closest_frightened_ghost_right_distance,
                        closest_frightened_ghost_up_distance,
                        closest_frightened_ghost_down_distance,
                        entrapment_left,
                        entrapment_right,
                        entrapment_up,
                        entrapment_down,
                        is_direction_left,
                        is_direction_right,
                        is_direction_up,
                        is_direction_down,
                    ],
                )
            )
        )
