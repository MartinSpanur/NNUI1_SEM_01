import heapq
import itertools

import numpy as np


class AStar:
    # Directions are ordered clockwise:
    # north, east, south, west
    DIRECTIONS = (
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1),
    )

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    MOVE_COST = 5
    TURN_90_COST = 2
    TURN_180_COST = 3

    def __init__(
        self,
        matrix: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> None:
        self.original_matrix = np.asarray(matrix).copy()
        self.grid = self.original_matrix.copy()

        self.start = start
        self.goal = goal

        self.rows, self.columns = self.grid.shape

        # A state consists of:
        # (row, column, orientation)
        self.start_state = (
            self.start[0],
            self.start[1],
            self.SOUTH,
        )

        # Counter prevents Python from comparing states when two heap
        # entries have equal priorities.
        self._counter = itertools.count()

        # Heap entries contain:
        # (estimated_total_cost, insertion_order, cost_so_far, state)
        self.open_heap: list[
            tuple[int, int, int, tuple[int, int, int]]
        ] = []

        start_priority = self._heuristic(
            self.start[0],
            self.start[1],
        )

        heapq.heappush(
            self.open_heap,
            (
                start_priority,
                next(self._counter),
                0,
                self.start_state,
            ),
        )

        self.cost_so_far: dict[
            tuple[int, int, int],
            int,
        ] = {
            self.start_state: 0,
        }

        self.came_from: dict[
            tuple[int, int, int],
            tuple[int, int, int] | None,
        ] = {
            self.start_state: None,
        }

        self.closed_states: set[
            tuple[int, int, int]
        ] = set()

        self.finished = False
        self.path_found = False
        self.goal_state: tuple[int, int, int] | None = None

        self._restore_start_and_goal()

    def _heuristic(self, row: int, column: int) -> int:
        """
        Estimate the remaining cost using Manhattan distance.

        Turning costs are omitted to keep the heuristic admissible.
        """
        goal_row, goal_column = self.goal

        distance = (
            abs(goal_row - row)
            + abs(goal_column - column)
        )

        return distance * self.MOVE_COST

    def _turn_cost(
        self,
        current_direction: int,
        new_direction: int,
    ) -> int:
        """Return the cost of changing orientation."""

        direction_difference = (
            new_direction - current_direction
        ) % 4

        if direction_difference == 0:
            return 0

        if direction_difference == 2:
            return self.TURN_180_COST

        return self.TURN_90_COST

    def _is_traversable(
        self,
        row: int,
        column: int,
    ) -> bool:
        """Check whether a position is within the grid and not black."""

        if not (
            0 <= row < self.rows
            and 0 <= column < self.columns
        ):
            return False

        # A black square has value 1.
        return self.original_matrix[row, column] != 1

    def _restore_start_and_goal(self) -> None:
        """Ensure the start and goal retain their display colors."""

        start_row, start_column = self.start
        goal_row, goal_column = self.goal

        self.grid[start_row, start_column] = 3
        self.grid[goal_row, goal_column] = 4

    def _mark_traversed(
        self,
        row: int,
        column: int,
    ) -> None:
        """Mark an expanded square cyan unless it is start or goal."""

        position = (row, column)

        if position == self.start or position == self.goal:
            return

        if self.grid[row, column] != 1:
            self.grid[row, column] = 2

    def _build_path(self) -> None:
        """Mark the lowest-cost path in magenta."""

        if self.goal_state is None:
            return

        state: tuple[int, int, int] | None = self.goal_state

        while state is not None:
            row, column, _direction = state
            position = (row, column)

            # Keep the goal yellow so it remains visually distinguishable.
            # The start is already magenta.
            if position != self.goal:
                self.grid[row, column] = 3

            state = self.came_from[state]

        self._restore_start_and_goal()

    def a_star_step(self) -> np.ndarray:
        """
        Perform one A* expansion and return the resulting display grid.

        One call removes one lowest-cost state from the open queue,
        marks its square as traversed, and adds its valid neighbors.
        """
        if self.finished:
            return self.grid.copy()

        current_state = None
        current_cost = 0

        # Ignore outdated or already-expanded heap entries.
        while self.open_heap:
            (
                _priority,
                _order,
                heap_cost,
                candidate_state,
            ) = heapq.heappop(self.open_heap)

            if candidate_state in self.closed_states:
                continue

            best_known_cost = self.cost_so_far.get(candidate_state)

            if best_known_cost is None:
                continue

            if heap_cost != best_known_cost:
                continue

            current_state = candidate_state
            current_cost = heap_cost
            break

        # No states remain, so no path exists.
        if current_state is None:
            self.finished = True
            self.path_found = False
            self._restore_start_and_goal()
            return self.grid.copy()

        current_row, current_column, current_direction = current_state

        self.closed_states.add(current_state)
        self._mark_traversed(current_row, current_column)

        # The first goal state removed from the priority queue is optimal.
        if (current_row, current_column) == self.goal:
            self.finished = True
            self.path_found = True
            self.goal_state = current_state
            self._build_path()
            return self.grid.copy()

        for new_direction, (row_change, column_change) in enumerate(
            self.DIRECTIONS
        ):
            next_row = current_row + row_change
            next_column = current_column + column_change

            if not self._is_traversable(next_row, next_column):
                continue

            next_state = (
                next_row,
                next_column,
                new_direction,
            )

            movement_cost = (
                self.MOVE_COST
                + self._turn_cost(
                    current_direction,
                    new_direction,
                )
            )

            new_cost = current_cost + movement_cost

            previous_cost = self.cost_so_far.get(next_state)

            if (
                    previous_cost is not None
                    and new_cost >= previous_cost
            ):
                continue

            self.cost_so_far[next_state] = new_cost
            self.came_from[next_state] = current_state

            priority = (
                    new_cost
                    + self._heuristic(next_row, next_column)
            )

            heapq.heappush(
                self.open_heap,
                (
                    priority,
                    next(self._counter),
                    new_cost,
                    next_state,
                ),
            )

        self._restore_start_and_goal()
        return self.grid.copy()



    def a_star_full(self) -> np.ndarray:
        """
        Run A* until the goal is found or no valid path remains.

        Returns:
            A copy of the resulting grid matrix.
        """
        while not self.finished:
            self.a_star_step()

        return self.grid.copy()
