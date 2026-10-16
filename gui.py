import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
import numpy as np

from image_in_out import bmp_to_binary_matrix, matrix_to_bmp
from a_star import AStar

class MatrixGridGUI(tk.Tk):
    """Tkinter GUI for displaying a NumPy matrix as a square-cell grid."""

    COLOR_MAP = {
        0: "#FFFFFF",  # White
        1: "#000000",  # Black
        2: "#00FFFF",  # Cyan
        3: "#FF00FF",  # Magenta
        4: "#FFFF00",  # Yellow
    }

    def __init__(self) -> None:
        super().__init__()

        self.title("Matrix Grid")
        self.geometry("900x700")
        self.minsize(450, 350)

        self.matrix: np.ndarray | None = None

        self.start: tuple[int, int] | None = None
        self.goal: tuple[int, int] | None = None

        self.a_star: AStar | None = None

        self._create_controls()
        self._create_grid_area()

        self.after_idle(self._load_default_image)

    def _create_controls(self) -> None:
        """Create the top control area."""

        self.controls = ttk.Frame(self, padding=8)
        self.controls.pack(side=tk.TOP, fill=tk.X)

        # Open and Save button group
        file_controls = ttk.Frame(self.controls)
        file_controls.pack(side=tk.LEFT)

        self.open_button = ttk.Button(
            file_controls,
            text="Open",
            command=self._open_bmp,
        )
        self.open_button.pack(side=tk.LEFT, padx=(0, 4))

        self.save_button = ttk.Button(
            file_controls,
            text="Save",
            command=self._save_bmp,
        )
        self.save_button.pack(side=tk.LEFT)

        # Start, Step, and Reset button group
        run_controls = ttk.Frame(self.controls)
        run_controls.pack(side=tk.RIGHT)

        self.start_button = ttk.Button(
            run_controls,
            text="Start",
            command=self._run_a_star_full,
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 4))

        self.step_button = ttk.Button(
            run_controls,
            text="Step",
            command=self._run_a_star_step,
        )
        self.step_button.pack(side=tk.LEFT, padx=(0, 4))

        self.reset_button = ttk.Button(
            run_controls,
            text="Reset",
            command=self._reset_all,
        )
        self.reset_button.pack(side=tk.LEFT)

    # GRID SECTION
    def _create_grid_area(self) -> None:
        """Create the canvas used for drawing the matrix grid."""

        self.canvas = tk.Canvas(
            self,
            background="#D0D0D0",
            highlightthickness=0,
        )

        # The canvas is not packed yet, so the grid is initially invisible.
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.canvas.bind("<Button-3>", self._select_start)
        self.canvas.bind("<Button-1>", self._select_goal)

    def set_matrix(self, matrix: np.ndarray) -> None:
        """
        Initialize the grid from a two-dimensional NumPy matrix.

        The number of grid rows and columns is determined from the matrix
        shape. Calling this method also makes the grid area visible.

        Args:
            matrix:
                A 2D NumPy array containing values from 0 to 4.

        Raises:
            ValueError:
                If the matrix is not two-dimensional, is empty, or contains
                unsupported values.
        """
        self._reset_all(redraw=False)

        matrix = np.asarray(matrix)

        if matrix.ndim != 2:
            raise ValueError(
                f"Expected a 2D matrix, but received shape {matrix.shape}."
            )

        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("The matrix cannot be empty.")

        allowed_values = np.array(list(self.COLOR_MAP))
        valid_mask = np.isin(matrix, allowed_values)

        if not np.all(valid_mask):
            invalid_values = np.unique(matrix[~valid_mask])
            raise ValueError(
                f"Unsupported matrix values: {invalid_values}. "
                f"Allowed values are {allowed_values.tolist()}."
            )

        self._reset_all(redraw=False)
        self.matrix = matrix.astype(np.uint8, copy=True)

        # Make the grid area fill everything below the controls.
        if not self.canvas.winfo_manager():
            self.canvas.pack(
                side=tk.TOP,
                fill=tk.BOTH,
                expand=True,
            )

        # Wait until Tkinter has calculated the canvas dimensions.
        self.update_idletasks()
        self.draw_grid()

    def hide_grid(self) -> None:
        """Hide the grid area without deleting the current matrix."""
        self.canvas.pack_forget()

    def show_grid(self) -> None:
        """Show the grid area if a matrix has already been initialized."""
        if self.matrix is None:
            return

        if not self.canvas.winfo_manager():
            self.canvas.pack(
                side=tk.TOP,
                fill=tk.BOTH,
                expand=True,
            )

        self.update_idletasks()
        self.draw_grid()

    def clear_matrix(self) -> None:
        """Remove the matrix and return the GUI to its initial state."""
        self.matrix = None
        self.canvas.delete("all")
        self.canvas.pack_forget()

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        """Redraw the grid whenever the available space changes."""
        if self.matrix is not None:
            self.draw_grid()

    def draw_grid(self) -> None:
        """
        Draw the current matrix using the largest possible square cells.

        The cell size is calculated from both available width and available
        height. The smaller result is used, ensuring that every cell remains
        square and the entire grid fits inside the canvas.
        """
        if self.matrix is None:
            return

        self.canvas.delete("all")

        rows, columns = self.matrix.shape

        available_width = self.canvas.winfo_width()
        available_height = self.canvas.winfo_height()

        if available_width <= 1 or available_height <= 1:
            return

        cell_size = min(
            available_width / columns,
            available_height / rows,
        )

        grid_width = cell_size * columns
        grid_height = cell_size * rows

        # Center the grid in any unused space.
        offset_x = (available_width - grid_width) / 2
        offset_y = (available_height - grid_height) / 2

        for row in range(rows):
            for column in range(columns):
                value = int(self.matrix[row, column])
                color = self.COLOR_MAP[value]

                x1 = offset_x + column * cell_size
                y1 = offset_y + row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=color,
                    outline="#808080",
                    width=1,
                )

    def _load_default_image(self) -> None:
        image_path = Path(__file__).resolve().parent / "Bludiste.bmp"

        try:
            matrix = bmp_to_binary_matrix(image_path)
            self.set_matrix(matrix)
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f"Could not load Bludiste.bmp: {error}")

    # OPEN/SAVE SECTION
    def _open_bmp(self) -> None:
        image_path = filedialog.askopenfilename(
            title="Open BMP image",
            filetypes=[
                ("BMP images", "*.bmp"),
                ("All files", "*.*"),
            ],
        )

        # The user closed the dialog without selecting a file.
        if not image_path:
            return

        try:
            matrix = bmp_to_binary_matrix(image_path)
            self.set_matrix(matrix)
        except (OSError, ValueError) as error:
            print(f"Could not open the image: {error}")

    def _save_bmp(self) -> None:
        if self.matrix is None:
            return

        output_path = filedialog.asksaveasfilename(
            title="Save BMP image",
            defaultextension=".bmp",
        )

        if not output_path:
            return

        # Always use the BMP extension.
        output_path = str(Path(output_path).with_suffix(".bmp"))

        try:
            matrix_to_bmp(self.matrix, output_path)
        except (OSError, ValueError) as error:
            print(f"Could not save the image: {error}")

    # STAR/GOAL SELECT SECTION
    def _get_clicked_cell(
            self,
            event: tk.Event,
    ) -> tuple[int, int] | None:
        if self.matrix is None:
            return None

        rows, columns = self.matrix.shape

        available_width = self.canvas.winfo_width()
        available_height = self.canvas.winfo_height()

        cell_size = min(
            available_width / columns,
            available_height / rows,
        )

        grid_width = cell_size * columns
        grid_height = cell_size * rows

        offset_x = (available_width - grid_width) / 2
        offset_y = (available_height - grid_height) / 2

        relative_x = event.x - offset_x
        relative_y = event.y - offset_y

        if (
                relative_x < 0
                or relative_y < 0
                or relative_x >= grid_width
                or relative_y >= grid_height
        ):
            return None

        column = int(relative_x / cell_size)
        row = int(relative_y / cell_size)

        return row, column

    def _select_start(self, event: tk.Event) -> None:
        """Select the start square using the right mouse button."""

        if self.matrix is None:
            return

        cell = self._get_clicked_cell(event)

        if cell is None:
            return

        row, column = cell

        # Only a zero-valued square can be selected.
        if self.matrix[row, column] != 0:
            return

        # Restore the previous start square.
        if self.start is not None:
            previous_row, previous_column = self.start

            if self.matrix[previous_row, previous_column] == 3:
                self.matrix[previous_row, previous_column] = 0

        self.matrix[row, column] = 3
        self.start = (row, column)

        self.draw_grid()

    def _select_goal(self, event: tk.Event) -> None:
        """Select the goal square using the left mouse button."""

        if self.matrix is None:
            return

        cell = self._get_clicked_cell(event)

        if cell is None:
            return

        row, column = cell

        # Only a zero-valued square can be selected.
        if self.matrix[row, column] != 0:
            return

        # Restore the previous goal square.
        if self.goal is not None:
            previous_row, previous_column = self.goal

            if self.matrix[previous_row, previous_column] == 4:
                self.matrix[previous_row, previous_column] = 0

        self.matrix[row, column] = 4
        self.goal = (row, column)

        self.draw_grid()

    def _reset_all(self, redraw: bool = True) -> None:
        """Reset the grid so that only white and black squares remain."""

        if self.matrix is not None:
            self.matrix[~np.isin(self.matrix, [0, 1])] = 0

        self.start = None
        self.goal = None
        self.a_star = None

        if redraw and self.matrix is not None:
            self.draw_grid()

    # A* SECTION
    def _create_a_star(self) -> AStar | None:
        if (
                self.matrix is None
                or self.start is None
                or self.goal is None
        ):
            return None

        self.a_star = AStar(
            matrix=self.matrix,
            start=self.start,
            goal=self.goal,
        )

        return self.a_star

    def _run_a_star_full(self) -> None:
        algorithm = self._create_a_star()

        if algorithm is None:
            return

        self.matrix = algorithm.a_star_full()
        self.draw_grid()

    def _run_a_star_step(self) -> None:
        if self.a_star is None:
            self.a_star = self._create_a_star()

        if self.a_star is None:
            return

        self.matrix = self.a_star.a_star_step()
        self.draw_grid()

