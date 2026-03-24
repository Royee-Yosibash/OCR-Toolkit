from ocr_backbone.input_image import InputImage
import numpy as np


def grid_split_image(input_image: InputImage, grid: tuple[int,int])-> list[InputImage]:
        """Split an image into a grid of sub-images.

        Args:
            image: Input image as a numpy array (H x W x C).
            rows: Number of rows in the grid.
            cols: Number of columns in the grid.

        Returns:
            A list (rows x cols) of sub-image InputImages.
        """
        rows = grid[0]
        cols = grid[1]
        h, w = input_image.image.shape[:2]
        row_edges = np.linspace(0, h, rows + 1, dtype=int)
        col_edges = np.linspace(0, w, cols + 1, dtype=int)
        grid = []
        for r in range(rows):
            for c in range(cols):
                sub_image = input_image.image[row_edges[r] : row_edges[r + 1], 
                                              col_edges[c] : col_edges[c + 1]]
                grid.append(InputImage(image=sub_image, 
                                       x_offset= input_image.x_offset + int(col_edges[c]), 
                                       y_offset= input_image.y_offset + int(row_edges[r]))
                            )
                
        return grid