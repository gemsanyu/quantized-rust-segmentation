from functools import partial
from typing import List, Tuple

from torchvision.utils import draw_bounding_boxes
from PIL import Image, ImageDraw

draw_bboxes = partial(draw_bounding_boxes, fill=False, width=2, font_size=25)

def create_polygon_mask(image_size: Tuple[int,int], 
                        vertices: List[Tuple[float,float]]):
    """
    Create a grayscale image with a white polygonal area on a black background.

    Parameters:
    - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
    - vertices (list): A list of tuples, each containing the x, y coordinates of a vertex
                        of the polygon. Vertices should be in clockwise or counter-clockwise order.

    Returns:
    - PIL.Image.Image: A PIL Image object containing the polygonal mask.
    """
    mask_img = Image.new('L', image_size, 0)
    ImageDraw.Draw(mask_img, 'L').polygon(vertices, fill=(255))
    return mask_img