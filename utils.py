import pygame
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 1200, 800

# Modern color palette
COLORS = {
    'background': (250, 250, 252),      # Almost white
    'panel': (240, 240, 245),           # Light gray panel
    'text': (50, 50, 60),               # Dark gray text
    'text_secondary': (100, 100, 120),  # Lighter text
    'primary': (70, 130, 180),          # Steel blue
    'primary_light': (100, 160, 210),   # Lighter blue
    'primary_dark': (45, 95, 135),      # Darker blue
    'accent': (60, 179, 113),           # Medium sea green
    'accent_light': (100, 219, 153),    # Lighter green
    'accent_hover': (80, 199, 133),     # Slightly darker accent for hover
    'node': (220, 220, 230),            # Light node color
    'node_border': (180, 180, 195),     # Node border
    'leaf_node': (200, 230, 200),       # Light green leaf
    'highlight': (255, 165, 0),         # Orange highlight
    'correct': (100, 200, 100),         # Green success
    'incorrect': (230, 100, 100),       # Red error
    'feature_1': (70, 130, 180),        # Blue for class 0
    'feature_2': (60, 179, 113),        # Green for class 1
    'feature_3': (255, 165, 0),         # Orange for class 2
    'grid': (230, 230, 235),            # Very light gray for grid lines
    'button_disabled': (200, 200, 210)  # Disabled button color
}


def get_font(size, bold=False):
    """Get a font with the specified size and weight"""
    return pygame.font.SysFont("Arial", size, bold=bold)


def load_dataset():

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    return X, y, feature_names, target_names