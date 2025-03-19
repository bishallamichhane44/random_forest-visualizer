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
    'grid': (230, 230, 235),            # Very light gray for grid lines
    'button_disabled': (200, 200, 210),  # Disabled button color
    'feature_1': (70, 130, 180),    # Blue 
    'feature_2': (60, 179, 113),    # Green
    'feature_3': (255, 165, 0),     # Orange
    'feature_4': (220, 20, 60),     # Crimson
    'feature_5': (106, 90, 205),    # Slate blue
    'feature_6': (255, 215, 0),     # Gold
    'feature_7': (65, 105, 225),    # Royal blue
    'feature_8': (50, 205, 50),     # Lime green
}


def get_font(size, bold=False):
    """Get a font with the specified size and weight"""
    return pygame.font.SysFont("Arial", size, bold=bold)


def load_dataset():
    """Load crop recommendation dataset from Kaggle"""
    import os
    import pandas as pd
    import kagglehub
    from sklearn.preprocessing import LabelEncoder
    
    try:
        # Download the dataset
        dataset_path = kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")
        print(f"Dataset downloaded to: {dataset_path}")
        
        # Find the CSV file in the downloaded directory
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the downloaded dataset")
            
        csv_path = os.path.join(dataset_path, csv_files[0])
        print(f"Loading dataset from: {csv_path}")
        
        # Load the dataset
        df = pd.read_csv(csv_path)
        
        # Extract features and target
        # The crop dataset typically has: N, P, K, temperature, humidity, ph, rainfall, label
        feature_cols = [col for col in df.columns if col.lower() != 'label' and col.lower() != 'crop']
        target_col = 'label' if 'label' in df.columns else 'crop'
        
        X = df[feature_cols].values
        
        # Encode the target labels to integers
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[target_col])
        
        # Get the feature names and target names
        feature_names = feature_cols
        target_names = label_encoder.classes_.tolist()
        
        return X, y, feature_names, target_names
        
    except Exception as e:
        print(f"Error loading crop dataset: {e}")
        print("Falling back to default Iris dataset")
        
        # Fallback to Iris dataset
        from sklearn import datasets
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        return X, y, feature_names, target_names