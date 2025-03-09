import numpy as np
import pygame
from models.decision_tree import DecisionTree
from utils import COLORS, get_font, WIDTH

class RandomForest:
    def __init__(self, n_trees=5, max_depth=3, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        self.current_tree_index = 0
        self.build_progress = 0  # 0 to 100
        self.build_complete = False
    
    def fit(self, X, y):
        # Initialize but don't actually build the trees yet
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            self.trees.append(tree)
        
        self.X = X
        self.y = y
        self.bootstrap_indices = []
        
        for _ in range(self.n_trees):
    
            n_samples = X.shape[0]
            bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
            self.bootstrap_indices.append(bootstrap_idx)
            
        self.current_tree_index = 0
        self.build_progress = 0
        self.build_complete = False
        return self
    
    def build_next_tree(self):
        """Build the next tree in the forest"""
        if self.current_tree_index >= len(self.trees):
            self.build_complete = True
            return True  # All trees built
        

        bootstrap_idx = self.bootstrap_indices[self.current_tree_index]
        X_bootstrap = self.X[bootstrap_idx]
        y_bootstrap = self.y[bootstrap_idx]
        
        # Train the tree
        self.trees[self.current_tree_index].fit(X_bootstrap, y_bootstrap)
        
        # Move to next tree
        self.current_tree_index += 1
        self.build_progress = (self.current_tree_index / self.n_trees) * 100
        
        # Check if all trees are built
        if self.current_tree_index >= len(self.trees):
            self.build_complete = True
            
        return self.build_complete
    
    def build_all_trees(self):
        """Build all remaining trees at once"""
        while not self.build_complete:
            self.build_next_tree()
        return True
            
    def visualize_current_tree(self, screen, feature_names):
        """Visualize the current tree being built"""
        if 0 <= self.current_tree_index - 1 < len(self.trees):
            current_tree = self.trees[self.current_tree_index - 1]
            current_tree.visualize_tree(screen, feature_names)    
            
    def visualize_forest(self, screen, feature_names, selected_tree=None):
        """Visualize all trees in the forest or a selected tree"""
        if not self.trees:
            return
            
        n_built_trees = min(self.current_tree_index, len(self.trees))
        
        if selected_tree is not None and 0 <= selected_tree < n_built_trees:
            # Show only the selected tree
            self.trees[selected_tree].visualize_tree(screen, feature_names)
            
            # Display which tree we're showing
            tree_text = f"Tree {selected_tree + 1}/{n_built_trees}"
            text_surface = get_font(18, bold=True).render(tree_text, True, COLORS['text'])
            text_rect = text_surface.get_rect(topleft=(20, 20))
            screen.blit(text_surface, text_rect)
        else:
 
            tree_spacing = WIDTH / (n_built_trees + 1)
            
            for i in range(n_built_trees):
                x = (i + 1) * tree_spacing
                y = 100
                

                tree_color = COLORS['primary_light'] if i == selected_tree else COLORS['primary_dark']
                

                pygame.draw.rect(screen, COLORS['node_border'], (int(x)-5, y+15, 10, 15))

                pygame.draw.circle(screen, tree_color, (int(x), y), 20)
                pygame.draw.circle(screen, COLORS['node_border'], (int(x), y), 20, 2)
                

                text_surface = get_font(14).render(str(i+1), True, COLORS['text'])
                text_rect = text_surface.get_rect(center=(int(x), y))
                screen.blit(text_surface, text_rect)
    
    def predict(self, X):
        """Make predictions using the random forest"""

        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        

        predictions = [np.bincount(sample_pred).argmax() for sample_pred in tree_predictions]
        return np.array(predictions)

    def predict_with_viz(self, X, visualize=False):
        """Make predictions and return detailed path for visualization"""
        all_predictions = []
        detailed_paths = []
        
        for tree_idx, tree in enumerate(self.trees):
            tree_preds = []
            paths = []
            
            for x in X:
                prediction, path = self._predict_with_path(x, tree.root)
                tree_preds.append(prediction)
                paths.append(path)
                
            all_predictions.append(tree_preds)
            detailed_paths.append(paths)
        
     
        all_predictions = np.array(all_predictions)
        all_predictions = np.swapaxes(all_predictions, 0, 1)
        final_predictions = [np.bincount(sample_pred).argmax() 
                             for sample_pred in all_predictions]
        
        return np.array(final_predictions), detailed_paths
        
    def _predict_with_path(self, x, node, path=None):
        """Traverse a single tree and record the path for visualization"""
        if path is None:
            path = []
            

        path.append(node)
        
        if node.is_leaf_node():
            return node.value, path
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_with_path(x, node.left, path)
        return self._predict_with_path(x, node.right, path)

    def visualize_forest_with_transform(self, screen, feature_names, selected_tree=None, zoom_level=1.0, camera_offset=[0, 0]):
        """Visualize all trees in the forest or a selected tree with camera transform"""
        if not self.trees:
            return
            
        n_built_trees = min(self.current_tree_index, len(self.trees))
        
        if selected_tree is not None and 0 <= selected_tree < n_built_trees:

            self.trees[selected_tree].visualize_tree_with_transform(screen, feature_names, zoom_level, camera_offset)
            
            tree_text = f"Tree {selected_tree + 1}/{n_built_trees}"
            text_surface = get_font(18, bold=True).render(tree_text, True, COLORS['text'])
            text_rect = text_surface.get_rect(topleft=(140, 100))
            screen.blit(text_surface, text_rect)
        else:
            tree_spacing = WIDTH / (n_built_trees + 1)
            
            for i in range(n_built_trees):
                # Apply camera transform to tree positions
                x = ((i + 1) * tree_spacing) * zoom_level + camera_offset[0]
                y = 100 * zoom_level + camera_offset[1]
                
                tree_color = COLORS['primary_light'] if i == selected_tree else COLORS['primary_dark']
                
                # Scale sizes based on zoom level
                rect_size = 15 * zoom_level
                circle_radius = 20 * zoom_level
                line_width = max(1, int(2 * zoom_level))
                font_size = max(8, int(14 * zoom_level))
                
                pygame.draw.rect(screen, COLORS['node_border'], 
                            (int(x) - rect_size/2, y + rect_size, rect_size, 15 * zoom_level))
                
                pygame.draw.circle(screen, tree_color, (int(x), int(y)), int(circle_radius))
                pygame.draw.circle(screen, COLORS['node_border'], (int(x), int(y)), int(circle_radius), line_width)
                
                text_surface = get_font(font_size).render(str(i+1), True, COLORS['text'])
                text_rect = text_surface.get_rect(center=(int(x), int(y)))
                screen.blit(text_surface, text_rect)