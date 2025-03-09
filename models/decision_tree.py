import numpy as np
import random
import pygame
from models.node import Node
from utils import COLORS, get_font

class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
        self.feature_idxs = []
        self.build_steps = []
        
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.feature_idxs = random.sample(range(X.shape[1]), self.n_features)
        self.root = self._grow_tree(X, y)
        return self
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_classes = len(np.unique(y))
        
 
        self.build_steps.append(('checking', depth, n_samples, n_classes))
        
     
        if (depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            self.build_steps.append(('leaf', depth, leaf_value))
            node = Node(depth=depth)
            node.value = leaf_value
            node.samples = n_samples
            return node
        

        best_feature_idx, best_threshold = self._best_split(X, y, self.feature_idxs)
        

        left_idxs = X[:, best_feature_idx] <= best_threshold
        right_idxs = ~left_idxs
        
        self.build_steps.append(('split', depth, best_feature_idx, best_threshold, 
                               np.sum(left_idxs), np.sum(right_idxs)))
        
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        

        node = Node(depth=depth)
        node.feature_idx = best_feature_idx
        node.threshold = best_threshold
        node.left = left
        node.right = right
        node.samples = n_samples
        return node
    
    def _best_split(self, X, y, feature_idxs):
        best_gain = -1
        split_feature_idx, split_threshold = None, None
        
        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:

                left_idxs = X_column <= threshold
                right_idxs = ~left_idxs
                
                if np.sum(left_idxs) > 0 and np.sum(right_idxs) > 0:
                    gain = self._information_gain(y, y[left_idxs], y[right_idxs])
                    
                    if gain > best_gain:
                        best_gain = gain
                        split_feature_idx = feature_idx
                        split_threshold = threshold
        
        return split_feature_idx, split_threshold
    
    def _information_gain(self, parent, left_child, right_child):
  
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        
        gain = self._entropy(parent) - (weight_left * self._entropy(left_child) + 
                                      weight_right * self._entropy(right_child))
        return gain
    
    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
    
    def _most_common_label(self, y):
        counter = np.bincount(y)
        return np.argmax(counter)
    
    def _assign_positions(self, node, x, y, level_width, level_height):
        if node is None:
            return
        
    
        node.pos = (x, y)
        node.radius = 22 - node.depth * 2  
        node.radius = max(node.radius, 10)  
        

        if not node.is_leaf_node():
            next_level_width = level_width / 2
            self._assign_positions(node.left, x - next_level_width, y + level_height, next_level_width, level_height)
            self._assign_positions(node.right, x + next_level_width, y + level_height, next_level_width, level_height)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def visualize_tree(self, screen, feature_names):
        if not self.root:
            return
        

        from utils import WIDTH
        self._assign_positions(self.root, WIDTH//2, 150, WIDTH//3, 120)
        
       
        self._draw_edges(screen, self.root)
        
     
        self._draw_nodes(screen, self.root, feature_names)
    
    def _draw_edges(self, screen, node):
        if node.is_leaf_node() or node.left is None:
            return
        
     
        pygame.draw.line(screen, COLORS['node_border'], 
                         node.pos, node.left.pos, 2)
        
     
        pygame.draw.line(screen, COLORS['node_border'], 
                         node.pos, node.right.pos, 2)
        
      
        self._draw_edges(screen, node.left)
        self._draw_edges(screen, node.right)
    
    def _draw_nodes(self, screen, node, feature_names, highlight=None):
        if node is None:
            return
        
     
        node_color = COLORS['leaf_node'] if node.is_leaf_node() else COLORS['node']
        
    
        if highlight == node:
            node_color = COLORS['highlight']
            

        shadow_offset = 2
        pygame.draw.circle(screen, COLORS['node_border'], 
                          (node.pos[0] + shadow_offset, node.pos[1] + shadow_offset), 
                          node.radius)
        pygame.draw.circle(screen, node_color, node.pos, node.radius)
        pygame.draw.circle(screen, COLORS['node_border'], node.pos, node.radius, 1)
        
       
        if node.is_leaf_node():
            class_text = f"Class {node.value}"
            text_surface = get_font(12).render(class_text, True, COLORS['text'])
            text_rect = text_surface.get_rect(center=node.pos)
            screen.blit(text_surface, text_rect)
        else:
            feature_name = feature_names[node.feature_idx].split(' (')[0]
            if len(feature_name) > 10:
                feature_name = feature_name[:10] + ".."
            feature_text = f"{feature_name}"
            threshold_text = f"≤ {node.threshold:.2f}"
            
            text_surface = get_font(12).render(feature_text, True, COLORS['text'])
            threshold_surface = get_font(12).render(threshold_text, True, COLORS['text'])
            

            text_rect = text_surface.get_rect(center=(node.pos[0], node.pos[1]-8))
            threshold_rect = threshold_surface.get_rect(center=(node.pos[0], node.pos[1]+8))
            screen.blit(text_surface, text_rect)
            screen.blit(threshold_surface, threshold_rect)
            
        if node.is_leaf_node():
            text_rect = text_surface.get_rect(center=node.pos)
            screen.blit(text_surface, text_rect)
            

        samples_text = f"n={node.samples}"
        samples_surface = get_font(10).render(samples_text, True, COLORS['text_secondary'])
        samples_rect = samples_surface.get_rect(center=(node.pos[0], node.pos[1]+node.radius+10))
        screen.blit(samples_surface, samples_rect)
        
        # Recursively draw children
        self._draw_nodes(screen, node.left, feature_names, highlight)
        self._draw_nodes(screen, node.right, feature_names, highlight)

    def visualize_tree_with_transform(self, screen, feature_names, zoom_level=1.0, camera_offset=[0, 0]):
        """Visualize the tree with camera transform applied"""
        if not self.root:
            return
        
        from utils import WIDTH
        self._assign_positions(self.root, WIDTH//2, 150, WIDTH//3, 120)
        
        # Draw edges with transform
        self._draw_edges_with_transform(screen, self.root, zoom_level, camera_offset)
        
        # Draw nodes with transform
        self._draw_nodes_with_transform(screen, self.root, feature_names, zoom_level, camera_offset)

    def _draw_edges_with_transform(self, screen, node, zoom_level, camera_offset, highlight=None):
        """Draw tree edges with camera transform applied"""
        if node.is_leaf_node() or node.left is None:
            return
        
        start_x = node.pos[0] * zoom_level + camera_offset[0]
        start_y = node.pos[1] * zoom_level + camera_offset[1]
        
        left_x = node.left.pos[0] * zoom_level + camera_offset[0]
        left_y = node.left.pos[1] * zoom_level + camera_offset[1]
        
        right_x = node.right.pos[0] * zoom_level + camera_offset[0]
        right_y = node.right.pos[1] * zoom_level + camera_offset[1]
        

        line_width = max(1, int(2 * zoom_level))
        
        pygame.draw.line(screen, COLORS['node_border'], 
                    (int(start_x), int(start_y)), (int(left_x), int(left_y)), line_width)
        
        pygame.draw.line(screen, COLORS['node_border'], 
                    (int(start_x), int(start_y)), (int(right_x), int(right_y)), line_width)
        

        self._draw_edges_with_transform(screen, node.left, zoom_level, camera_offset, highlight)
        self._draw_edges_with_transform(screen, node.right, zoom_level, camera_offset, highlight)

    def _draw_nodes_with_transform(self, screen, node, feature_names, zoom_level, camera_offset, highlight=None):
        """Draw tree nodes with camera transform applied"""
        if node is None:
            return
        
        # Apply zoom and camera offset to node position
        x = node.pos[0] * zoom_level + camera_offset[0]
        y = node.pos[1] * zoom_level + camera_offset[1]
        transformed_pos = (int(x), int(y))
        
        # Scale radius based on zoom
        transformed_radius = int(node.radius * zoom_level)
        
        # Determine node color
        node_color = COLORS['leaf_node'] if node.is_leaf_node() else COLORS['node']
        
        if highlight == node:
            node_color = COLORS['highlight']
        
        # Draw shadow with transform
        shadow_offset = int(2 * zoom_level)
        pygame.draw.circle(screen, COLORS['node_border'], 
                        (transformed_pos[0] + shadow_offset, transformed_pos[1] + shadow_offset), 
                        transformed_radius)
        
        # Draw node
        pygame.draw.circle(screen, node_color, transformed_pos, transformed_radius)
        pygame.draw.circle(screen, COLORS['node_border'], transformed_pos, transformed_radius, max(1, int(zoom_level)))
        
        # Calculate appropriate font size based on zoom
        font_size = max(8, int(12 * zoom_level))
        small_font_size = max(6, int(10 * zoom_level))
        
        # Draw node text
        if node.is_leaf_node():
            class_text = f"Class {node.value}"
            text_surface = get_font(font_size).render(class_text, True, COLORS['text'])
            text_rect = text_surface.get_rect(center=transformed_pos)
            screen.blit(text_surface, text_rect)
        else:
            feature_name = feature_names[node.feature_idx].split(' (')[0]
            if len(feature_name) > 10:
                feature_name = feature_name[:10] + ".."
            feature_text = f"{feature_name}"
            threshold_text = f"≤ {node.threshold:.2f}"
            
            text_surface = get_font(font_size).render(feature_text, True, COLORS['text'])
            threshold_surface = get_font(font_size).render(threshold_text, True, COLORS['text'])
            
            text_rect = text_surface.get_rect(center=(int(x), int(y - 8 * zoom_level)))
            threshold_rect = threshold_surface.get_rect(center=(int(x), int(y + 8 * zoom_level)))
            screen.blit(text_surface, text_rect)
            screen.blit(threshold_surface, threshold_rect)
        
        # Draw sample count
        samples_text = f"n={node.samples}"
        samples_surface = get_font(small_font_size).render(samples_text, True, COLORS['text_secondary'])
        samples_rect = samples_surface.get_rect(center=(int(x), int(y + transformed_radius + 10 * zoom_level)))
        screen.blit(samples_surface, samples_rect)
        
        # Recursively draw children
        self._draw_nodes_with_transform(screen, node.left, feature_names, zoom_level, camera_offset, highlight)
        self._draw_nodes_with_transform(screen, node.right, feature_names, zoom_level, camera_offset, highlight)