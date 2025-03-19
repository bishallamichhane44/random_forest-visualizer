import pygame
import numpy as np
from pygame.rect import Rect
from sklearn.model_selection import train_test_split

from models.random_forest import RandomForest
from ui.button import Button
from ui.slider import Slider
from utils import WIDTH, HEIGHT, COLORS, get_font, load_dataset

class RandomForestVisualizer:
    def __init__(self):
        # Load dataset
        self.X, self.y, self.feature_names, self.target_names = load_dataset()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Parameters for random forest
        self.n_trees = 5
        self.max_depth = 3
        self.forest = None
        
        # Visualization state
        self.state = "setup"  
        self.selected_tree = 0
        self.selected_test_sample = 0
        self.test_predictions = None
        self.prediction_paths = None
        self.viz_step = 0
        self.animation_timer = 0
        self.animation_speed = 30  
        
        # Camera/viewport settings for zoomable and pannable view
        self.zoom_level = 1.0
        self.camera_offset = [0, 0]
        self.is_dragging = False
        self.drag_start = None
        self.min_zoom = 0.5
        self.max_zoom = 3.0

        self.buttons = []
        self.sliders = []
        self._create_ui_elements()

        self.show_predictions = False
    
    def _create_ui_elements(self):
        # Clear previous elements
        self.buttons = []
        self.sliders = []
        
        if self.state == "setup":
          
            n_trees_slider = Slider(
                pygame.Rect(WIDTH//2 - 150, HEIGHT//2 - 150, 300, 30),
                3, 10, self.n_trees, "Number of Trees"
            )
            self.sliders.append(n_trees_slider)
            
            max_depth_slider = Slider(
                pygame.Rect(WIDTH//2 - 150, HEIGHT//2 - 100, 300, 30),
                1, 5, self.max_depth, "Max Tree Depth"
            )
            self.sliders.append(max_depth_slider)
            
 
            start_btn = Button(
                pygame.Rect(WIDTH//2 - 100, HEIGHT//2 + 300, 200, 40),
                "Start Training", "start_training", True, True
            )
            self.buttons.append(start_btn)

            about_btn = Button(
                pygame.Rect(WIDTH - 120, HEIGHT - 60, 100, 40),
                "About", "about", True, False
            )
            self.buttons.append(about_btn)

        elif self.state == "training":
          
            continue_btn = Button(
                pygame.Rect(WIDTH//2 - 100, HEIGHT - 80, 200, 40),
                "Build Next Tree", "build_next", 
                self.forest and not self.forest.build_complete, True
            )
            self.buttons.append(continue_btn)
            
            # Button to build all trees at once
            build_all_btn = Button(
                pygame.Rect(WIDTH//2 + 110, HEIGHT - 80, 180, 40),
                "Build All Trees", "build_all",
                self.forest and not self.forest.build_complete, False
            )
            self.buttons.append(build_all_btn)
            
            # Button to test tree
            test_btn = Button(
                pygame.Rect(WIDTH//2 - 290, HEIGHT - 80, 180, 40),
                "Test Forest", "test_forest",
                self.forest and self.forest.build_complete, False
            )
            self.buttons.append(test_btn)
            
            # Buttons to navigate the trees
            if self.forest and self.forest.current_tree_index > 0:
                prev_tree_btn = Button(
                    pygame.Rect(20, 150, 40, 40),
                    "◄", "prev_tree",
                    self.selected_tree > 0, False
                )
                self.buttons.append(prev_tree_btn)
                
                next_tree_btn = Button(
                    pygame.Rect(70, 150, 40, 40),
                    "►", "next_tree",
                    self.selected_tree < self.forest.current_tree_index - 1, False
                )
                self.buttons.append(next_tree_btn)

                # About button
                about_btn = Button(
                    pygame.Rect(WIDTH - 120, HEIGHT - 80, 100, 40),
                    "About", "about", True, False
                )
                self.buttons.append(about_btn)
                
                # Reset view button
                reset_view_btn = Button(
                    pygame.Rect(WIDTH - 240, HEIGHT - 80, 100, 40),
                    "Reset View", "reset_view", True, False
                )
                self.buttons.append(reset_view_btn)
        
        elif self.state == "testing":
    
            back_btn = Button(
                pygame.Rect(20, HEIGHT - 80, 180, 40),
                "Back to Forest", "back_to_viz", True, False
            )
            self.buttons.append(back_btn)
            
     
            reset_btn = Button(
                pygame.Rect(WIDTH - 200, HEIGHT - 80, 180, 40),
                "Reset Visualizer", "reset", True, False
            )
            self.buttons.append(reset_btn)
            
     
            prev_sample_btn = Button(
                pygame.Rect(WIDTH - 240, 100, 40, 40),
                "◄", "prev_sample",
                self.selected_test_sample > 0, False
            )
            self.buttons.append(prev_sample_btn)
            
            next_sample_btn = Button(
                pygame.Rect(WIDTH - 50, 100, 40, 40),
                "►", "next_sample",
                self.selected_test_sample < len(self.X_test) - 1, False
            )
            self.buttons.append(next_sample_btn)
   
            toggle_btn = Button(
                pygame.Rect(WIDTH//2 - 100, HEIGHT - 80, 200, 40),
                "Toggle Path View" if self.show_predictions else "Toggle Voting View", 
                "toggle_view", True, False
            )
            self.buttons.append(toggle_btn)
            
            # Add reset view button in testing mode too
            if self.show_predictions:
                reset_view_btn = Button(
                    pygame.Rect(WIDTH - 320, HEIGHT - 80, 100, 40),
                    "Reset View", "reset_view", True, False
                )
                self.buttons.append(reset_view_btn)


        elif self.state == "about":

            back_btn = Button(
                pygame.Rect(WIDTH//2 - 60, HEIGHT - 80, 120, 40),
                "Back", "back_from_about", True, True
            )
            self.buttons.append(back_btn)
    
    def start(self):
        """Initialize the forest with the dataset and parameters"""
        for slider in self.sliders:
            if slider.label == "Number of Trees":
                self.n_trees = int(slider.value)
            elif slider.label == "Max Tree Depth":
                self.max_depth = int(slider.value)
        

        self.forest = RandomForest(n_trees=self.n_trees, max_depth=self.max_depth, n_features=None)
        self.forest.fit(self.X_train, self.y_train)
        
   
        self.state = "training"
        self.selected_tree = 0
        
        # Reset camera transform
        self.reset_view()

        self._create_ui_elements()
    
    def build_next_tree(self):
        """Train the next tree in the forest"""
        if self.forest:
            complete = self.forest.build_next_tree()
            
            if complete:
                print(f"All {self.n_trees} trees built!")
            

            if self.forest.current_tree_index > 0:
                self.selected_tree = self.forest.current_tree_index - 1
            

            self._create_ui_elements()
    
    def build_all_trees(self):
        """Build all remaining trees at once"""
        if self.forest:
            self.forest.build_all_trees()
            
    
            if self.forest.current_tree_index > 0:
                self.selected_tree = self.forest.current_tree_index - 1
   
            self._create_ui_elements()
    
    def test_forest(self):
        """Test the forest on the test dataset"""
        if not self.forest or not self.forest.build_complete:
            return
        

        self.test_predictions, self.prediction_paths = self.forest.predict_with_viz(self.X_test)
        

        accuracy = np.sum(self.test_predictions == self.y_test) / len(self.y_test)
        print(f"Accuracy: {accuracy:.4f}")
        
      
        self.state = "testing"
        self.selected_test_sample = 0
        self.viz_step = 0
        self.show_predictions = False
        
        # Reset the view when entering testing mode
        self.reset_view()

        self._create_ui_elements()
        
    def reset_view(self):
        """Reset the camera to default position and zoom"""
        self.zoom_level = 1.0
        self.camera_offset = [0, 0]
    
    def handle_events(self):
        """Handle pygame events"""
        mouse_pos = pygame.mouse.get_pos()
   
        for button in self.buttons:
            button.update(mouse_pos)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  
            
 
            for button in self.buttons:
                action = button.handle_event(event)
                if action:
                    self._handle_button_action(action)
            
    
            for slider in self.sliders:
                if slider.handle_event(event):
                    # Update parameters based on slider values
                    if slider.label == "Number of Trees":
                        self.n_trees = int(slider.value)
                    elif slider.label == "Max Tree Depth":
                        self.max_depth = int(slider.value)
            
            # Handle zooming with mouse wheel
            if event.type == pygame.MOUSEWHEEL:
                if self._is_in_visualization_area(mouse_pos):
                    # Scale the zoom based on the wheel movement
                    zoom_change = event.y * 0.1  # Adjust sensitivity as needed
                    new_zoom = self.zoom_level + zoom_change
                    
                    # Clamp zoom level to min/max values
                    new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
                    
                    # Calculate zoom around the mouse position
                    if new_zoom != self.zoom_level:
                        mouse_x, mouse_y = mouse_pos
                        
                        # Get world coordinates of mouse before zoom
                        world_x = (mouse_x - self.camera_offset[0]) / self.zoom_level
                        world_y = (mouse_y - self.camera_offset[1]) / self.zoom_level
                        
                        # Update zoom level
                        self.zoom_level = new_zoom
                        

                        self.camera_offset[0] = mouse_x - world_x * self.zoom_level
                        self.camera_offset[1] = mouse_y - world_y * self.zoom_level
            

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    if self._is_in_visualization_area(mouse_pos) and not self._is_over_ui():
                        self.is_dragging = True
                        self.drag_start = mouse_pos
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    self.is_dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.is_dragging:
                    # Calculate drag distance
                    dx = event.pos[0] - self.drag_start[0]
                    dy = event.pos[1] - self.drag_start[1]
                    
                    # Update camera offset
                    self.camera_offset[0] += dx
                    self.camera_offset[1] += dy
                    
                    # Update drag start position for next frame
                    self.drag_start = event.pos
        
        return True  # Continue running
    
    def _is_in_visualization_area(self, mouse_pos):
        """Check if mouse is in the main visualization area (excluding UI elements)"""
        x, y = mouse_pos
        
        if self.state == "training":
            # Tree visualization area (exclude sidebar and bottom panel)
            sidebar_width = 120
            bottom_panel_height = 80
            return x > sidebar_width and y > 80 and y < HEIGHT - bottom_panel_height
        
        elif self.state == "testing" and self.show_predictions:
            # Decision path visualization area
            sidebar_width = 250
            bottom_panel_height = 80
            return x < WIDTH - sidebar_width and y > 80 and y < HEIGHT - bottom_panel_height
        
        return False
    
    def _is_over_ui(self):
        """Check if mouse is over a UI element"""
        mouse_pos = pygame.mouse.get_pos()
        
        # Check if mouse is over any button
        for button in self.buttons:
            if button.rect.collidepoint(mouse_pos):
                return True
        
        # Check if mouse is over any slider
        for slider in self.sliders:
            if slider.rect.collidepoint(mouse_pos):
                return True
                
        return False
    
    def _handle_button_action(self, action):
        """Handle button actions"""
        if action == "start_training":
            self.start()
            
        elif action == "build_next":
            self.build_next_tree()
            
        elif action == "build_all":
            self.build_all_trees()
            
        elif action == "test_forest":
            self.test_forest()
            
        elif action == "prev_tree":
            if self.selected_tree > 0:
                self.selected_tree -= 1
                self._create_ui_elements()
                
        elif action == "next_tree":
            if self.forest and self.selected_tree < self.forest.current_tree_index - 1:
                self.selected_tree += 1
                self._create_ui_elements()
                
        elif action == "prev_sample":
            if self.selected_test_sample > 0:
                self.selected_test_sample -= 1
                self.viz_step = 0
                self._create_ui_elements()
                
        elif action == "next_sample":
            if self.selected_test_sample < len(self.X_test) - 1:
                self.selected_test_sample += 1
                self.viz_step = 0
                self._create_ui_elements()
        
        elif action == "back_to_viz":
            self.state = "training"
            self.reset_view()
            self._create_ui_elements()
            
        elif action == "reset":
            self.__init__()  
        
        elif action == "toggle_view":
            self.show_predictions = not self.show_predictions
            self._create_ui_elements()

        elif action == "about":
            self.state = "about"
            self._create_ui_elements()
        
        elif action == "back_from_about":
            self.state = "setup"
            self._create_ui_elements()
            
        elif action == "reset_view":
            self.reset_view()
    
    def update(self):
        """Update visualizer state"""
        current_time = pygame.time.get_ticks()
        
        if self.state == "testing" and self.test_predictions is not None:
            if current_time - self.animation_timer > self.animation_speed:
                self.animation_timer = current_time
                
                max_steps = 0
                for tree_idx in range(len(self.forest.trees)):
                    if tree_idx < len(self.prediction_paths):
                        path_length = len(self.prediction_paths[tree_idx][self.selected_test_sample])
                        max_steps = max(max_steps, path_length)
                
                if self.viz_step < max_steps:
                    self.viz_step += 1
    
    def draw(self, screen):
        """Draw the current state of the visualizer"""
        screen.fill(COLORS['background'])
        
        # Draw a header area
        header_rect = pygame.Rect(0, 0, WIDTH, 80)
        pygame.draw.rect(screen, COLORS['panel'], header_rect)
        pygame.draw.line(screen, COLORS['node_border'], (0, 80), (WIDTH, 80), 1)
        
        # Draw application title
        title_surface = get_font(28, bold=True).render("Random Forest Visualizer", True, COLORS['text'])
        title_rect = title_surface.get_rect(midtop=(WIDTH//2, 20))
        screen.blit(title_surface, title_rect)
        
        # Create the zoom indicator text
        if (self.state == "training" or (self.state == "testing" and self.show_predictions)) and self.zoom_level != 1.0:
            zoom_text = f"Zoom: {self.zoom_level:.1f}x"
            zoom_surface = get_font(14).render(zoom_text, True, COLORS['text_secondary'])
            zoom_rect = zoom_surface.get_rect(topright=(WIDTH - 20, 85))
            screen.blit(zoom_surface, zoom_rect)
            
            # Hint for panning
            hint_text = "Drag to pan, scroll to zoom"
            hint_surface = get_font(12).render(hint_text, True, COLORS['text_secondary'])
            hint_rect = hint_surface.get_rect(topright=(WIDTH - 20, 105))
            screen.blit(hint_surface, hint_rect)

        if self.state == "setup":
            self._draw_setup(screen)
        elif self.state == "training":
            self._draw_training(screen)
        elif self.state == "testing":
            self._draw_testing(screen)
        elif self.state == "about":
            self._draw_about(screen)
        
        # Draw all UI elements
        for slider in self.sliders:
            slider.draw(screen)
            
        for button in self.buttons:
            button.draw(screen)
    
    def _draw_setup(self, screen):
        """Draw setup screen"""

        instructions = [
            "Configure your Random Forest parameters:",
         
        ]
        
        for i, line in enumerate(instructions):
            y_offset = 150 + i * 30
            text_surface = get_font(16).render(line, True, COLORS['text'])
            text_rect = text_surface.get_rect(center=(WIDTH//2, y_offset))
            screen.blit(text_surface, text_rect)
        

        self._draw_dataset_preview(screen, 450)
    
    def _draw_dataset_preview(self, screen, y_start):
        """Draw a preview of the dataset"""
 
        plot_width, plot_height = 600, 200
        plot_x = (WIDTH - plot_width) // 2
        plot_y = y_start
        
  
        panel_rect = pygame.Rect(plot_x - 20, plot_y - 40, plot_width + 40, plot_height + 80)
        pygame.draw.rect(screen, COLORS['panel'], panel_rect, border_radius=5)
        

        title_surface = get_font(18, bold=True).render("Iris Dataset Preview", True, COLORS['text'])
        title_rect = title_surface.get_rect(center=(WIDTH//2, plot_y - 20))
        screen.blit(title_surface, title_rect)
        

        plot_rect = pygame.Rect(plot_x, plot_y, plot_width, plot_height)
        pygame.draw.rect(screen, COLORS['background'], plot_rect)

        for i in range(5):

            x = plot_x + i * plot_width // 4
            pygame.draw.line(screen, COLORS['grid'], 
                           (x, plot_y), (x, plot_y + plot_height), 1)
            
        
            y = plot_y + i * plot_height // 4
            pygame.draw.line(screen, COLORS['grid'], 
                           (plot_x, y), (plot_x + plot_width, y), 1)
        

        pygame.draw.line(screen, COLORS['text'], 
                       (plot_x, plot_y + plot_height), 
                       (plot_x + plot_width, plot_y + plot_height), 2)
        pygame.draw.line(screen, COLORS['text'], 
                       (plot_x, plot_y), 
                       (plot_x, plot_y + plot_height), 2)
        

        x_label = self.feature_names[0]
        y_label = self.feature_names[1]
        
    
        x_label = x_label.split(' (')[0]
        y_label = y_label.split(' (')[0]
        
        if len(x_label) > 20:
            x_label = x_label[:17] + "..."
        if len(y_label) > 20:
            y_label = y_label[:17] + "..."
        
        x_surface = get_font(14).render(x_label, True, COLORS['text'])
        x_rect = x_surface.get_rect(center=(plot_x + plot_width//2, plot_y + plot_height + 25))
        screen.blit(x_surface, x_rect)
        
        y_surface = get_font(14).render(y_label, True, COLORS['text'])
        y_surface = pygame.transform.rotate(y_surface, 90)
        y_rect = y_surface.get_rect(center=(plot_x - 35, plot_y + plot_height//2))
        screen.blit(y_surface, y_rect)
        

        x_min, x_max = np.min(self.X[:, 0]), np.max(self.X[:, 0])
        y_min, y_max = np.min(self.X[:, 1]), np.max(self.X[:, 1])
        

        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        

        for i in range(len(self.X)):

            x_scaled = plot_x + (self.X[i, 0] - x_min) / (x_max - x_min) * plot_width
            y_scaled = plot_y + plot_height - (self.X[i, 1] - y_min) / (y_max - y_min) * plot_height
            

            point_color = [
                COLORS['feature_1'], 
                COLORS['feature_2'], 
                COLORS['feature_3']
            ][self.y[i]]
            
            # Draw point
            pygame.draw.circle(screen, point_color, (int(x_scaled), int(y_scaled)), 4)
        
        # Draw legend
        legend_x = plot_x + plot_width - 120
        legend_y = plot_y + 20
        
        legend_title = get_font(14, bold=True).render("Classes:", True, COLORS['text'])
        screen.blit(legend_title, (legend_x, legend_y - 25))
        
        for i, name in enumerate(self.target_names):

            pygame.draw.circle(screen, 
                             [COLORS['feature_1'], COLORS['feature_2'], COLORS['feature_3']][i], 
                             (legend_x + 7, legend_y + i * 25 + 7), 7)
            

            label_surface = get_font(14).render(name, True, COLORS['text'])
            screen.blit(label_surface, (legend_x + 20, legend_y + i * 25))
    
    def _draw_training(self, screen):
        """Draw training visualization"""
        if not self.forest:
            return
            
        sidebar_width = 120
        sidebar_rect = pygame.Rect(0, 80, sidebar_width, HEIGHT - 80)
        pygame.draw.rect(screen, COLORS['panel'], sidebar_rect)
        pygame.draw.line(screen, COLORS['node_border'], 
                        (sidebar_width, 80), (sidebar_width, HEIGHT), 1)
        
        sidebar_title = get_font(16, bold=True).render("Trees", True, COLORS['text'])
        title_rect = sidebar_title.get_rect(center=(sidebar_width//2, 90))
        screen.blit(sidebar_title, title_rect) 
        
        n_built_trees = self.forest.current_tree_index
        progress_text = f"Built: {n_built_trees}/{self.forest.n_trees}"
        progress_surface = get_font(14).render(progress_text, True, COLORS['text'])
        progress_rect = progress_surface.get_rect(center=(sidebar_width//2, 110))
        screen.blit(progress_surface, progress_rect)
        
        bar_width = 80
        bar_height = 8
        bar_x = (sidebar_width - bar_width) // 2
        bar_y = 125
        
        pygame.draw.rect(screen, COLORS['background'], (bar_x, bar_y, bar_width, bar_height), border_radius=4)

        filled_width = int(bar_width * self.forest.build_progress / 100)
        if filled_width > 0:
            fill_rect = pygame.Rect(bar_x, bar_y, filled_width, bar_height)
            pygame.draw.rect(screen, COLORS['primary'], fill_rect, border_radius=4)

        pygame.draw.rect(screen, COLORS['node_border'], (bar_x, bar_y, bar_width, bar_height), width=1, border_radius=4)
        
        tree_y = 200
        for i in range(n_built_trees):
            tree_rect = pygame.Rect(10, tree_y + i * 40, sidebar_width - 20, 30)
            
            if i == self.selected_tree:
                pygame.draw.rect(screen, COLORS['primary_light'], tree_rect, border_radius=4)
                pygame.draw.rect(screen, COLORS['primary_dark'], tree_rect, width=1, border_radius=4)
            else:
                pygame.draw.rect(screen, COLORS['background'], tree_rect, border_radius=4)
                pygame.draw.rect(screen, COLORS['node_border'], tree_rect, width=1, border_radius=4)
            
            tree_label = f"Tree {i + 1}"
            tree_surface = get_font(14).render(tree_label, True, COLORS['text'])
            tree_rect = tree_surface.get_rect(center=(sidebar_width//2, tree_y + i * 40 + 15))
            screen.blit(tree_surface, tree_rect)
        
        content_x = sidebar_width + 20
        content_width = WIDTH - sidebar_width - 40
        
        if self.selected_tree < self.forest.current_tree_index:
            param_text = f"Depth: {self.max_depth} | Features: {self.forest.trees[self.selected_tree].n_features}"
            param_surface = get_font(14).render(param_text, True, COLORS['text_secondary'])
            param_rect = param_surface.get_rect(topleft=(content_x, 120))
            screen.blit(param_surface, param_rect)
            
            # Create a clipping rect for the tree visualization area
            viz_rect = pygame.Rect(sidebar_width, 80, WIDTH - sidebar_width, HEIGHT - 160)
            orig_clip = screen.get_clip()
            screen.set_clip(viz_rect)
            
            # Visualize the tree with the current zoom level and offset
            self.forest.visualize_forest_with_transform(screen, self.feature_names, self.selected_tree, 
                                                self.zoom_level, self.camera_offset)
            
            # Restore original clipping
            screen.set_clip(orig_clip)
        else:
            message = "Build your first tree to begin visualization"
            msg_surface = get_font(18).render(message, True, COLORS['text_secondary'])
            msg_rect = msg_surface.get_rect(center=(content_x + content_width//2, HEIGHT//2))
            screen.blit(msg_surface, msg_rect)
    
    def _draw_testing(self, screen):
        """Draw testing visualization"""
        if not self.forest or self.test_predictions is None:
            return
            

        accuracy = np.sum(self.test_predictions == self.y_test) / len(self.y_test) * 100
        accuracy_text = f"Accuracy: {accuracy:.1f}%"
        accuracy_surface = get_font(18, bold=True).render(accuracy_text, True, COLORS['text'])
        accuracy_rect = accuracy_surface.get_rect(center=(WIDTH//2, 120))
        screen.blit(accuracy_surface, accuracy_rect)
        

        sidebar_width = 250
        sidebar_rect = pygame.Rect(WIDTH - sidebar_width, 80, sidebar_width, HEIGHT - 80)
        pygame.draw.rect(screen, COLORS['panel'], sidebar_rect)
        pygame.draw.line(screen, COLORS['node_border'], 
                        (WIDTH - sidebar_width, 80), (WIDTH - sidebar_width, HEIGHT), 1)
        

        sample_title = get_font(16, bold=True).render("Sample Information", True, COLORS['text'])
        title_rect = sample_title.get_rect(center=(WIDTH - sidebar_width//2, 110))
        screen.blit(sample_title, title_rect)
        

        sample_text = f"Sample {self.selected_test_sample + 1} of {len(self.X_test)}"
        sample_surface = get_font(14).render(sample_text, True, COLORS['text'])
        sample_rect = sample_surface.get_rect(center=(WIDTH - sidebar_width//2, 140))
        screen.blit(sample_surface, sample_rect)
        

        x_sample = self.X_test[self.selected_test_sample]
        feature_y = 180
        for i, (feature, value) in enumerate(zip(self.feature_names, x_sample)):
            feature_name = feature.split(' (')[0]
            if len(feature_name) > 15:
                feature_name = feature_name[:15] + "..."
                

            feature_text = f"{feature_name}:"
            feature_surface = get_font(14).render(feature_text, True, COLORS['text_secondary'])
            screen.blit(feature_surface, (WIDTH - sidebar_width + 15, feature_y + i * 30))
            
            value_text = f"{value:.2f}"
            value_surface = get_font(14, bold=True).render(value_text, True, COLORS['text'])
            value_rect = value_surface.get_rect(
                midright=(WIDTH - 15, feature_y + i * 30 + 9))
            screen.blit(value_surface, value_rect)
            

        result_y = feature_y + 4 * 30 + 20
        pygame.draw.line(screen, COLORS['node_border'], 
                        (WIDTH - sidebar_width + 10, result_y - 10), 
                        (WIDTH - 10, result_y - 10), 1)
        

        true_class = self.y_test[self.selected_test_sample]
        pred_class = self.test_predictions[self.selected_test_sample]
        
        true_label = get_font(14).render("True class:", True, COLORS['text_secondary'])
        screen.blit(true_label, (WIDTH - sidebar_width + 15, result_y))
        
        true_value = get_font(14, bold=True).render(self.target_names[true_class], True, COLORS['text'])
        true_rect = true_value.get_rect(midright=(WIDTH - 15, result_y + 9))
        screen.blit(true_value, true_rect)
        
        pred_label = get_font(14).render("Prediction:", True, COLORS['text_secondary'])
        screen.blit(pred_label, (WIDTH - sidebar_width + 15, result_y + 30))
        
        # Color based on correctness
        pred_color = COLORS['correct'] if true_class == pred_class else COLORS['incorrect']
        
        pred_value = get_font(14, bold=True).render(self.target_names[pred_class], True, pred_color)
        pred_rect = pred_value.get_rect(midright=(WIDTH - 15, result_y + 30 + 9))
        screen.blit(pred_value, pred_rect)
        

        if self.show_predictions:
            self._draw_decision_paths(screen)
        else:
            self._draw_voting_info(screen)
    
    def _draw_decision_paths(self, screen):
        """Draw the decision paths through trees for the selected sample"""
        # Draw title
        path_title = "Decision Path Animation"
        path_surface = get_font(18, bold=True).render(path_title, True, COLORS['text'])
        path_rect = path_surface.get_rect(topleft=(20, 160))
        screen.blit(path_surface, path_rect)
        
        current_tree_idx = self.selected_tree
        
        if (current_tree_idx < len(self.forest.trees) and 
            current_tree_idx < len(self.prediction_paths) and
            self.selected_test_sample < len(self.prediction_paths[current_tree_idx])):
            
            tree = self.forest.trees[current_tree_idx]
            path = self.prediction_paths[current_tree_idx][self.selected_test_sample]
            
            # Create a clipping rect for the tree visualization area
            viz_rect = pygame.Rect(0, 160, WIDTH - 250, HEIGHT - 240)
            orig_clip = screen.get_clip()
            screen.set_clip(viz_rect)
            
            # Draw the tree with camera transform
            tree.visualize_tree_with_transform(screen, self.feature_names, 
                                            self.zoom_level, self.camera_offset)
            
            # Draw path highlighting with camera transform
            for i in range(min(self.viz_step, len(path))):
                node = path[i]
                if node and hasattr(node, 'pos'):
                    # Apply zoom and offset to node position
                    transformed_pos = (
                        int(node.pos[0] * self.zoom_level + self.camera_offset[0]),
                        int(node.pos[1] * self.zoom_level + self.camera_offset[1])
                    )
                    transformed_radius = int(node.radius * self.zoom_level)
                    
                    # Draw highlight around the node
                    highlight_radius = transformed_radius + int(5 * self.zoom_level)
                    pygame.draw.circle(screen, COLORS['highlight'], transformed_pos, highlight_radius, 3)
                    
                    # Draw decision path line
                    if i == self.viz_step - 1 and not node.is_leaf_node() and i+1 < len(path):
                        next_node = path[i+1]
                        
                        if hasattr(next_node, 'pos'):
                            # Apply zoom and offset to next node position
                            next_transformed_pos = (
                                int(next_node.pos[0] * self.zoom_level + self.camera_offset[0]),
                                int(next_node.pos[1] * self.zoom_level + self.camera_offset[1])
                            )
                            
                            # Draw line between current and next node
                            pygame.draw.line(screen, COLORS['highlight'], 
                                        transformed_pos, next_transformed_pos, 
                                        max(2, int(4 * self.zoom_level)))
                            
                            # Draw decision text
                            x_sample = self.X_test[self.selected_test_sample]
                            feature_val = x_sample[node.feature_idx]
                            
                            decision_text = f"Feature {node.feature_idx} = {feature_val:.2f}"
                            if feature_val <= node.threshold:
                                decision_text += f" ≤ {node.threshold:.2f} (Left)"
                            else:
                                decision_text += f" > {node.threshold:.2f} (Right)"
                            
                            # Scale text size based on zoom
                            font_size = max(8, int(12 * self.zoom_level))
                            decision_surface = get_font(font_size).render(decision_text, True, COLORS['background'])
                            
                            # Position at midpoint of the line
                            mid_x = (transformed_pos[0] + next_transformed_pos[0]) // 2
                            mid_y = (transformed_pos[1] + next_transformed_pos[1]) // 2 - int(15 * self.zoom_level)
                            decision_rect = decision_surface.get_rect(center=(mid_x, mid_y))
                            
                            # Draw background for text
                            bg_rect = decision_rect.copy()
                            bg_rect.inflate_ip(10, 6)
                            pygame.draw.rect(screen, COLORS['highlight'], bg_rect, border_radius=4)
                            
                            # Draw text
                            screen.blit(decision_surface, decision_rect)
                    
                    # Show prediction for leaf nodes
                    if node.is_leaf_node() and i == len(path) - 1:
                        pred_text = f"Tree predicts: {self.target_names[node.value]}"
                        
                        # Scale text size based on zoom
                        font_size = max(10, int(14 * self.zoom_level))
                        pred_surface = get_font(font_size, bold=True).render(pred_text, True, COLORS['highlight'])
                        
                        pred_rect = pred_surface.get_rect(center=(
                            transformed_pos[0], 
                            transformed_pos[1] + transformed_radius + int(25 * self.zoom_level)
                        ))
                        screen.blit(pred_surface, pred_rect)
            
            # Restore original clipping
            screen.set_clip(orig_clip)
    
    def _draw_voting_info(self, screen):
        """Draw information about how the trees vote for the final prediction"""
        votes = {}  # Dictionary to count votes
        

        for tree_idx, tree in enumerate(self.forest.trees):
            if tree_idx < len(self.prediction_paths):
                # Get the leaf node from the path
                path = self.prediction_paths[tree_idx][self.selected_test_sample]
                if path and path[-1].is_leaf_node():
                    prediction = path[-1].value
                    tree_name = f"Tree {tree_idx + 1}"
                    
                    if prediction not in votes:
                        votes[prediction] = []
                    votes[prediction].append(tree_name)
        

        vote_title = "Random Forest Voting"
        vote_surface = get_font(20, bold=True).render(vote_title, True, COLORS['text'])
        vote_rect = vote_surface.get_rect(topleft=(30, 160))
        screen.blit(vote_surface, vote_rect)
        

        chart_x = 50
        chart_width = WIDTH - chart_x - 300
        chart_y = 220
        bar_height = 70
        spacing = 20
        

        total_votes = sum(len(trees) for trees in votes.values())
        

        sorted_classes = sorted(votes.keys(), key=lambda k: len(votes[k]), reverse=True)
        
        for i, class_idx in enumerate(sorted_classes):
            class_name = self.target_names[class_idx]
            vote_count = len(votes[class_idx])
            percentage = (vote_count / total_votes) * 100
            
        
            bar_rect = pygame.Rect(chart_x, chart_y + i * (bar_height + spacing), chart_width, bar_height)
            pygame.draw.rect(screen, COLORS['panel'], bar_rect, border_radius=5)
            

            fill_width = int(chart_width * vote_count / total_votes)
            if fill_width > 0:
                fill_rect = pygame.Rect(chart_x, chart_y + i * (bar_height + spacing), fill_width, bar_height)
                
              
                if class_idx == self.test_predictions[self.selected_test_sample]:
           
                    fill_color = COLORS['accent']
                else:
                
                    class_colors = [COLORS['feature_1'], COLORS['feature_2'], COLORS['feature_3']]
                    fill_color = class_colors[class_idx % len(class_colors)]
                
                pygame.draw.rect(screen, fill_color, fill_rect, border_radius=5)
            
      
            class_text = f"{class_name}: {vote_count} votes ({percentage:.1f}%)"
            class_surface = get_font(16, bold=True).render(class_text, True, COLORS['text'])
            screen.blit(class_surface, (chart_x + 10, chart_y + i * (bar_height + spacing) + 10))
            

            tree_text = ", ".join(votes[class_idx])
            if len(tree_text) > 50:
                tree_text = tree_text[:47] + "..."
            tree_surface = get_font(12, bold=True).render(tree_text, True, COLORS['background'])
            screen.blit(tree_surface, (chart_x + 10, chart_y + i * (bar_height + spacing) + bar_height - 30))
        

        pred_class = self.test_predictions[self.selected_test_sample]
        result_y = chart_y + len(sorted_classes) * (bar_height + spacing) + 20
        
        result_box = pygame.Rect(chart_x, result_y, 300, 60)
        pygame.draw.rect(screen, COLORS['panel'], result_box, border_radius=5)
        
        result_label = get_font(14).render("Final Forest Prediction:", True, COLORS['text_secondary'])
        screen.blit(result_label, (chart_x + 15, result_y + 10))
        

        true_class = self.y_test[self.selected_test_sample]
        result_color = COLORS['correct'] if true_class == pred_class else COLORS['incorrect']
        
        result_text = self.target_names[pred_class]
        result_surface = get_font(24, bold=True).render(result_text, True, result_color)
        screen.blit(result_surface, (chart_x + 15, result_y + 30))
        
   
        icon_x = chart_x + 200
        icon_y = result_y + 30
        if true_class == pred_class:

            points = [(icon_x - 10, icon_y), (icon_x - 5, icon_y + 10), (icon_x + 10, icon_y - 10)]
            pygame.draw.lines(screen, COLORS['correct'], False, points, 3)
        else:
            # Draw X
            pygame.draw.line(screen, COLORS['incorrect'], 
                           (icon_x - 10, icon_y - 10), (icon_x + 10, icon_y + 10), 3)
            pygame.draw.line(screen, COLORS['incorrect'], 
                           (icon_x - 10, icon_y + 10), (icon_x + 10, icon_y - 10), 3)
            
    def _draw_about(self, screen):
        """Draw the about screen with developer credits"""
 
        about_panel = pygame.Rect(WIDTH//2 - 300, HEIGHT//2 - 200, 600, 500)
        pygame.draw.rect(screen, COLORS['panel'], about_panel, border_radius=10)
        pygame.draw.rect(screen, COLORS['node_border'], about_panel, width=2, border_radius=10)
        
     
        about_title = get_font(30, bold=True).render("About", True, COLORS['text'])
        about_title_rect = about_title.get_rect(center=(WIDTH//2, HEIGHT//2 - 150))
        screen.blit(about_title, about_title_rect)
        

        description = "Random Forest Visualizer is an interactive tool to visualize"
        description2 = "how Random Forests work, from training to prediction."
        
        desc_surface = get_font(16).render(description, True, COLORS['text'])
        desc_rect = desc_surface.get_rect(center=(WIDTH//2, HEIGHT//2 - 80))
        screen.blit(desc_surface, desc_rect)
        
        desc_surface2 = get_font(16).render(description2, True, COLORS['text'])
        desc_rect2 = desc_surface2.get_rect(center=(WIDTH//2, HEIGHT//2 - 50))
        screen.blit(desc_surface2, desc_rect2)
        
    
        pygame.draw.line(screen, COLORS['node_border'], 
                        (about_panel.left + 100, HEIGHT//2), 
                        (about_panel.right - 100, HEIGHT//2), 2)
        

        dev_text = get_font(22, bold=True).render("Developed by", True, COLORS['text'])
        dev_rect = dev_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 40))
        screen.blit(dev_text, dev_rect)
        
 
        name1 = get_font(20).render("Bishal Lamichhane (078BCT035)", True, COLORS['primary_dark'])
        name1_rect = name1.get_rect(center=(WIDTH//2, HEIGHT//2 + 80))
        screen.blit(name1, name1_rect)
        
        name2 = get_font(20).render("Bipin Bahsyal (078BCT033)", True, COLORS['primary_dark'])
        name2_rect = name2.get_rect(center=(WIDTH//2, HEIGHT//2 + 120))
        screen.blit(name2, name2_rect)

        name4 = get_font(20).render("Ayush KC (078BCT025)", True, COLORS['primary_dark'])
        name4_rect = name4.get_rect(center=(WIDTH//2, HEIGHT//2 + 160))
        screen.blit(name4, name4_rect)
        
    
        date_text = get_font(14).render("© 2025", True, COLORS['text_secondary'])
        date_rect = date_text.get_rect(center=(WIDTH//2, about_panel.bottom - 40))
        screen.blit(date_text, date_rect)