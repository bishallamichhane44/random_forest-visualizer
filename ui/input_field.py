import pygame
from utils import COLORS, get_font

class InputField:
    def __init__(self, rect, initial_value=0.0, label="", min_value=None, max_value=None):
        self.rect = rect
        self.value = initial_value  
        self.text = str(initial_value)  # Text representation for editing
        self.label = label
        self.min_value = min_value
        self.max_value = max_value
        
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0
        self.cursor_blink_speed = 500  # milliseconds
        
        # Colors
        self.background_color = COLORS['background']
        self.border_color = COLORS['node_border']
        self.active_border_color = COLORS['primary']
        self.text_color = COLORS['text']
        
    def handle_event(self, event):
        """Handle pygame events"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the input field was clicked
            if self.rect.collidepoint(event.pos):
                self.active = True
                return True
            else:
                self.active = False
                self._validate_and_update()
                return False
                
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                # Enter key submits the value
                self.active = False
                self._validate_and_update()
                return True
                
            elif event.key == pygame.K_BACKSPACE:
                # Delete last character
                self.text = self.text[:-1]
                return True
                
            else:
                # Add character if it's valid
                if event.unicode.isdigit():  # Allow digits
                    self.text += event.unicode
                    return True
                elif event.unicode == '.' and '.' not in self.text:  # Allow one decimal point
                    self.text += event.unicode
                    return True
                elif event.unicode == '-' and not self.text:  # Allow minus sign at start
                    self.text += event.unicode
                    return True
                    
        return False
    
    def _validate_and_update(self):
        """Validate the current text and update the value"""
        try:
            if self.text:
                # Convert to float
                new_value = float(self.text)
                
                # Apply min/max constraints
                if self.min_value is not None:
                    new_value = max(new_value, self.min_value)
                if self.max_value is not None:
                    new_value = min(new_value, self.max_value)
                    
                self.value = new_value
                self.text = f"{new_value:.2f}" 
            else:
                # If empty, set to min value or 0
                self.value = self.min_value if self.min_value is not None else 0.0
                self.text = f"{self.value:.2f}"
                
        except ValueError:
            # If invalid input, revert to current value
            self.text = f"{self.value:.2f}"
    
    def update(self):
        """Update the input field state, like cursor blinking"""
        current_time = pygame.time.get_ticks()
        
        if current_time - self.cursor_timer > self.cursor_blink_speed:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = current_time
    
    def draw(self, screen):
        """Draw the input field"""
        # Draw the background
        pygame.draw.rect(screen, self.background_color, self.rect, border_radius=4)
        
        # Draw the border with appropriate color based on active state
        border_color = self.active_border_color if self.active else self.border_color
        pygame.draw.rect(screen, border_color, self.rect, width=2, border_radius=4)
        
        # Draw the text
        display_text = self.text
        if self.active and self.cursor_visible:
            display_text += "|"  # Add blinking cursor
            
        text_surface = get_font(14).render(display_text, True, self.text_color)
        text_rect = text_surface.get_rect(midleft=(self.rect.x + 10, self.rect.centery))
        screen.blit(text_surface, text_rect)