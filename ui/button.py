import pygame
from utils import COLORS, get_font

class Button:
    def __init__(self, rect, text, action, enabled=True, primary=False):
        self.rect = rect
        self.text = text
        self.action = action
        self.enabled = enabled
        self.primary = primary  # Primary buttons use accent color
        self.hover = False
    
    def draw(self, screen):
  
        if not self.enabled:
            bg_color = COLORS['button_disabled']
            text_color = COLORS['text_secondary']
        elif self.primary:
            bg_color = COLORS['accent_hover'] if self.hover else COLORS['accent']
            text_color = COLORS['background']
        else:
            bg_color = COLORS['primary_light'] if self.hover else COLORS['primary']
            text_color = COLORS['background']
        
       
        pygame.draw.rect(screen, bg_color, self.rect, border_radius=4)
        
       
        if self.enabled:
            border_color = COLORS['primary_dark'] if not self.primary else COLORS['accent_hover']
            pygame.draw.rect(screen, border_color, self.rect, width=1, border_radius=4)
        
     
        font = get_font(16, bold=True) if self.primary else get_font(16)
        text_surface = font.render(self.text, True, text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
    
    def update(self, mouse_pos):
    
        self.hover = self.enabled and self.rect.collidepoint(mouse_pos)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: 
            if self.enabled and self.rect.collidepoint(event.pos):
                return self.action
        return None