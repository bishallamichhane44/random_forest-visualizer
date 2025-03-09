import pygame
from utils import COLORS, get_font

class Slider:
    def __init__(self, rect, min_val, max_val, initial_val, label, value_format="{:.0f}"):
        self.rect = rect
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.value_format = value_format
        self.dragging = False
        self.handle_radius = 10
    
    def draw(self, screen):

        track_rect = pygame.Rect(self.rect.x, self.rect.y + self.rect.height//2 - 2, self.rect.width, 4)
        pygame.draw.rect(screen, COLORS['node_border'], track_rect, border_radius=2)
        

        fill_width = int((self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width)
        fill_rect = pygame.Rect(self.rect.x, self.rect.y + self.rect.height//2 - 2, fill_width, 4)
        pygame.draw.rect(screen, COLORS['primary'], fill_rect, border_radius=2)
        

        handle_x = self.rect.x + fill_width
        handle_y = self.rect.y + self.rect.height // 2
        pygame.draw.circle(screen, COLORS['background'], (handle_x, handle_y), self.handle_radius)
        pygame.draw.circle(screen, COLORS['primary'], (handle_x, handle_y), self.handle_radius, 2)
        

        label_surface = get_font(14).render(self.label, True, COLORS['text'])
        screen.blit(label_surface, (self.rect.x, self.rect.y - 25))
        

        value_text = self.value_format.format(self.value)
        value_surface = get_font(14).render(value_text, True, COLORS['text'])
        value_rect = value_surface.get_rect(midright=(self.rect.right, self.rect.y - 10))
        screen.blit(value_surface, value_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:

            handle_x = self.rect.x + int((self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width)
            handle_y = self.rect.y + self.rect.height // 2
            handle_rect = pygame.Rect(handle_x - self.handle_radius, handle_y - self.handle_radius, 
                                      2 * self.handle_radius, 2 * self.handle_radius)
            if handle_rect.collidepoint(event.pos):
                self.dragging = True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        
        elif event.type == pygame.MOUSEMOTION and self.dragging:
       
            pos_ratio = max(0, min(1, (event.pos[0] - self.rect.x) / self.rect.width))
            self.value = self.min_val + pos_ratio * (self.max_val - self.min_val)
            return True  
        
        return False