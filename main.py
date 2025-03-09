import pygame
import sys
from utils import WIDTH, HEIGHT
from visualizer import RandomForestVisualizer

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Random Forest Visualizer")
    
    clock = pygame.time.Clock()
    visualizer = RandomForestVisualizer()
    
    running = True
    while running:

        running = visualizer.handle_events()
        
  
        visualizer.update()
        
        # Draw
        visualizer.draw(screen)
        
      
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()