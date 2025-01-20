import pygame
import sys

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Demo")
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
rect_size = 30
rect_pos = [WIDTH // 2, HEIGHT // 2] 
rect_velocity = [0, 0] 
dragging = False
prev_mouse_pos = None
friction = 0.98  # Współczynnik tarcia, aby zatrzymać obiekt
bounce_factor = -0.8  # Współczynnik odbicia (negatywna zmiana kierunku)
clock = pygame.time.Clock()
while True:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            # Sprawdź, czy kliknięcie jest wewnątrz kwadratu
            if rect_pos[0] < mouse_pos[0] < rect_pos[0] + rect_size and \
               rect_pos[1] < mouse_pos[1] < rect_pos[1] + rect_size:
                dragging = True
                prev_mouse_pos = mouse_pos

        if event.type == pygame.MOUSEBUTTONUP:
            if dragging:
                dragging = False

    if dragging:
        mouse_pos = pygame.mouse.get_pos()
        if prev_mouse_pos:
            rect_velocity = [
                mouse_pos[0] - prev_mouse_pos[0],
                mouse_pos[1] - prev_mouse_pos[1],
            ]
        rect_pos[0] = mouse_pos[0] - rect_size // 2
        rect_pos[1] = mouse_pos[1] - rect_size // 2
        prev_mouse_pos = mouse_pos
    else:
        # Symulacja ślizgania
        rect_pos[0] += rect_velocity[0]
        rect_pos[1] += rect_velocity[1]
        # Zastosuj tarcie
        rect_velocity[0] *= friction
        rect_velocity[1] *= friction

        # Odbicie od ścian
        if rect_pos[0] <= 0:  # Lewa ściana
            rect_pos[0] = 0
            rect_velocity[0] *= bounce_factor
        if rect_pos[0] + rect_size >= WIDTH:  # Prawa ściana
            rect_pos[0] = WIDTH - rect_size
            rect_velocity[0] *= bounce_factor
        if rect_pos[1] <= 0:  # Górna ściana
            rect_pos[1] = 0
            rect_velocity[1] *= bounce_factor
        if rect_pos[1] + rect_size >= HEIGHT:  # Dolna ściana
            rect_pos[1] = HEIGHT - rect_size
            rect_velocity[1] *= bounce_factor

        # Zatrzymaj, jeśli prędkość jest bardzo mała
        if abs(rect_velocity[0]) < 0.1:
            rect_velocity[0] = 0
        if abs(rect_velocity[1]) < 0.1:
            rect_velocity[1] = 0
    pygame.draw.rect(screen, BLUE, (*rect_pos, rect_size, rect_size))
    pygame.display.flip()
    clock.tick(60)
