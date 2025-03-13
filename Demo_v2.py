import pygame
import math
import random
import sys

def generate_flow_field():
    for y in range(rows):
        for x in range(cols):
            angle = random.uniform(0, 2 * math.pi)  # Losowy kąt
            flow_field[y][x] = (math.cos(angle), math.sin(angle))  # Wektor jednostkowy

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Siatka")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

GRID_SIZE = 20
rows = HEIGHT // GRID_SIZE
cols = WIDTH // GRID_SIZE
flow_field = [[(0, 0) for _ in range(cols)] for _ in range(rows)]
generate_flow_field()

clock = pygame.time.Clock()
while True:
    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    for y in range(rows):
        for x in range(cols):
            start_x = x * GRID_SIZE
            start_y = y * GRID_SIZE
            end_x = start_x + flow_field[y][x][0] * GRID_SIZE // 2
            end_y = start_y + flow_field[y][x][1] * GRID_SIZE // 2
            pygame.draw.line(screen, WHITE, (start_x, start_y), (end_x, end_y), 1)

    pygame.display.flip()
    clock.tick(60)
