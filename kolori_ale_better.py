from tkinter import *
import pygame
import numpy as np
import math
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random
import sys

pygame.init()
screen = pygame.display.set_mode((512, 512))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)
font2 = pygame.font.Font(None, 24)

current_screen = "menu"

# Constants for the simulation
N = 64  # Grid size (N x N)
dt = 0.1  # Time step
diff = 0.0001  # Diffusion rate
visc = 0.0001 # Viscosity

# Initialize density and velocity fields
size = (N + 2, N + 2)
u = np.zeros(size)  # Horizontal velocity
v = np.zeros(size)  # Vertical velocity
u_prev = np.zeros(size)
v_prev = np.zeros(size)
dens = np.zeros(size)  # Density field
dens_prev = np.zeros(size)

def color1(d):
    return (int(d * 2 * 255), 0, 0) if d < 0.5 else (255, int((d - 0.5) * 2 * 255), 0)

def color2(d):
    return (
        max(0, min(255, int((math.sin(d * math.pi) * 255)))),
        max(0, min(255, int((math.sin(d * math.pi * 1.5) * 255) if d > 0.2 else 0))),
        max(0, min(255, int((math.cos(d * math.pi * 1.5) * 255) if d > 0.5 else 0)))
    )

def color3(d):
    return (
        max(0, min(255, int((math.sin(d * math.pi) * 255)))),
        max(0, min(255, int((math.sin(d * math.pi * 2) * 127 + 128)))),
        max(0, min(255, int((math.cos(d * math.pi) * 255))))
    )

# Add sources to the field
def add_source(x, s, dt):
    x += dt * s

# Set boundary conditions
def set_bnd(b, x):
    x[0, 1:N+1] = -x[1, 1:N+1] if b == 1 else x[1, 1:N+1]
    x[N+1, 1:N+1] = -x[N, 1:N+1] if b == 1 else x[N, 1:N+1]
    x[1:N+1, 0] = -x[1:N+1, 1] if b == 2 else x[1:N+1, 1]
    x[1:N+1, N+1] = -x[1:N+1, N] if b == 2 else x[1:N+1, N]
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, N+1] = 0.5 * (x[1, N+1] + x[0, N])
    x[N+1, 0] = 0.5 * (x[N, 0] + x[N+1, 1])
    x[N+1, N+1] = 0.5 * (x[N, N+1] + x[N+1, N])

# Diffuse the field using Gauss-Seidel relaxation
def diffuse(b, x, x0, diff, dt):
    a = dt * diff * N * N
    for _ in range(20):  # Iterative solver
        x[1:N+1, 1:N+1] = (x0[1:N+1, 1:N+1] + a * (
            x[0:N, 1:N+1] + x[2:N+2, 1:N+1] + 
            x[1:N+1, 0:N] + x[1:N+1, 2:N+2])) / (1 + 4 * a)
        set_bnd(b, x)

# Advect the field by tracing backward through the velocity field
def advect(b, d, d0, u, v, dt):
    dt0 = dt * N
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            x = i - dt0 * u[i, j]
            y = j - dt0 * v[i, j]
            if x < 0.5: x = 0.5
            if x > N + 0.5: x = N + 0.5
            if y < 0.5: y = 0.5
            if y > N + 0.5: y = N + 0.5
            i0, j0 = int(x), int(y)
            i1, j1 = i0 + 1, j0 + 1
            s1, t1 = x - i0, y - j0
            s0, t0 = 1 - s1, 1 - t1
            d[i, j] = (s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                       s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1]))
    set_bnd(b, d)

# Perform one step of density update
def dens_step(dens, dens_prev, u, v, diff, dt):
    add_source(dens, dens_prev, dt)
    dens, dens_prev = dens_prev, dens
    diffuse(0, dens, dens_prev, diff, dt)
    dens, dens_prev = dens_prev, dens
    advect(0, dens, dens_prev, u, v, dt)
    dens *= 0.85

# Perform one step of velocity update
def vel_step(u, v, u_prev, v_prev, visc, dt):
    add_source(u, u_prev, dt)
    add_source(v, v_prev, dt)
    u, u_prev = u_prev, u
    diffuse(1, u, u_prev, visc, dt)
    v, v_prev = v_prev, v
    diffuse(2, v, v_prev, visc, dt)
    project(u, v, u_prev, v_prev)
    u, u_prev = u_prev, u
    v, v_prev = v_prev, v
    advect(1, u, u_prev, u_prev, v_prev, dt)
    advect(2, v, v_prev, u_prev, v_prev, dt)
    project(u, v, u_prev, v_prev)

# Project the velocity field to be mass-conserving
def project(u, v, p, div):
    h = 1.0 / N
    div[1:N+1, 1:N+1] = -0.5 * h * (
        u[2:N+2, 1:N+1] - u[0:N, 1:N+1] + 
        v[1:N+1, 2:N+2] - v[1:N+1, 0:N])
    p.fill(0)
    set_bnd(0, div)
    set_bnd(0, p)
    
    for _ in range(20):
        p[1:N+1, 1:N+1] = (div[1:N+1, 1:N+1] + 
                           p[0:N, 1:N+1] + p[2:N+2, 1:N+1] + 
                           p[1:N+1, 0:N] + p[1:N+1, 2:N+2]) / 4
        set_bnd(0, p)
    
    u[1:N+1, 1:N+1] -= 0.5 * (p[2:N+2, 1:N+1] - p[0:N, 1:N+1]) / h
    v[1:N+1, 1:N+1] -= 0.5 * (p[1:N+1, 2:N+2] - p[1:N+1, 0:N]) / h
    set_bnd(1, u)
    set_bnd(2, v)

def reset_simulation():
    global dens, dens_prev, u, v, u_prev, v_prev
    dens.fill(0)
    dens_prev.fill(0)
    u.fill(0)
    v.fill(0)
    u_prev.fill(0)
    v_prev.fill(0)

def main_menu():
    global current_screen
    running = True

    while running:
        screen.fill((20, 20, 20))
        title = font.render("Splash", True, (255, 255, 255))
        screen.blit(title, (213, 100))

        btn1_text = font.render("Fire", True, (0, 0, 0))
        btn1_rect = pygame.Rect(206, 170, 100, 50)
        pygame.draw.rect(screen, (220, 40, 40), btn1_rect)
        screen.blit(btn1_text, (btn1_rect.x+30, btn1_rect.y+15))

        btn2_text = font.render("Fire2", True, (0, 0, 0))
        btn2_rect = pygame.Rect(206, 240, 100, 50)
        pygame.draw.rect(screen, (248, 106, 0), btn2_rect)
        screen.blit(btn2_text, (btn2_rect.x + 23, btn2_rect.y + 15))

        btn3_text = font.render("Tak", True, (0, 0, 0))
        btn3_rect = pygame.Rect(206, 310, 100, 50)
        pygame.draw.rect(screen, (0, 136, 255), btn3_rect)
        screen.blit(btn3_text, (btn3_rect.x + 30, btn3_rect.y + 15))

        miau = font2.render("Przycisk ESC pozwoli Ci tu wrócić", True, (255, 255, 255))
        screen.blit(miau, (120, 410))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn1_rect.collidepoint(event.pos):
                    current_screen = "color1"
                    return  # Wyjście do `run_simulation()`
                if btn2_rect.collidepoint(event.pos):
                    current_screen = "color2"
                    return
                if btn3_rect.collidepoint(event.pos):
                    current_screen = "color3"
                    return

        pygame.display.flip()
        clock.tick(60)

def visualize_fluid(color):
    global current_screen
    reset_simulation()

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    current_screen = "menu"
                    return
            if pygame.mouse.get_pressed()[0]:  # Lewy przycisk myszy
                x, y = pygame.mouse.get_pos()
                grid_x = int(x / 8) + 1
                grid_y = int(y / 8) + 1
                if 1 <= grid_x <= N and 1 <= grid_y <= N:
                    dens[grid_x, grid_y] += 1000  # Dodaj gęstość
                    u[grid_x, grid_y] += 5 * random.choice([-1, 1])  # Losowy kierunek prędkości

        vel_step(u, v, u_prev, v_prev, visc, dt)
        dens_step(dens, dens_prev, u, v, diff, dt)

        screen.fill((0, 0, 0))
        for i in range(1, N + 1):
           for j in range(1, N + 1):
                d = dens[i, j]
                d = max(0, min(d, 1))
                rect = pygame.Rect((i - 1) * 8, (j - 1) * 8, 8, 8)
                pygame.draw.rect(screen, color(d), rect)

        pygame.display.flip()
        clock.tick(60)

while True:
    if current_screen == "menu":
        main_menu()
    elif current_screen == "color1":
        visualize_fluid(color1)
    elif current_screen == "color2":
        visualize_fluid(color2)
    elif current_screen == "color3":
        visualize_fluid(color3)
