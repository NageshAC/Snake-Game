import pygame
import random

pygame.init()  
screensize = (800, 600)
screen = pygame.display.set_mode(screensize) 
pygame.display.set_caption("Final project") 

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

radius = 25  # 공의 크기
color_list = [RED, BLUE, GREEN]

class Ball:
    def __init__(self):
        self.x = random.randint(radius, screensize[0]-radius)
        self.y = random.randint(radius, screensize[1]-radius)
        self.dx = 2
        self.dy = 2
        self.color = random.choice(color_list)
        print(self.color)

    def move(self):
        self.x += self.dx
        self.y += self.dy
        if not radius < self.x < screensize[0]-radius:
            self.dx = -self.dx
            self.color = random.choice(color_list)
        elif not radius < self.y < screensize[1]-radius:
            self.dy = -self.dy
            self.color = random.choice(color_list)

    def draw(self, surf):
        pygame.draw.circle(surf, self.color, (self.x, self.y), radius)

b1 = Ball()
b2 = Ball()

go = True
while go:
    clock = pygame.time.Clock() 
    clock.tick(150) 
    for event in pygame.event.get():  
        if event.type == pygame.QUIT:  
            go = False  

    b1.move()
    b2.move()

    screen.fill(WHITE)  
    b1.draw(screen)
    b2.draw(screen)
    pygame.display.flip() 