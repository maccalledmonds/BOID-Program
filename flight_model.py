import pygame
from pygame.math import Vector2
import numpy as np

pygame.init()
screen = pygame.display.set_mode((720, 720))
clock = pygame.time.Clock()
running = True

class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y, size_factor):
        pos = (x, y)
        base_shape = [(1, 1), (1, 16), (21, 8)]
        new_shape = [(int(tup[0] * size_factor), int(tup[1] * size_factor)) for tup in base_shape]

        super().__init__()

        # Load sprite image
        surface_width = int(40 * size_factor)
        surface_height = int(30 * size_factor)
        self.original_image = pygame.Surface((surface_width, surface_height), pygame.SRCALPHA)
        pygame.draw.polygon(self.original_image, (0,0,0), new_shape)

        self.image = self.original_image
        self.rect = self.image.get_rect(center=pos)

        # Vector position and movement
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(0, 0)
        self.acceleration = 0.3
        self.friction = 0.05

        # Rotation
        self.angle = 0
        self.rotation_speed = 5  # degrees per frame

    def update(self):
        keys = pygame.key.get_pressed()

        # ROTATION
        if keys[pygame.K_LEFT]:
            self.angle += self.rotation_speed
        if keys[pygame.K_RIGHT]:
            self.angle -= self.rotation_speed

        # MOVEMENT IN DIRECTION OF ROTATION
        if keys[pygame.K_UP]:
            # direction the sprite is facing
            direction = pygame.math.Vector2(1, 0).rotate(-self.angle)
            self.vel += direction * self.acceleration

        # Apply friction
        self.vel *= (1 - self.friction)

        # Update position
        self.pos += self.vel
        self.rect.center = self.pos
        
        # Set boundary
        screen = pygame.display.get_surface()
        self.width, self.height = screen.get_size()
        x_pos, y_pos = self.pos
        # if x_pos > self.width:
        #     self.pos.x = 0
        # elif x_pos < 0:
        #     self.pos.x = self.width
        # if y_pos > self.height:
        #     self.pos.y = 0
        # elif y_pos < 0:
        #     self.pos.y = self.height

        # Rotate bird away from boundary and corners (optimized for minimal rotation)
        def angle_diff(a, b):
            d = (a - b + 180) % 360 - 180
            return d

        near_left = x_pos < self.width * 0.1
        near_right = x_pos > self.width * 0.9
        near_top = y_pos < self.height * 0.1
        near_bottom = y_pos > self.height * 0.9

        # Helper function to rotate toward target with minimal rotation
        def rotate_to_target(current_angle, target_angle, threshold=15, rotation_speed=10):
            diff = angle_diff(current_angle, target_angle)
            if abs(diff) > threshold:
                # Rotate in the direction that requires less rotation
                return current_angle + (rotation_speed if diff < 0 else -rotation_speed)
            return current_angle

        # Handle corners first
        if near_left and near_top:
            # Top-left corner: face down-right (angle 315)
            self.angle = rotate_to_target(self.angle, 315)
        elif near_right and near_top:
            # Top-right corner: face down-left (angle 225)
            self.angle = rotate_to_target(self.angle, 225)
        elif near_left and near_bottom:
            # Bottom-left corner: face up-right (angle 45)
            self.angle = rotate_to_target(self.angle, 45)
        elif near_right and near_bottom:
            # Bottom-right corner: face up-left (angle 135)
            self.angle = rotate_to_target(self.angle, 135)
        else:
            # Handle edges (not corners)
            if near_right:
                self.angle = rotate_to_target(self.angle, 180)
            elif near_left:
                self.angle = rotate_to_target(self.angle, 0)
            if near_bottom:
                self.angle = rotate_to_target(self.angle, 90)
            elif near_top:
                self.angle = rotate_to_target(self.angle, 270)

        # ROTATE IMAGE
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

        # self.rect.clamp_ip(pygame.display.get_surface().get_rect())

x, y = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2) #Player_pos set in the middle
bird = Bird(x, y, 1)
bird_sprites_group = pygame.sprite.Group(bird)
screen = pygame.display.get_surface()
width, height = screen.get_size()
print(f"surface size is: {width}, {height}")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white") #disable this later on and create a method to trace the bird path

    bird_sprites_group.update()
    bird_sprites_group.draw(screen)


    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()