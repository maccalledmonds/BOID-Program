"""Boids / Flocking simulation (Pygame)

Controls:
 - SPACE : add 10 more boids
 - T : toggle trails (don't clear background)
 - +/- : change separation weight
 - ESC or close window : quit

Drop this file in the project and run: python flight_model.py
"""

import pygame
import random
import colorsys
from pygame.math import Vector2


WIDTH, HEIGHT = 900, 700
NUM_BOIDS = 80
MAX_SPEED = 6
MAX_FORCE = 0.07

SEPARATION_RADIUS = 25
ALIGNMENT_RADIUS = 60
COHESION_RADIUS = 50

SEPARATION_WEIGHT = 1.8
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 1.5

PIXEL_SEP_RADIUS = 26
PIXEL_SAMPLE_STEP = 6

BACKGROUND_COLOR = (255, 255, 255)


def hsv_to_rgb_int(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


class Boid:
    def __init__(self, x, y, color):
        self.pos = Vector2(x, y)
        angle = random.random() * 360
        self.vel = Vector2(1, 0).rotate(angle) * random.uniform(1.0, MAX_SPEED)
        self.acc = Vector2(0, 0)
        self.color = color
        self.size = 6
        # Current visible angle (degrees). Used when avoiding boundaries.
        self.angle = self.vel.angle_to(Vector2(1, 0))
        self.rotation_speed = 10

    def edges(self):
        # clamp position so boids don't escape window when using avoidance
        self.pos.x = max(0, min(self.pos.x, WIDTH - 1))
        self.pos.y = max(0, min(self.pos.y, HEIGHT - 1))

    def apply_force(self, force: Vector2):
        self.acc += force

    def update(self):
        self.vel += self.acc
        if self.vel.length() > MAX_SPEED:
            self.vel.scale_to_length(MAX_SPEED)
        self.pos += self.vel
        self.acc *= 0
        
        # Update angle and apply edge behavior, detect being near edges and gently rotate away
        x_pos, y_pos = self.pos.x, self.pos.y
        near_left = x_pos < WIDTH * 0.1
        near_right = x_pos > WIDTH * 0.9
        near_top = y_pos < HEIGHT * 0.1
        near_bottom = y_pos > HEIGHT * 0.9

        # choose a target angle depending on corner/edge
        target = None
        if near_left and near_top:
            target = 315
        elif near_right and near_top:
            target = 225
        elif near_left and near_bottom:
            target = 45
        elif near_right and near_bottom:
            target = 135
        else:
            if near_right:
                target = 180
            elif near_left:
                target = 0
            if near_bottom:
                if target is None:
                    target = 90
            elif near_top:
                if target is None:
                    target = 270

        if target is not None:
            self.angle = self._rotate_to_target(self.angle, target, rotation_speed=self.rotation_speed)
            # this pushes a small force toward the new heading so the bird moves away
            direction = Vector2(1, 0).rotate(-self.angle)
            self.apply_force(direction * 0.08)

        self.edges()

    def draw(self, surf):
        # draw as rotated triangle pointing in velocity direction
        heading = self.vel.angle_to(Vector2(1, 0))
        p1 = Vector2(self.size * 2, 0).rotate(-heading) + self.pos
        p2 = Vector2(-self.size, self.size).rotate(-heading) + self.pos
        p3 = Vector2(-self.size, -self.size).rotate(-heading) + self.pos
        pygame.draw.polygon(surf, self.color, [p1, p2, p3])

    # Steering behaviors using neighbors list (vector-based)
    def separation(self, boids):
        steer = Vector2(0, 0)
        total = 0
        for other in boids:
            if other is self:
                continue
            d = self.pos.distance_to(other.pos)
            if d < SEPARATION_RADIUS and d > 0:
                diff = (self.pos - other.pos).normalize() / d
                steer += diff
                total += 1
        if total > 0:
            steer /= total
            if steer.length() > 0:
                steer.scale_to_length(MAX_SPEED)
                steer -= self.vel
                if steer.length() > MAX_FORCE:
                    steer.scale_to_length(MAX_FORCE)
        return steer

    def alignment(self, boids):
        avg = Vector2(0, 0)
        total = 0
        for other in boids:
            if other is self:
                continue
            d = self.pos.distance_to(other.pos)
            if d < ALIGNMENT_RADIUS:
                avg += other.vel
                total += 1
        if total > 0:
            avg /= total
            avg.scale_to_length(MAX_SPEED)
            steer = avg - self.vel
            if steer.length() > MAX_FORCE:
                steer.scale_to_length(MAX_FORCE)
            return steer
        return Vector2(0, 0)

    def cohesion(self, boids):
        center = Vector2(0, 0)
        total = 0
        for other in boids:
            if other is self:
                continue
            d = self.pos.distance_to(other.pos)
            if d < COHESION_RADIUS:
                center += other.pos
                total += 1
        if total > 0:
            center /= total
            desired = (center - self.pos)
            if desired.length() > 0:
                desired.scale_to_length(MAX_SPEED)
                steer = desired - self.vel
                if steer.length() > MAX_FORCE:
                    steer.scale_to_length(MAX_FORCE)
                return steer
        return Vector2(0, 0)

    def _rotate_to_target(self, current_angle, target_angle, threshold=15, rotation_speed=4):
        """Return a new angle moved toward target using minimal rotation."""
        diff = (current_angle - target_angle + 180) % 360 - 180
        if abs(diff) > threshold:
            return current_angle + (rotation_speed if diff < 0 else -rotation_speed)
        return current_angle


def run():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    # create boids with distinct colors
    boids = []
    for i in range(NUM_BOIDS):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        color = hsv_to_rgb_int((i / NUM_BOIDS) % 1.0, 0.75, 0.9)
        boids.append(Boid(x, y, color))

    show_trails = False
    running = True

    global SEPARATION_WEIGHT

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE: # spawn more boids
                    for _ in range(10):
                        i = random.randint(0, 1000)
                        color = hsv_to_rgb_int(random.random(), 0.8, 0.9)
                        boids.append(Boid(random.uniform(0, WIDTH), random.uniform(0, HEIGHT), color))
                elif event.key == pygame.K_t:
                    show_trails = not show_trails
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    SEPARATION_WEIGHT += 0.1
                elif event.key == pygame.K_MINUS or event.key == pygame.K_UNDERSCORE:
                    SEPARATION_WEIGHT = max(0.0, SEPARATION_WEIGHT - 0.1)

        if not show_trails:
            screen.fill(BACKGROUND_COLOR)
        

        # If using pixel-based separation, we must first draw boids to the screen
        # so their colors exist in pixels to be sampled. But we'll still use
        # vector-based alignment/cohesion; pixel mode only replaces separation.
        for b in boids:
            b.draw(screen)

        # compute and apply steering
        for b in boids:
            sep = Vector2(0, 0)
            sep = b.separation(boids)
            ali = b.alignment(boids)
            coh = b.cohesion(boids)

            b.apply_force(sep * SEPARATION_WEIGHT)
            b.apply_force(ali * ALIGNMENT_WEIGHT)
            b.apply_force(coh * COHESION_WEIGHT)

        # update (after drawing) so pixel sampling shows current frame
        for b in boids:
            b.update()

        # HUD
        hud_text = f"Birds: {len(boids)} Seperation_Weight: {SEPARATION_WEIGHT:.1f}  Trails: {show_trails}"
        text = font.render(hud_text, True, (30, 30, 30))
        screen.blit(text, (8, 8))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    run()