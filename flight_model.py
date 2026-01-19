import math
import random
import colorsys
from collections import deque
import pygame

# Simulation configuration
WIDTH = 1000
HEIGHT = 700
UI_HEIGHT = 180
SIM_HEIGHT = HEIGHT - UI_HEIGHT

NUM_BOIDS = 60
MAX_SPEED = 10
MAX_FORCE = 0.05

SEPARATION_RADIUS = 22
ALIGNMENT_RADIUS = 48
COHESION_RADIUS = 48

SEPARATION_WEIGHT = 4
ALIGNMENT_WEIGHT = 4
COHESION_WEIGHT = 4

# Edge avoidance parameters
NEAR_EDGE_RATIO = 0.06            # fraction of width/height considered 'near' an edge
EDGE_THRESHOLD_RATIO = 0.12       # fraction of min(width,sim_height) used to scale reaction
ROTATION_BASE = 4.0               # base rotation speed (degrees per frame)
ROTATION_EXTRA = 16.0             # additional rotation scaling when very close to edge
ROTATION_BASE_SPEED = 3.5        # baseline speed used to scale rotation with MAX_SPEED

TRAIL_HISTORY = 30
TRAIL_POINT_STEP = 2
TRAIL_WIDTH = 1

# Visual sizes
BOID_SIZE = 6
RADIUS_MULTIPLIER = 4
BACKGROUND_COLOR = (245, 248, 250)


def hsv_to_rgb_int(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


class Boid:
    def __init__(self, x, y, color):
        self.pos = pygame.math.Vector2(x, y)
        ang = random.uniform(0, math.tau)
        direction = pygame.math.Vector2(math.cos(ang), math.sin(ang))
        self.vel = direction * MAX_SPEED
        self.acc = pygame.math.Vector2(0, 0)
        self.color = color
        self.history = deque(maxlen=TRAIL_HISTORY)
        self._hist_step = 0
        self.angle = math.degrees(ang)

    def apply_force(self, f: pygame.math.Vector2):
        self.acc += f

    def update(self):
        self.vel += self.acc
        l2 = self.vel.length_squared()
        if l2 > 0.0001:
            if l2 > (MAX_SPEED * MAX_SPEED):
                self.vel.scale_to_length(MAX_SPEED)
            self.angle = self.vel.angle_to(pygame.math.Vector2(1, 0))
        self.pos += self.vel
        self.acc *= 0

        # Edge avoidance: if near an edge, then rotate the desired heading away based on direction of approach
        x_pos, y_pos = self.pos.x, self.pos.y
        near_left = x_pos < WIDTH * NEAR_EDGE_RATIO
        near_right = x_pos > WIDTH * (1.0 - NEAR_EDGE_RATIO)
        near_top = y_pos < SIM_HEIGHT * NEAR_EDGE_RATIO
        near_bottom = y_pos > SIM_HEIGHT * (1.0 - NEAR_EDGE_RATIO)

        dist_left = x_pos
        dist_right = WIDTH - x_pos
        dist_top = y_pos
        dist_bottom = SIM_HEIGHT - y_pos
        dist_to_edge = min(dist_left, dist_right, dist_top, dist_bottom)

        edge_threshold = min(WIDTH, SIM_HEIGHT) * EDGE_THRESHOLD_RATIO
        proximity = 0.0
        if dist_to_edge < edge_threshold:
            proximity = max(0.0, (edge_threshold - dist_to_edge) / edge_threshold)

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
            # rotation speed scales with MAX_SPEED so faster boids turn quicker
            speed_scale = MAX_SPEED / ROTATION_BASE_SPEED if ROTATION_BASE_SPEED > 0 else 1.0
            rotation_speed = ROTATION_BASE * speed_scale + proximity * ROTATION_EXTRA * speed_scale
            self.angle = self.rotate_to_target(self.angle, target, rotation_speed=rotation_speed)
            # slowly nudge velocity direction toward the heading represented by self.angle
            direction = pygame.math.Vector2(1, 0).rotate(-self.angle)
            blend = min(1.0, 0.12 + proximity * 0.7)
            self.vel = self.vel + (direction * MAX_SPEED - self.vel) * blend

        self._hist_step += 1
        if self._hist_step >= TRAIL_POINT_STEP:
            self.history.append(pygame.math.Vector2(self.pos.x, self.pos.y))
            self._hist_step = 0

    def neighbors(self, boids, radius):
        res = []
        rr = (radius * RADIUS_MULTIPLIER) ** 2
        for other in boids:
            if other is self:
                continue
            if (other.pos - self.pos).length_squared() <= rr:
                res.append(other)
        return res

    def separation(self, boids):
        steer = pygame.math.Vector2(0, 0)
        total = 0
        for other in self.neighbors(boids, SEPARATION_RADIUS):
            diff = self.pos - other.pos
            d = diff.length()
            if d > 0:
                steer += diff.normalize() / d
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
        avg = pygame.math.Vector2(0, 0)
        total = 0
        for other in self.neighbors(boids, ALIGNMENT_RADIUS):
            avg += other.vel
            total += 1
        if total > 0:
            avg /= total
            if avg.length() > 0:
                avg.scale_to_length(MAX_SPEED)
                steer = avg - self.vel
                if steer.length() > MAX_FORCE:
                    steer.scale_to_length(MAX_FORCE)
                return steer
        return pygame.math.Vector2(0, 0)

    def cohesion(self, boids):
        center = pygame.math.Vector2(0, 0)
        total = 0
        for other in self.neighbors(boids, COHESION_RADIUS):
            center += other.pos
            total += 1
        if total > 0:
            center /= total
            desired = center - self.pos
            if desired.length() > 0:
                desired.scale_to_length(MAX_SPEED)
                steer = desired - self.vel
                if steer.length() > MAX_FORCE:
                    steer.scale_to_length(MAX_FORCE)
                return steer
        return pygame.math.Vector2(0, 0)

    def rotate_to_target(self, current_angle, target_angle, threshold=2, rotation_speed=4):
        # normalize difference to [-180,180]
        diff = (target_angle - current_angle + 180) % 360 - 180
        if abs(diff) <= threshold:
            return target_angle
        step = rotation_speed if diff > 0 else -rotation_speed
        # clamp step to not overshoot
        if abs(step) > abs(diff):
            return target_angle
        return (current_angle + step) % 360

    def draw(self, surf):
        ang = math.radians(-self.angle)
        size = BOID_SIZE
        p1 = (self.pos.x + math.cos(ang) * size * 1.6, self.pos.y + math.sin(ang) * size * 1.6)
        p2 = (self.pos.x + math.cos(ang + 2.5) * size, self.pos.y + math.sin(ang + 2.5) * size)
        p3 = (self.pos.x + math.cos(ang - 2.5) * size, self.pos.y + math.sin(ang - 2.5) * size)
        pygame.draw.polygon(surf, self.color, [p1, p2, p3])


class SliderUI:
    def __init__(self, x, y, w, label, minv, maxv, getter, setter):
        self.x, self.y, self.w = x, y, w
        self.h = 18
        self.label = label
        self.minv, self.maxv = minv, maxv
        self.get = getter
        self.set = setter
        self.drag = False

    def rect(self):
        return pygame.Rect(self.x, self.y, self.w, self.h)

    def knob_x(self):
        v = self.get()
        t = (v - self.minv) / (self.maxv - self.minv)
        return int(self.x + t * self.w)

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect().collidepoint(ev.pos):
                self.drag = True
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.drag = False
        elif ev.type == pygame.MOUSEMOTION and self.drag:
            mx = max(self.x, min(ev.pos[0], self.x + self.w))
            t = (mx - self.x) / self.w
            self.set(self.minv + t * (self.maxv - self.minv))

    def draw(self, surf, font):
        pygame.draw.rect(surf, (220, 220, 220), (self.x, self.y + self.h // 2 - 3, self.w, 6))
        kx = self.knob_x(); ky = self.y + self.h // 2
        pygame.draw.circle(surf, (80, 80, 80), (kx, ky), 8)
        lbl = font.render(f"{self.label}: {self.get():.2f}", True, (30, 30, 30))
        surf.blit(lbl, (self.x, self.y - 18))


def run():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids — flight_model.py")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    boids = []
    for i in range(NUM_BOIDS):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, SIM_HEIGHT)
        color = hsv_to_rgb_int((i / max(1, NUM_BOIDS)) % 1.0, 0.7, 0.95)
        boids.append(Boid(x, y, color))

    global SEPARATION_WEIGHT, ALIGNMENT_WEIGHT, COHESION_WEIGHT

    pad = 14
    sx = pad
    #places five sliders inside the UI panel (Radius multiplier + existing sliders)
    sy = SIM_HEIGHT + 8
    slider_w = WIDTH - pad * 2
    radius_slider = SliderUI(sx, sy + 20, slider_w, 'Visual Range', 0.1, 8.0, lambda: RADIUS_MULTIPLIER, lambda v: globals().update({'RADIUS_MULTIPLIER': v}))
    speed_slider = SliderUI(sx, sy + 50, slider_w, 'Max Speed', 0.5, 40.0, lambda: MAX_SPEED, lambda v: globals().update({'MAX_SPEED': v}))
    sep_slider = SliderUI(sx, sy + 80, slider_w, 'Separation', 0.0, 8.0, lambda: SEPARATION_WEIGHT, lambda v: globals().update({'SEPARATION_WEIGHT': v}))
    ali_slider = SliderUI(sx, sy + 110, slider_w, 'Alignment', 0.0, 8.0, lambda: ALIGNMENT_WEIGHT, lambda v: globals().update({'ALIGNMENT_WEIGHT': v}))
    coh_slider = SliderUI(sx, sy + 140, slider_w, 'Cohesion', 0.0, 8.0, lambda: COHESION_WEIGHT, lambda v: globals().update({'COHESION_WEIGHT': v}))
    sliders = [radius_slider, speed_slider, sep_slider, ali_slider, coh_slider]

    show_trails = False
    running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    for _ in range(10):
                        x = random.uniform(0, WIDTH)
                        y = random.uniform(0, SIM_HEIGHT)
                        color = hsv_to_rgb_int(random.random(), 0.75, 0.95)
                        boids.append(Boid(x, y, color))
                elif ev.key == pygame.K_t:
                    show_trails = not show_trails
            for s in sliders:
                s.handle_event(ev)

        for b in boids:
            s = b.separation(boids)
            a = b.alignment(boids)
            c = b.cohesion(boids)

            b.apply_force(s * SEPARATION_WEIGHT)
            b.apply_force(a * ALIGNMENT_WEIGHT)
            b.apply_force(c * COHESION_WEIGHT)

        for b in boids:
            b.update()

        screen.fill(BACKGROUND_COLOR)

        if show_trails:
            trail_surf = pygame.Surface((WIDTH, SIM_HEIGHT), pygame.SRCALPHA)
            for b in boids:
                pts = list(b.history)
                n = len(pts)
                if n < 2:
                    continue
                for i in range(1, n):
                    p1 = pts[i - 1]
                    p2 = pts[i]
                    alpha = int(255 * (i / max(1, n - 1)) * 0.85)
                    col = (b.color[0], b.color[1], b.color[2], alpha)
                    pygame.draw.line(trail_surf, col, (p1.x, p1.y), (p2.x, p2.y), TRAIL_WIDTH)
            screen.blit(trail_surf, (0, 0))

        for b in boids:
            b.draw(screen)

        ui_y = SIM_HEIGHT
        pygame.draw.rect(screen, (245, 245, 245), (0, ui_y, WIDTH, UI_HEIGHT))
        pygame.draw.line(screen, (200, 200, 200), (0, ui_y), (WIDTH, ui_y), 2)
        for s in sliders:
            s.draw(screen, font)

        hud = font.render(f"Boids: {len(boids)} (Press 'Space' to Add More)     Trails: {'On' if show_trails else 'Off'} (Press 'T' to Toggle)", True, (40, 40, 40))
        screen.blit(hud, (8, 8))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    run() 