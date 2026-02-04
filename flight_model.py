import math
import random
import colorsys
from collections import deque
import pygame
import numpy as np
from gl_renderer import GLRenderer

# Simulation configuration
WIDTH = 1000
HEIGHT = 700
UI_HEIGHT = 180
SIM_HEIGHT = HEIGHT - UI_HEIGHT
DEPTH = max(WIDTH, SIM_HEIGHT, 600) * 2  # doubled boundary cube size

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
        # 3D position and velocity
        z = random.uniform(-DEPTH/2.0, DEPTH/2.0)
        self.pos = pygame.math.Vector3(x, y, z)
        # random 3D direction
        d = pygame.math.Vector3(random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1))
        if d.length() == 0:
            d = pygame.math.Vector3(1, 0, 0)
        d.scale_to_length(1.0)
        self.vel = d * MAX_SPEED
        # accelerations are 3D
        self.acc = pygame.math.Vector3(0, 0, 0)
        self.color = color
        self.history = deque(maxlen=TRAIL_HISTORY)
        self._hist_step = 0
        self.angle = 0.0

    def apply_force(self, f):
        # accept either 2D or 3D vectors
        if isinstance(f, pygame.math.Vector2):
            f = pygame.math.Vector3(f.x, f.y, 0.0)
        self.acc += f

    def update(self):
        self.vel += self.acc
        l2 = self.vel.length_squared()
        if l2 > 0.0001:
            if l2 > (MAX_SPEED * MAX_SPEED):
                self.vel.scale_to_length(MAX_SPEED)
            # update 2D heading angle from XY components for UI drawing
            self.angle = math.degrees(math.atan2(self.vel.y, self.vel.x))
        self.pos += self.vel
        self.acc *= 0

        # 3D Edge avoidance: compute distance to each wall and nudge inward when near
        x_pos, y_pos, z_pos = self.pos.x, self.pos.y, self.pos.z
        z_min = -DEPTH / 2.0
        z_max = DEPTH / 2.0
        dist_left = x_pos
        dist_right = WIDTH - x_pos
        dist_top = y_pos
        dist_bottom = SIM_HEIGHT - y_pos
        dist_z_min = z_pos - z_min
        dist_z_max = z_max - z_pos
        dist_to_edge = min(dist_left, dist_right, dist_top, dist_bottom, dist_z_min, dist_z_max)

        edge_threshold = min(WIDTH, SIM_HEIGHT, DEPTH) * EDGE_THRESHOLD_RATIO
        proximity = 0.0
        if dist_to_edge < edge_threshold:
            proximity = max(0.0, (edge_threshold - dist_to_edge) / edge_threshold)
            # pick inward direction away from the closest face(s)
            dir_vec = pygame.math.Vector3(0, 0, 0)
            # X axis
            if dist_left == dist_to_edge:
                dir_vec += pygame.math.Vector3(1, 0, 0)
            elif dist_right == dist_to_edge:
                dir_vec += pygame.math.Vector3(-1, 0, 0)
            # Y axis
            if dist_top == dist_to_edge:
                dir_vec += pygame.math.Vector3(0, 1, 0)
            elif dist_bottom == dist_to_edge:
                dir_vec += pygame.math.Vector3(0, -1, 0)
            # Z axis
            if dist_z_min == dist_to_edge:
                dir_vec += pygame.math.Vector3(0, 0, 1)
            elif dist_z_max == dist_to_edge:
                dir_vec += pygame.math.Vector3(0, 0, -1)
            if dir_vec.length() == 0:
                # fallback: push toward center
                center = pygame.math.Vector3(WIDTH/2.0, SIM_HEIGHT/2.0, 0.0)
                dir_vec = (center - self.pos)
            dir_vec.normalize_ip()
            blend = min(1.0, 0.12 + proximity * 0.7)
            desired = dir_vec * MAX_SPEED
            self.vel = self.vel + (desired - self.vel) * blend

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
        steer = pygame.math.Vector3(0, 0, 0)
        total = 0
        for other in self.neighbors(boids, SEPARATION_RADIUS):
            diff = self.pos - other.pos
            d = diff.length()
            if d > 0:
                steer += (diff.normalize() / d)
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
        avg = pygame.math.Vector3(0, 0, 0)
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
        return pygame.math.Vector3(0, 0, 0)

    def cohesion(self, boids):
        center = pygame.math.Vector3(0, 0, 0)
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
        return pygame.math.Vector3(0, 0, 0)

    def rotate_to_target(self, current_angle, target_angle, threshold=2, rotation_speed=4):
        # keep legacy method for 2D UI rotation (unused for 3D calculations)
        diff = (target_angle - current_angle + 180) % 360 - 180
        if abs(diff) <= threshold:
            return target_angle
        step = rotation_speed if diff > 0 else -rotation_speed
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

    def draw_gl(self, renderer):
        """Draw slider using OpenGL via the renderer."""
        # track background
        renderer.draw_ui_quad(self.x, self.y + self.h // 2 - 3, self.w, 6, (0.86, 0.86, 0.86, 1.0))
        # knob
        kx = self.knob_x()
        ky = self.y + self.h // 2
        renderer.draw_ui_circle(kx, ky, 8, (0.31, 0.31, 0.31, 1.0))


def run():
    pygame.init()
    # Create an OpenGL-capable window but keep using Pygame for events/input
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    flags = pygame.OPENGL | pygame.DOUBLEBUF
    screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
    pygame.display.set_caption("Boids — flight_model.py (ModernGL)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    boids = []
    for i in range(NUM_BOIDS):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, SIM_HEIGHT)
        color = hsv_to_rgb_int((i / max(1, NUM_BOIDS)) % 1.0, 0.7, 0.95)
        boids.append(Boid(x, y, color))

    # Initialize ModernGL renderer (uses the current OpenGL context created above)
    renderer = GLRenderer(WIDTH, HEIGHT)
    # persistent orbit center (so panning updates the rotation center)
    renderer.look_at = (WIDTH / 2.0, SIM_HEIGHT / 2.0, 0.0)
    renderer.set_camera_spherical(renderer.yaw, renderer.pitch, renderer.distance, look_at=renderer.look_at)
    # camera interaction state (trackpad/mouse drag and wheel)
    dragging = False

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
    dragging = False
    right_dragging = False
    prev_mouse_pos = pygame.mouse.get_pos()

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
            
            # Check if any slider is being dragged (skip camera controls if so)
            slider_dragging = any(s.drag for s in sliders)
            
            # Check if mouse is in the UI area
            mouse_in_ui = ev.pos[1] > SIM_HEIGHT if hasattr(ev, 'pos') else pygame.mouse.get_pos()[1] > SIM_HEIGHT
            
            #Camera interaction model - only when not interacting with UI
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if not mouse_in_ui:
                    dragging = True
                    # Reset get_rel() to prevent snapping from accumulated deltas
                    pygame.mouse.get_rel()
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                if not mouse_in_ui:
                    right_dragging = True
                    # Reset get_rel() to prevent snapping from accumulated deltas
                    pygame.mouse.get_rel()
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 3:
                right_dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging:
                # Orbit camera: horizontal drag -> yaw, vertical drag -> pitch
                # Only process if we started dragging in 3D area (dragging flag is set)
                dx, dy = ev.rel

                # sensitivities (resolution-scaled)
                yaw_sens = 0.25 * (800.0 / max(1, WIDTH))
                pitch_sens = 0.2 * (600.0 / max(1, SIM_HEIGHT))

                # map dx -> yaw (rotate around world-up), dy -> pitch (tilt)
                # inverted: drag left->right, drag right->left, drag down->up, drag up->down
                yaw_delta = dx * yaw_sens
                pitch_delta = dy * pitch_sens

                renderer.yaw += yaw_delta
                renderer.pitch += pitch_delta
                renderer.pitch = max(-89.0, min(89.0, renderer.pitch))
                # update camera (orbit around persistent look_at)
                renderer.set_camera_spherical(renderer.yaw, renderer.pitch, renderer.distance, look_at=renderer.look_at)
                
            elif ev.type == pygame.MOUSEMOTION and right_dragging:
                # panning: move look_at in camera right/up plane based on mouse deltas
                # use get_rel() for reliable per-frame deltas while a button is held
                dx, dy = ev.rel

                # compute camera basis from yaw/pitch (degrees)
                yaw_rad = math.radians(renderer.yaw)
                pitch_rad = math.radians(renderer.pitch)
                dir_x = math.cos(pitch_rad) * math.cos(yaw_rad)
                dir_y = math.cos(pitch_rad) * math.sin(yaw_rad)
                dir_z = math.sin(pitch_rad)
                # forward points from camera toward look_at, which is -dir
                forward = (-dir_x, -dir_y, -dir_z)
                # world up
                upw = (0.0, 1.0, 0.0)
                # right = normalize(cross(world_up, forward))
                rx = upw[1] * forward[2] - upw[2] * forward[1]
                ry = upw[2] * forward[0] - upw[0] * forward[2]
                rz = upw[0] * forward[1] - upw[1] * forward[0]
                rlen = math.sqrt(rx * rx + ry * ry + rz * rz) + 1e-9
                rx /= rlen; ry /= rlen; rz /= rlen
                # up vector for camera = cross(forward, right)
                ux = forward[1] * rz - forward[2] * ry
                uy = forward[2] * rx - forward[0] * rz
                uz = forward[0] * ry - forward[1] * rx
                ulen = math.sqrt(ux * ux + uy * uy + uz * uz) + 1e-9
                ux /= ulen; uy /= ulen; uz /= ulen

                # pan scale: proportional to distance so panning feels consistent
                pan_scale = renderer.distance * 0.001

                lx, ly, lz = renderer.look_at
                # move look_at: inverted panning (drag right = pan left, drag down = pan up)
                nx = lx + rx * dx * pan_scale + ux * dy * pan_scale
                ny = ly + ry * dx * pan_scale + uy * dy * pan_scale
                nz = lz + rz * dx * pan_scale + uz * dy * pan_scale
                # persist the new look_at so future orbit uses the panned center
                renderer.look_at = (nx, ny, nz)
                renderer.set_camera_spherical(renderer.yaw, renderer.pitch, renderer.distance, look_at=renderer.look_at)
            elif ev.type == pygame.MOUSEWHEEL:
                # zoom (keep multiplicative feel); clamp distance
                if ev.y > 0:
                    renderer.distance *= 0.85
                else:
                    renderer.distance *= 1.15

                # clamp to sensible bounds (allow much larger distance so boundary stays visible)
                renderer.distance = max(10.0, min(20000.0, renderer.distance))

                renderer.set_camera_spherical(
                    renderer.yaw,
                    renderer.pitch,
                    renderer.distance,
                    look_at=renderer.look_at,
                )

        for b in boids:
            s = b.separation(boids)
            a = b.alignment(boids)
            c = b.cohesion(boids)

            b.apply_force(s * SEPARATION_WEIGHT)
            b.apply_force(a * ALIGNMENT_WEIGHT)
            b.apply_force(c * COHESION_WEIGHT)

        for b in boids:
            b.update()

        # (Camera is controlled by trackpad/mouse drag and wheel; keyboard controls removed)

        # Prepare per-instance arrays for ModernGL
        n = len(boids)
        positions = np.zeros((n, 3), dtype='f4')
        velocities = np.zeros((n, 3), dtype='f4')
        colors = np.zeros((n, 3), dtype='f4')
        for i, b in enumerate(boids):
            positions[i, :] = (b.pos.x, b.pos.y, b.pos.z)
            velocities[i, :] = (b.vel.x, b.vel.y, b.vel.z)
            colors[i, :] = (b.color[0] / 255.0, b.color[1] / 255.0, b.color[2] / 255.0)

        # Render the flock with ModernGL (instanced)
        renderer.render(positions, velocities, colors)
        # draw the container boundary in 3D as a cube (use DEPTH for all dimensions)
        renderer.draw_boundary(DEPTH, DEPTH, DEPTH, color=(0.15, 0.15, 0.15))

        # Draw floating UI panel at the bottom
        # panel background (dark for contrast)
        renderer.draw_ui_quad(0, SIM_HEIGHT, WIDTH, UI_HEIGHT, (0.2, 0.25, 0.3, 0.95))
        # top border line
        renderer.draw_ui_quad(0, SIM_HEIGHT, WIDTH, 2, (0.4, 0.45, 0.5, 1.0))
        # draw sliders
        for s in sliders:
            s.draw_gl(renderer)
        # draw slider labels using Pygame text rendered to texture
        for s in sliders:
            lbl_surf = font.render(f"{s.label}: {s.get():.2f}", True, (220, 220, 220))
            renderer.draw_text_texture(lbl_surf, s.x, s.y - 16)
        # draw instructions
        instr_surf = font.render("Drag sliders | SPACE: +10 boids | T: trails | Mouse: orbit | Scroll: zoom", True, (180, 180, 180))
        renderer.draw_text_texture(instr_surf, 14, HEIGHT - 24)

        # Swap buffers (keeps Pygame event loop and input handling intact)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    run()