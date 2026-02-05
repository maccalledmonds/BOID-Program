import math
import numpy as np
import moderngl
from pyrr import Matrix44, Vector3


VERTEX_SHADER = '''
    #version 330
    in vec3 in_pos;                 // mesh vertex (local coordinates, X = forward)
    in vec3 in_normal;              // per-vertex normal (local space)
    in vec3 instance_pos;           // per-instance position
    in vec3 instance_vel;           // per-instance velocity (direction)
    in vec3 instance_col;           // per-instance color

    uniform mat4 u_view;
    uniform mat4 u_proj;
    uniform float u_scale;

    out vec3 v_color;
    out vec3 v_normal;
    out vec3 v_world_pos;

    // build an orthonormal basis that maps model X->forward, Y->right, Z->up
    mat3 build_basis(vec3 forward) {
        vec3 f = normalize(forward);
        if (length(f) < 1e-6) {
            f = vec3(1.0, 0.0, 0.0);
        }
        vec3 world_up = vec3(0.0, 1.0, 0.0);
        // if forward is nearly parallel to world_up, pick a different up
        if (abs(dot(f, world_up)) > 0.999) {
            world_up = vec3(1.0, 0.0, 0.0);
        }
        vec3 r = normalize(cross(world_up, f)); // right
        vec3 u = cross(f, r); // up
        // columns order: forward, right, up so local (x,y,z) -> x*forward + y*right + z*up
        return mat3(f, r, u);
    }

    void main() {
        v_color = instance_col;
        // construct basis from velocity
        mat3 basis = build_basis(instance_vel);
        // apply uniform scale and orient the vertex
        vec3 local = in_pos * u_scale;
        // map local coords using the basis: local.x*forward + local.y*right + local.z*up
        vec3 world = instance_pos + basis * local;
        gl_Position = u_proj * u_view * vec4(world, 1.0);
        // rotate normal by basis (no scale assumed)
        v_normal = normalize(basis * in_normal);
        v_world_pos = world;
    }
'''


FRAGMENT_SHADER = '''
    #version 330
    in vec3 v_color;
    in vec3 v_normal;
    in vec3 v_world_pos;
    out vec4 f_color;

    uniform vec3 u_light_dir;
    uniform float u_ambient;
    uniform float u_diffuse;

    void main() {
        vec3 N = normalize(v_normal);
        vec3 L = normalize(u_light_dir);
        float d = max(dot(N, L), 0.0);
        vec3 color = v_color * (u_ambient + u_diffuse * d);
        f_color = vec4(color, 1.0);
    }
'''


class GLRenderer:
    """ModernGL renderer for instanced boid triangles.

    - Creates a ModernGL context from the current GL context (must be current)
    - Uses instanced rendering: per-instance attributes are position, velocity, color.
    - Exposes `render(positions, velocities, colors)` which accepts NumPy arrays.
    """

    def __init__(self, width, height, camera_pos=(500.0, 350.0, 900.0)):
        # Create context from the current OpenGL context (must be current)
        self.ctx = moderngl.create_context()
        self.width = width
        self.height = height
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

        # 3D elongated pyramid mesh (forward along +X). tip is farther on +X for elongation
        tip = np.array([2.0, 0.0, 0.0], dtype='f4')
        # base is an equilateral triangle lying in the YZ plane, centered at x=-0.5
        s = 0.8  # side length of the equilateral base
        h = s * 0.8660254037844386  # sqrt(3)/2
        b1 = np.array([-0.5,  2.0 * h / 3.0,  0.0], dtype='f4')    # top vertex (centroid at origin)
        b2 = np.array([-0.5, -1.0 * h / 3.0, -s / 2.0], dtype='f4')
        b3 = np.array([-0.5, -1.0 * h / 3.0,  s / 2.0], dtype='f4')

        faces = [
            (tip, b1, b2),
            (tip, b2, b3),
            (tip, b3, b1),
            (b1, b3, b2),  # base
        ]

        positions = []
        normals = []
        for a, b, c in faces:
            # face normal
            u = b - a
            v = c - a
            n = np.cross(u, v)
            n = n / (np.linalg.norm(n) + 1e-9)
            for p in (a, b, c):
                positions.extend(p.tolist())
                normals.extend(n.tolist())

        mesh = np.array(positions + normals, dtype='f4')
        # interleave: we will create buffer with pos then normals separated by attribute format
        # pack as [pos..., normal...] per vertex by concatenating two arrays
        vert_count = int(len(positions) / 3)
        interleaved = np.empty(vert_count * 6, dtype='f4')
        interleaved[0::6] = np.array(positions[0::3], dtype='f4')
        interleaved[1::6] = np.array(positions[1::3], dtype='f4')
        interleaved[2::6] = np.array(positions[2::3], dtype='f4')
        interleaved[3::6] = np.array(normals[0::3], dtype='f4')
        interleaved[4::6] = np.array(normals[1::3], dtype='f4')
        interleaved[5::6] = np.array(normals[2::3], dtype='f4')

        self.vbo = self.ctx.buffer(interleaved.tobytes())

        # Empty instance buffer (will be updated per-frame)
        # layout: vec3 pos, vec3 vel, vec3 col -> 9 floats per instance
        self.instance_buffer = self.ctx.buffer(reserve=9 * 4 * 1000)  # reserve for up to 1000 instances

        # Vertex Array: bind mesh and instance attributes (instance divisor)
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, '3f 3f', 'in_pos', 'in_normal'),
                (self.instance_buffer, '3f 3f 3f/i', 'instance_pos', 'instance_vel', 'instance_col'),
            ],
        )

        # camera (spherical controls)
        self.width = width
        self.height = height
        self.camera_pos = Vector3(camera_pos)
        self.view = Matrix44.look_at(self.camera_pos, (width/2.0, height/2.0, 0.0), (0.0, 1.0, 0.0))
        self.proj = Matrix44.perspective_projection(45.0, width / float(height), 0.1, 20000.0)

        # set uniforms
        self.prog['u_view'].write(self.view.astype('f4').tobytes())
        self.prog['u_proj'].write(self.proj.astype('f4').tobytes())
        self.prog['u_scale'].value = 8.0

        # simple lighting
        self.prog['u_light_dir'].value = (0.3, 0.5, -1.0)
        self.prog['u_ambient'].value = 0.25
        self.prog['u_diffuse'].value = 0.85

        # state
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        # camera spherical state (degrees)
        self.yaw = 0.0
        self.pitch = -25.0
        self.distance = 900.0
        self.look_at = (width/2.0, height/2.0, 0.0)

        # line program for boundary visualization
        LINE_VS = '''
            #version 330
            in vec3 in_pos;
            uniform mat4 u_view;
            uniform mat4 u_proj;
            void main() {
                gl_Position = u_proj * u_view * vec4(in_pos, 1.0);
            }
        '''
        LINE_FS = '''
            #version 330
            uniform vec3 u_color;
            out vec4 f_color;
            void main() { f_color = vec4(u_color, 1.0); }
        '''
        self.line_prog = self.ctx.program(vertex_shader=LINE_VS, fragment_shader=LINE_FS)
        self.line_vbo = None
        self.line_vao = None

        # 2D UI shader for drawing quads (panel backgrounds, slider tracks, knobs)
        UI_VS = '''
            #version 330
            in vec2 in_pos;
            uniform mat4 u_ortho;
            void main() {
                gl_Position = u_ortho * vec4(in_pos, 0.0, 1.0);
            }
        '''
        UI_FS = '''
            #version 330
            uniform vec4 u_color;
            out vec4 f_color;
            void main() { f_color = u_color; }
        '''
        self.ui_prog = self.ctx.program(vertex_shader=UI_VS, fragment_shader=UI_FS)
        self.ui_vbo = self.ctx.buffer(reserve=6 * 2 * 4 * 100)  # reserve for many quads
        self.ui_vao = self.ctx.vertex_array(self.ui_prog, [(self.ui_vbo, '2f', 'in_pos')])
        self._update_ortho()

    def update_camera(self, camera_pos=None, look_at=None):
        if camera_pos is not None:
            self.camera_pos = Vector3(camera_pos)
        if look_at is None:
            look_at = (self.width/2.0, self.height/2.0, 0.0)
        self.view = Matrix44.look_at(self.camera_pos, look_at, (0.0, 1.0, 0.0))
        self.prog['u_view'].write(self.view.astype('f4').tobytes())

    def set_camera_spherical(self, yaw_deg, pitch_deg, distance, look_at=None):
        self.yaw = yaw_deg
        self.pitch = pitch_deg
        self.distance = distance
        if look_at is not None:
            self.look_at = look_at
        # convert spherical to Cartesian (Y is world-up)
        # yaw rotates around the Y axis (changes X and Z), pitch moves up/down (changes Y)
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        x = self.look_at[0] + self.distance * math.cos(pitch) * math.cos(yaw)
        y = self.look_at[1] + self.distance * math.sin(pitch)
        z = self.look_at[2] + self.distance * math.cos(pitch) * math.sin(yaw)
        self.update_camera((x, y, z), self.look_at)

    def resize(self, width, height):
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
        self.proj = Matrix44.perspective_projection(45.0, width / float(height), 0.1, 20000.0)
        self.prog['u_proj'].write(self.proj.astype('f4').tobytes())
        self.line_prog['u_proj'].write(self.proj.astype('f4').tobytes())

    def render(self, positions: np.ndarray, velocities: np.ndarray, colors: np.ndarray):
        """Render instances.

        positions: (N,3) float32
        velocities: (N,3) float32
        colors: (N,3) float32 in 0..1
        """
        n = len(positions)
        if n == 0:
            return

        # pack instance data
        inst = np.hstack([positions.astype('f4'), velocities.astype('f4'), colors.astype('f4')])
        # write per-instance buffer
        self.instance_buffer.orphan(size=inst.nbytes)
        self.instance_buffer.write(inst.tobytes())

        # clear and draw
        self.ctx.clear(0.96, 0.97, 0.98)
        self.ctx.enable(moderngl.DEPTH_TEST)
        # update view uniform in case camera changed externally
        self.prog['u_view'].write(self.view.astype('f4').tobytes())
        self.line_prog['u_view'].write(self.view.astype('f4').tobytes())
        self.vao.render(instances=n)

    def draw_trail(self, points, color, alpha_fade=True):
        """Draw a trail as connected line segments with optional alpha fade.
        
        points: list of (x, y, z) tuples
        color: (r, g, b) in 0..1
        alpha_fade: if True, older points are more transparent
        """
        if len(points) < 2:
            return

        # Build a single contiguous vertex list and render as LINE_STRIP
        coords = np.array([c for p in points for c in p], dtype='f4')

        # Create or update a temporary VBO/VAO for this trail and render immediately.
        # We reuse self.line_vbo/self.line_vao for simplicity, but ensure the buffer
        # size matches the current coords to avoid leftover vertices from previous draws.
        if self.line_vbo is None:
            self.line_vbo = self.ctx.buffer(coords.tobytes())
            self.line_vao = self.ctx.vertex_array(self.line_prog, [(self.line_vbo, '3f', 'in_pos')])
        else:
            self.line_vbo.orphan(size=coords.nbytes)
            self.line_vbo.write(coords.tobytes())

        # Use a slightly dimmed version of the boid color for trails
        trail_color = (color[0] * 0.7, color[1] * 0.7, color[2] * 0.7)
        self.line_prog['u_color'].value = trail_color
        # ensure view/proj uniforms up to date
        self.line_prog['u_view'].write(self.view.astype('f4').tobytes())
        self.line_prog['u_proj'].write(self.proj.astype('f4').tobytes())

        # Render as a single line strip
        vertex_count = int(len(coords) / 3)
        self.line_vao.render(mode=moderngl.LINE_STRIP, vertices=vertex_count)

    def draw_boundary(self, width, sim_height, depth, color=(0.2, 0.2, 0.2)):
        # draw the 12 edges of the axis-aligned box from (0,0,-depth/2) to (width,sim_height,depth/2)
        zmin = -depth / 2.0
        zmax = depth / 2.0
        # 8 corners
        c0 = (0.0, 0.0, zmin)
        c1 = (width, 0.0, zmin)
        c2 = (width, sim_height, zmin)
        c3 = (0.0, sim_height, zmin)
        c4 = (0.0, 0.0, zmax)
        c5 = (width, 0.0, zmax)
        c6 = (width, sim_height, zmax)
        c7 = (0.0, sim_height, zmax)

        # edges as pairs (12 edges)
        edges = [
            (c0, c1), (c1, c2), (c2, c3), (c3, c0),  # bottom
            (c4, c5), (c5, c6), (c6, c7), (c7, c4),  # top
            (c0, c4), (c1, c5), (c2, c6), (c3, c7),  # vertical
        ]

        coords = np.array([coord for edge in edges for corner in edge for coord in corner], dtype='f4')

        if self.line_vbo is None:
            self.line_vbo = self.ctx.buffer(coords.tobytes())
            self.line_vao = self.ctx.vertex_array(self.line_prog, [(self.line_vbo, '3f', 'in_pos')])
        else:
            self.line_vbo.orphan(size=coords.nbytes)
            self.line_vbo.write(coords.tobytes())

        self.line_prog['u_color'].value = tuple(color)
        # ensure view/proj uniforms up to date
        self.line_prog['u_view'].write(self.view.astype('f4').tobytes())
        self.line_prog['u_proj'].write(self.proj.astype('f4').tobytes())
        self.line_vao.render(mode=moderngl.LINES)

    def _update_ortho(self):
        # orthographic projection for 2D UI (origin top-left, Y down)
        # pyrr expects: left, right, bottom, top, near, far
        # For Y-down with origin at top-left: left=0, right=width, bottom=height, top=0
        ortho = Matrix44.orthogonal_projection(0.0, float(self.width), float(self.height), 0.0, -1.0, 1.0, dtype='f4')
        self.ui_prog['u_ortho'].write(ortho.tobytes())

    def draw_ui_quad(self, x, y, w, h, color):
        """Draw a filled 2D quad. color is (r,g,b,a) in 0..1."""
        self._update_ortho()
        # two triangles for the quad
        verts = np.array([
            x, y,
            x + w, y,
            x + w, y + h,
            x, y,
            x + w, y + h,
            x, y + h,
        ], dtype='f4')
        self.ui_vbo.orphan(size=verts.nbytes)
        self.ui_vbo.write(verts.tobytes())
        self.ui_prog['u_color'].value = (color[0], color[1], color[2], color[3] if len(color) > 3 else 1.0)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ui_vao.render(moderngl.TRIANGLES, vertices=6)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def draw_ui_circle(self, cx, cy, r, color, segments=16):
        """Draw a filled 2D circle using a triangle fan."""
        self._update_ortho()
        verts = []
        for i in range(segments):
            a1 = 2 * math.pi * i / segments
            a2 = 2 * math.pi * (i + 1) / segments
            verts.extend([cx, cy])
            verts.extend([cx + r * math.cos(a1), cy + r * math.sin(a1)])
            verts.extend([cx + r * math.cos(a2), cy + r * math.sin(a2)])
        verts = np.array(verts, dtype='f4')
        self.ui_vbo.orphan(size=verts.nbytes)
        self.ui_vbo.write(verts.tobytes())
        self.ui_prog['u_color'].value = tuple(color)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ui_vao.render(moderngl.TRIANGLES, vertices=segments * 3)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _init_text_shader(self):
        """Initialize shader for textured quads (text rendering)."""
        if hasattr(self, 'text_prog'):
            return
        TEXT_VS = '''
            #version 330
            in vec2 in_pos;
            in vec2 in_uv;
            uniform mat4 u_ortho;
            out vec2 v_uv;
            void main() {
                gl_Position = u_ortho * vec4(in_pos, 0.0, 1.0);
                v_uv = in_uv;
            }
        '''
        TEXT_FS = '''
            #version 330
            in vec2 v_uv;
            uniform sampler2D u_tex;
            out vec4 f_color;
            void main() {
                f_color = texture(u_tex, v_uv);
            }
        '''
        self.text_prog = self.ctx.program(vertex_shader=TEXT_VS, fragment_shader=TEXT_FS)
        self.text_vbo = self.ctx.buffer(reserve=6 * 4 * 4)  # 6 verts, 4 floats each
        self.text_vao = self.ctx.vertex_array(self.text_prog, [(self.text_vbo, '2f 2f', 'in_pos', 'in_uv')])

    def draw_text_texture(self, pygame_surface, x, y):
        """Render a Pygame surface (e.g. rendered text) as a textured quad at (x, y)."""
        import pygame
        self._init_text_shader()
        w, h = pygame_surface.get_size()
        # Convert Pygame surface to RGBA bytes (Pygame uses BGR, need to flip)
        # Use pygame.image.tostring with 'RGBA' format
        rgba_str = pygame.image.tostring(pygame_surface, 'RGBA', True)
        # Create texture
        tex = self.ctx.texture((w, h), 4, rgba_str)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        # Quad vertices with UVs (flipped V because of coordinate systems)
        verts = np.array([
            x, y, 0, 1,
            x + w, y, 1, 1,
            x + w, y + h, 1, 0,
            x, y, 0, 1,
            x + w, y + h, 1, 0,
            x, y + h, 0, 0,
        ], dtype='f4')
        self.text_vbo.orphan(size=verts.nbytes)
        self.text_vbo.write(verts.tobytes())
        # Update ortho for text shader
        ortho = Matrix44.orthogonal_projection(0, self.width, self.height, 0, -1, 1)
        self.text_prog['u_ortho'].write(ortho.astype('f4').tobytes())
        # Render
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        tex.use(0)
        self.text_vao.render(moderngl.TRIANGLES, vertices=6)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        tex.release()
