import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    HalfwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
)
from xlb.operator.macroscopic import Macroscopic
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
from PIL import Image
import argparse
import os


# Parse command-line arguments
parser = argparse.ArgumentParser(description='LBM Flow Simulation')
parser.add_argument('image_path', type=str, help='Path to the PNG image (white=channel, black=obstacle)')
parser.add_argument('--steps', type=int, default=1000, help='Number of simulation steps (default: 1000)')
parser.add_argument('--post-process-interval', type=int, default=50, help='Interval for saving frames (default: 50)')
parser.add_argument("--gif-fps", type=int, default=10, help="Frames per second for output GIF (default: 10)")
parser.add_argument('--omega', type=float, default=1.1, help='Omega parameter for viscosity (default: 1.1)')
parser.add_argument('--u-max', type=float, default=0.02, help='Maximum inlet velocity (default: 0.02)')
parser.add_argument('--output', type=str, default=None, help='Output GIF filename (default: image_path with .gif extension)')

args = parser.parse_args()

# Set defaults from arguments
image_path = args.image_path
num_steps = args.steps
post_process_interval = args.post_process_interval
omega = args.omega
u_max = args.u_max

# Set output path: use provided output or derive from image_path
if args.output:
    output_gif = args.output
else:
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_gif = os.path.join(os.path.dirname(image_path), f'{image_name}.gif')

print(f"Image: {image_path}")
print(f"Steps: {num_steps}")
print(f"Post-process interval: {post_process_interval}")
print(f"Omega: {omega}")
print(f"U_max: {u_max}")
print(f"Output GIF: {output_gif}")
print()


img = Image.open(image_path).convert('L')  # Convert to grayscale
obstacle_image = np.array(img)

obstacle_image = (obstacle_image < 128).astype(bool).T

grid_shape = obstacle_image.shape


# Simulation Setup
compute_backend = ComputeBackend.JAX
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)

# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Create Grid
grid = grid_factory(grid_shape, compute_backend=compute_backend)

# Define Boundary Indices
box = grid.bounding_box_indices()
box_no_edge = grid.bounding_box_indices(remove_edges=True)
inlet = box_no_edge["left"]
outlet = box_no_edge["right"]
walls = [box["bottom"][i] + box["top"][i] for i in range(velocity_set.d)]
walls = np.unique(np.array(walls), axis=-1).tolist()

# Extract obstacle indices from binary image (True = obstacle)
obstacle_indices = np.where(obstacle_image)
obstacle = [tuple(obstacle_indices[i]) for i in range(velocity_set.d)]


# Parabolic inlet velocity profile
def bc_profile():
    H = float(grid_shape[1] - 1)

    def bc_profile_jax():
        y = jnp.arange(grid_shape[1])
        y_center = y - (H / 2.0)
        r_squared = (2.0 * y_center / H) ** 2.0
        u_x = u_max * jnp.maximum(0.0, 1.0 - r_squared)
        u_y = jnp.zeros_like(u_x)
        return jnp.stack([u_x, u_y])

    return bc_profile_jax


# Boundary Conditions
bc_left = RegularizedBC("velocity", profile=bc_profile(), indices=inlet)
bc_walls = FullwayBounceBackBC(indices=walls)
bc_outlet = ExtrapolationOutflowBC(indices=outlet)
bc_obstacle = HalfwayBounceBackBC(indices=obstacle)
boundary_conditions = [bc_walls, bc_left, bc_outlet, bc_obstacle]

# Setup Stepper
stepper = IncompressibleNavierStokesStepper(
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="BGK",
)
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

# Macroscopic Calculator
macro = Macroscopic(
    compute_backend=ComputeBackend.JAX,
    precision_policy=precision_policy,
    velocity_set=xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=ComputeBackend.JAX),
)

# Storage for animation frames
frames = []

# Simulation Loop
start_time = time.time()
for step in range(num_steps):
    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
    f_0, f_1 = f_1, f_0

    if step % post_process_interval == 0:
        rho, u = macro(f_0)
        u = u[:, 1:-1, 1:-1]
        u_magnitude = jnp.sqrt(u[0] ** 2 + u[1] ** 2)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot velocity magnitude
        im = ax.imshow(u_magnitude.T, origin='lower', cmap='bwr', vmin=0, vmax=u_max)

        # Overlay obstacle pattern as semi-transparent mask
        obstacle_display = np.ma.masked_array(
            np.ones_like(obstacle_image, dtype=float),
            mask=~obstacle_image
        )
        ax.imshow(obstacle_display.T, origin='lower', cmap='Greys', alpha=0.5, vmin=0, vmax=1)

        ax.set_title(f'Flow Past Obstacle - Step {step}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label='Velocity Magnitude')

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.frombuffer(buf, dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames.append(image)
        plt.close(fig)

        elapsed = time.time() - start_time
        print(f"Step {step}: Elapsed {elapsed:.2f}s")
print(f"Total simulation time: {time.time() - start_time:.2f}s, Total frames : {len(frames)}")
# Create GIF
if frames:
    imageio.mimsave(output_gif, frames, fps=args.gif_fps)
    print(f"Animation saved as {output_gif}")
