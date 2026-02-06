import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tide
from tide import CallbackState

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

print(f"Using device: {device}")

model_path = "examples/data/OverThrust.npy"
epsilon_true_raw = np.load(model_path)
print(f"Loaded model shape: {epsilon_true_raw.shape}")
print(f"Permittivity range: {epsilon_true_raw.min():.2f} - {epsilon_true_raw.max():.2f}")

air_layer = 3
ny, nx = epsilon_true_raw.shape
epsilon_true_np = epsilon_true_raw.copy()
epsilon_true_np[:air_layer, :] = 1.0

sigma_true_np = np.ones_like(epsilon_true_np) * 1e-3
sigma_true_np[:air_layer, :] = 0.0

dx = 0.01
model_size_y = ny * dx
model_size_x = nx * dx
epsilon_r = torch.tensor(epsilon_true_np, dtype=torch.float32, device=device)
sigma = torch.tensor(sigma_true_np, dtype=torch.float32, device=device)
mu_r = torch.ones_like(epsilon_r)
EP0 = 8.8541878128e-12
MU0 = 1.2566370614359173e-06
c0 = 1.0 / (EP0 * MU0)**0.5
min_epsilon = float(epsilon_r.min().item())
v_max = c0 / np.sqrt(min_epsilon)
dt = dx / (v_max * 2.0)  # CFL condition
nt = 3000  # Number of time steps
freq = 500e6  # 500 MHz
t = torch.arange(nt, device=device) * dt
t0 = 1.5 / freq  # Wavelet delay

source_amplitude = torch.zeros(1, 1, nt, device=device)
pi = 3.14159265359
source_amplitude[0, 0, :] = (1 - 2*(pi*freq*(t - t0))**2) * torch.exp(-(pi*freq*(t - t0))**2)

src_y, src_x = 50, nx // 2  # from the top
source_location = torch.tensor([[[src_y, src_x]]], device=device)
n_receivers = 10
receiver_spacing = 20  
receiver_y = src_y  
receiver_x_start = nx // 2 - n_receivers * receiver_spacing // 2
receiver_locations = [[[receiver_y, receiver_x_start + i * receiver_spacing]] for i in range(n_receivers)]
receiver_location = torch.tensor(receiver_locations, device=device).squeeze(1).unsqueeze(0)


model = tide.MaxwellTM(
    epsilon=epsilon_r,
    sigma=sigma,
    mu=mu_r,
    grid_spacing=dx,
)

snapshots = []
snapshot_interval = 10  # Save every 10 steps

def save_snapshot(state: CallbackState):
    """Callback: use CallbackState to save Ey field snapshots
    
    CallbackState provides a standardized interface for accessing wavefields,
    models, and gradients, and supports three views: 'inner', 'pml', 'full':
    - 'full': full wavefield including PML extensions [n_shots, ny+2*pml, nx+2*pml]
    - 'inner': original model region [n_shots, ny, nx]
    - 'pml': region including PML but excluding FD padding
    
    """
    # Use get_wavefield to fetch the Ey field in the inner region (original model size)
    Ey = state.get_wavefield("Ey", view="inner")
    # Ey shape: [n_shots, ny, nx] - same as the input model size
    snapshots.append(Ey[0].clone().cpu().numpy())
    
    # Print progress info
    if state.step % 100 == 0:
        max_amp = Ey.abs().max().item()
        print(f"  Step {state.step:4d}/{state.nt} | "
              f"Time: {state.time*1e9:.2f} ns | "
              f"Progress: {state.progress*100:.1f}% | "
              f"Max |Ey|: {max_amp:.4e}")

# ============== Run simulation ==============
print("\nRunning simulation...")

pml_width = 20  
use_python_backend = False

backend_name = "Python" if use_python_backend else ("CUDA" if device.type == "cuda" else "CPU (C)")
print(f"Backend: {backend_name}")

result = model(
    dt=dt,
    source_amplitude=source_amplitude,
    source_location=source_location,
    receiver_location=receiver_location,
    pml_width=pml_width,
    stencil=4,
    python_backend=use_python_backend,
    forward_callback=save_snapshot,
    callback_frequency=snapshot_interval,  # Call callback every snapshot_interval steps
)

print(f"Done! Saved {len(snapshots)} snapshots")

# ============== Visualization ==============
print("\nGenerating animation...")

fig, ax = plt.subplots(figsize=(10, 10))

# Compute color range - use a smaller range to observe boundary reflections
vmax = max(np.abs(s).max() for s in snapshots) * 0.005  # 5% of max to see reflections
vmin = -vmax
print(f"Color range: [{vmin:.2e}, {vmax:.2e}]")

inner_ny = ny
inner_nx = nx
inner_size_y = inner_ny * dx
inner_size_x = inner_nx * dx

print(f"Wavefield size: {inner_ny} x {inner_nx} ({inner_size_y:.2f} m x {inner_size_x:.2f} m)")

# Initial image - in meters
# Background: true permittivity model
epsilon_bg = epsilon_r.detach().cpu().numpy()
bg = ax.imshow(
    epsilon_bg,
    cmap="cividis",
    vmin=epsilon_bg.min(),
    vmax=epsilon_bg.max(),
    extent=(0, inner_size_x, inner_size_y, 0),
)

# Wavefield overlay
im = ax.imshow(
    snapshots[0],
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
    extent=(0, inner_size_x, inner_size_y, 0),
    alpha=0.65,
)
ax.set_xlabel('x (m)')
ax.set_ylabel('Depth (m)')
title = ax.set_title('Ey Field (Inner Region), t = 0.00 ns')

# Mark source location
ax.plot(src_x*dx, src_y*dx, 'k*', markersize=15, label='Source')

# Mark receiver locations
for i in range(n_receivers):
    rx = (receiver_x_start + i * receiver_spacing) * dx
    ry = receiver_y * dx
    ax.plot(rx, ry, 'gv', markersize=8)
ax.plot([], [], 'gv', markersize=8, label='Receivers')

plt.colorbar(im, label='Ey (V/m)', fraction=0.035, pad=0.02, shrink=0.8)
ax.legend(loc='upper right')

def animate(frame):
    im.set_array(snapshots[frame])
    t_ns = frame * snapshot_interval * dt * 1e9
    title.set_text(f'Ey Field (Inner Region), t = {t_ns:.2f} ns')
    return [im, title]

ani = animation.FuncAnimation(fig, animate, frames=len(snapshots), 
                               interval=50, blit=True)


print("Saving animation to wavefield.gif ...")
ani.save('wavefield.gif', writer='pillow', fps=20)
print("Done!")
