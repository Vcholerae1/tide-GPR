"""Demonstrates PyTorch checkpointing to reduce memory usage with TIDE.

This script shows how to use PyTorch's checkpointing feature with TIDE's
Maxwell TM propagator to reduce memory consumption during electromagnetic
wave propagation, at the cost of increased computation time.

The key idea is to split the time propagation into segments:
- For all segments except the last, use torch.utils.checkpoint.checkpoint
- The checkpoint wrapper discards intermediate activations during forward pass
- During backward pass, it recomputes the forward pass to get the activations

Memory savings: ~N times reduction (where N = number of segments)
Computation cost: ~2x forward pass time (due to recomputation)
"""

import torch
import torch.utils.checkpoint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Add parent directory to path for importing tide
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tide import MaxwellTM, ricker, CallbackState


def main():
    # =========================================================================
    # Setup
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters
    ny, nx = 200, 200  # Grid size
    dy, dx = 0.05, 0.05  # Grid spacing (m) - 10 points per wavelength at 300 MHz
    
    # Create a simple two-layer model
    epsilon_r = torch.ones(ny, nx, device=device, dtype=torch.float32)
    epsilon_r[ny//2:, :] = 4.0  # Higher permittivity in lower half
    
    sigma = torch.zeros(ny, nx, device=device, dtype=torch.float32)
    sigma[ny//2:, :] = 0.01  # Some conductivity in lower half
    
    mu_r = torch.ones(ny, nx, device=device, dtype=torch.float32)
    
    # Time parameters
    # CFL condition: dt <= c_max * dx / (v * sqrt(2)) where c_max=0.6
    # For dx=0.05m, v=3e8 m/s: max_dt = 0.6 * 0.05 / (3e8 * sqrt(2)) = 7.07e-11 s
    # Use dt smaller than max_dt to avoid internal upsampling (step_ratio > 1)
    dt = 6e-11  # 0.06 ns (satisfies CFL with step_ratio=1)
    nt = 2000  # Total time steps (total time = 120 ns)
    freq = 400e6  # 300 MHz source (wavelength in air = 1m)
    
    # Source and receiver setup
    n_shots = 4
    n_sources = 1
    n_receivers = 50
    
    # Source at top center
    source_location = torch.zeros(n_shots, n_sources, 2, dtype=torch.long, device=device)
    source_location[:, :, 0] = ny // 8  # y position
    source_location[0, :, 1] = nx // 5  # x position for shot 0
    source_location[1, :, 1] = 2 * nx // 5  # x position for shot 1
    source_location[2, :, 1] = 3 * nx // 5  # x position for shot 2
    source_location[3, :, 1] = 4 * nx // 5  # x position for shot 3

    # Receivers along a line
    receiver_location = torch.zeros(n_shots, n_receivers, 2, dtype=torch.long, device=device)
    receiver_location[:, :, 0] = ny // 4  # y position
    receiver_location[:, :, 1] = torch.linspace(10, nx-10, n_receivers, dtype=torch.long, device=device)
    
    # Source wavelet
    source_amplitude = ricker(freq, nt, dt, 1.0 / freq)
    source_amplitude = source_amplitude.reshape(1, 1, -1).expand(n_shots, n_sources, -1).to(device)
    
    # =========================================================================
    # Create model with gradient tracking
    # =========================================================================
    # We want to optimize epsilon_r
    # Note: Set epsilon_requires_grad=True explicitly for gradient computation
    
    # Create the propagator
    model = MaxwellTM(
        epsilon=epsilon_r,
        sigma=sigma,
        mu=mu_r,
        grid_spacing=[dy, dx],
        epsilon_requires_grad=True,  # Enable gradient for epsilon
    ).to(device)
    
    # =========================================================================
    # Checkpointing setup
    # =========================================================================
    n_segments = 5  # Split into 5 segments
    pml_width = 20
    snapshot_interval = 10  # Save wavefield every 10 steps
    
    # =========================================================================
    # First: Run with true model and capture wavefields to check for reflections
    # =========================================================================
    print("\n" + "="*60)
    print("Step 1: Run with TRUE model to generate observed data")
    print("        and capture wavefields to check for reflections")
    print("="*60)
    
    # True model: with interface
    epsilon_true = torch.ones(ny, nx, device=device, dtype=torch.float32)
    epsilon_true[ny//2:, :] = 4.0  # True interface at y=100
    
    model_true = MaxwellTM(
        epsilon=epsilon_true,
        sigma=sigma,
        mu=mu_r,
        grid_spacing=[dy, dx],
        epsilon_requires_grad=False,
    ).to(device)
    
    # Storage for wavefield snapshots
    true_model_snapshots = []
    
    def save_true_snapshot(state: CallbackState):
        """Save Ey field snapshot from true model."""
        Ey = state.get_wavefield("Ey", view="inner")
        true_model_snapshots.append(Ey[0].clone().cpu().numpy())
        if state.step % 200 == 0:
            max_amp = Ey.abs().max().item()
            print(f"  Step {state.step:4d}/{state.nt} | Max |Ey|: {max_amp:.4e}")
    
    # Run forward with true model
    wavefield_size = [n_shots, ny + 2 * pml_width, nx + 2 * pml_width]
    Ey_obs = torch.zeros(*wavefield_size, device=device, dtype=torch.float32)
    Hx_obs = torch.zeros(*wavefield_size, device=device, dtype=torch.float32)
    Hz_obs = torch.zeros(*wavefield_size, device=device, dtype=torch.float32)
    m_Ey_x_obs = torch.zeros(*wavefield_size, device=device, dtype=torch.float32)
    m_Ey_z_obs = torch.zeros(*wavefield_size, device=device, dtype=torch.float32)
    m_Hx_z_obs = torch.zeros(*wavefield_size, device=device, dtype=torch.float32)
    m_Hz_x_obs = torch.zeros(*wavefield_size, device=device, dtype=torch.float32)
    
    print("Running true model simulation...")
    Ey_obs, Hx_obs, Hz_obs, m_Ey_x_obs, m_Ey_z_obs, m_Hx_z_obs, m_Hz_x_obs, observed_data = model_true(
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        Ey_0=Ey_obs, Hx_0=Hx_obs, Hz_0=Hz_obs,
        m_Ey_x=m_Ey_x_obs, m_Ey_z=m_Ey_z_obs,
        m_Hx_z=m_Hx_z_obs, m_Hz_x=m_Hz_x_obs,
        pml_width=pml_width,
        forward_callback=save_true_snapshot,
        callback_frequency=snapshot_interval,
    )
    print(f"  Saved {len(true_model_snapshots)} wavefield snapshots")
    print(f"  Observed data max: {observed_data.abs().max().item():.4e}")
    
    # =========================================================================
    # Visualize wavefields from true model
    # =========================================================================
    print("\nGenerating wavefield animation from true model...")
    
    fig_wave, ax_wave = plt.subplots(figsize=(10, 10))
    vmax = max(np.abs(s).max() for s in true_model_snapshots) * 0.1
    vmin = -vmax
    
    im = ax_wave.imshow(
        true_model_snapshots[0],
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        extent=(0, nx * dx, ny * dx, 0),
    )
    ax_wave.axhline(y=ny//2*dx, color='white', linestyle='--', linewidth=1, label='Interface')
    ax_wave.axhline(y=ny//4*dx, color='yellow', linestyle=':', linewidth=1, label='Src/Rcv line')
    ax_wave.set_xlabel('x (m)')
    ax_wave.set_ylabel('y (m)')
    title = ax_wave.set_title('True Model - Ey Field, t = 0.00 ns')
    plt.colorbar(im, ax=ax_wave, label='Ey')
    
    def update_frame(frame):
        im.set_array(true_model_snapshots[frame])
        t_ns = frame * snapshot_interval * dt * 1e9
        title.set_text(f'True Model - Ey Field, t = {t_ns:.2f} ns')
        return [im, title]
    
    anim = animation.FuncAnimation(fig_wave, update_frame, frames=len(true_model_snapshots),
                                   interval=50, blit=True)
    anim.save('checkpointing_true_model_wavefield.gif', writer='pillow', fps=20)
    print("  Saved: checkpointing_true_model_wavefield.gif")
    plt.close(fig_wave)
    
    # Save key snapshots
    fig_snaps, axes = plt.subplots(2, 4, figsize=(16, 8))
    snap_indices = np.linspace(0, len(true_model_snapshots)-1, 8, dtype=int)
    for i, idx in enumerate(snap_indices):
        ax = axes[i//4, i%4]
        im = ax.imshow(true_model_snapshots[idx], cmap='RdBu_r', vmin=vmin, vmax=vmax,
                       extent=[0, nx*dx, ny*dx, 0])
        ax.axhline(y=ny//2*dx, color='white', linestyle='--', linewidth=0.5)
        ax.axhline(y=ny//4*dx, color='yellow', linestyle=':', linewidth=0.5)
        t_ns = idx * snapshot_interval * dt * 1e9
        ax.set_title(f't = {t_ns:.1f} ns')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    plt.tight_layout()
    plt.savefig('checkpointing_true_model_snapshots.png', dpi=150)
    print("  Saved: checkpointing_true_model_snapshots.png")
    plt.close(fig_snaps)
    
    # =========================================================================
    # Step 2: Run with initial model (homogeneous) and compute gradient
    # =========================================================================
    print("\n" + "="*60)
    print("Step 2: Run with INITIAL model (homogeneous) for FWI")
    print("="*60)
    
    # Wrapper function for checkpointing
    # Note: For checkpointing, all tensor arguments that need gradients
    # must be passed explicitly
    def propagate_segment(
        source_chunk,
        Ey_0, Hx_0, Hz_0,
        m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x,
    ):
        """Propagate one segment of the simulation."""
        return model(
            dt=dt,
            source_amplitude=source_chunk,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=pml_width,
            Ey_0=Ey_0,
            Hx_0=Hx_0,
            Hz_0=Hz_0,
            m_Ey_x=m_Ey_x,
            m_Ey_z=m_Ey_z,
            m_Hx_z=m_Hx_z,
            m_Hz_x=m_Hz_x,
            python_backend=True
        )
    
    # =========================================================================
    # Compute loss and backward pass
    # =========================================================================
    # observed_data was already generated in Step 1 above
    # Reset epsilon AND sigma to initial guess for inversion
    # IMPORTANT: Both epsilon and sigma interfaces cause reflections!
    model.epsilon.data.fill_(1.0)  # Start with homogeneous epsilon
    model.sigma.data.fill_(0.0)     # Start with zero conductivity (no sigma interface)
    
    # Now run forward with initial model (homogeneous) and compute gradient
    print("\nRunning forward pass with initial model (homogeneous)...")
    Ey = torch.zeros(*wavefield_size, device=device, dtype=torch.float32)
    Hx = torch.zeros(*wavefield_size, device=device, dtype=torch.float32)
    Hz = torch.zeros(*wavefield_size, device=device, dtype=torch.float32)
    m_Ey_x = torch.zeros(*wavefield_size, device=device, dtype=torch.float32)
    m_Ey_z = torch.zeros(*wavefield_size, device=device, dtype=torch.float32)
    m_Hx_z = torch.zeros(*wavefield_size, device=device, dtype=torch.float32)
    m_Hz_x = torch.zeros(*wavefield_size, device=device, dtype=torch.float32)
    simulated_data = torch.zeros(nt, n_shots, n_receivers, device=device, dtype=torch.float32, requires_grad=False)
    
    k = 0
    for seg in range(n_segments):
        chunk_nt = nt // n_segments
        src_chunk = source_amplitude[..., seg*chunk_nt:(seg+1)*chunk_nt]
        
        if seg < n_segments - 1:
            Ey, Hx, Hz, m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x, r_chunk = torch.utils.checkpoint.checkpoint(
                propagate_segment, src_chunk, Ey, Hx, Hz, m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x,
                use_reentrant=False
            )
        else:
            Ey, Hx, Hz, m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x, r_chunk = propagate_segment(
                src_chunk, Ey, Hx, Hz, m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x
            )
        
        simulated_data[k:k+chunk_nt, ...] = r_chunk
        k += chunk_nt
    
    # FWI loss: L2 norm of data residual
    residual = simulated_data - observed_data
    loss = 0.5 * torch.sum(residual ** 2)
    print(f"Loss (data misfit): {loss.item():.6e}")
    
    print("\nRunning backward pass (checkpointed segments will be recomputed)...")
    loss.backward()
    
    # Check gradient
    if model.epsilon.grad is not None:
        grad_norm = model.epsilon.grad.norm().item()
        print(f"Gradient norm w.r.t. epsilon_r: {grad_norm:.6e}")
    else:
        print("No gradient computed (this might indicate an issue)")
    
    # =========================================================================
    # Visualization
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot true model
    ax = axes[0, 0]
    im = ax.imshow(epsilon_true.cpu().numpy(), aspect='auto', cmap='viridis')
    ax.set_title('True Model (ε_r)')
    ax.axhline(y=ny//2, color='white', linestyle='--', linewidth=1, label='Interface')
    ax.axhline(y=ny//4, color='red', linestyle=':', linewidth=1, label='Source/Receiver line')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    # Plot initial model
    ax = axes[0, 1]
    im = ax.imshow(model.epsilon.detach().cpu().numpy(), aspect='auto', cmap='viridis',
                   vmin=1.0, vmax=4.0)
    ax.set_title('Initial Model (homogeneous)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    # Plot observed data (from true model)
    ax = axes[0, 2]
    r_data = observed_data[:, 0, :].detach().cpu().numpy()  # [n_receivers, nt]
    vmax = abs(r_data).max() * 0.1 if abs(r_data).max() > 0 else 1.0
    im = ax.imshow(r_data, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
    ax.set_title('Observed Data (Shot 0, true model)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Receiver')
    plt.colorbar(im, ax=ax)
    
    # Plot simulated data (from initial model)
    ax = axes[1, 0]
    r_data = simulated_data[:, 0, :].detach().cpu().numpy()
    vmax = abs(r_data).max() * 0.1 if abs(r_data).max() > 0 else 1.0
    im = ax.imshow(r_data, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
    ax.set_title('Simulated Data (Shot 0, initial model)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Receiver')
    plt.colorbar(im, ax=ax)
    
    # Plot data residual
    ax = axes[1, 1]
    r_data = residual[:, 0, :].detach().cpu().numpy()
    vmax = abs(r_data).max() * 0.1 if abs(r_data).max() > 0 else 1.0
    im = ax.imshow(r_data, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
    ax.set_title('Data Residual (Observed - Simulated)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Receiver')
    plt.colorbar(im, ax=ax)
    
    # Plot gradient
    ax = axes[1, 2]
    if model.epsilon.grad is not None:
        grad = model.epsilon.grad.cpu().numpy()
        vmax = abs(grad).max() * 0.5
        if vmax > 0:
            im = ax.imshow(grad, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
            ax.axhline(y=ny//2, color='black', linestyle='--', linewidth=1, label='True interface')
            plt.colorbar(im, ax=ax)
    ax.set_title('Gradient w.r.t. ε_r (should show interface)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig('example_checkpointing_result.png', dpi=150)
    print("\nFigure saved to 'example_checkpointing_result.png'")
    plt.show()


if __name__ == "__main__":
    main()
