# Inversion Workflows

Typical inversion loop:

1. Build initial epsilon/sigma model tensors with requires_grad=True.
2. Run forward propagation to generate synthetic receiver data.
3. Compute data misfit between synthetic and observed traces.
4. Backpropagate and update model with optimizer.
5. Repeat over iterations and frequency stages.

## Minimal PyTorch Inversion Skeleton

```python
import torch
import tide

# Assume `src`, `src_loc`, `rec_loc`, and `observed` are prepared first.
epsilon = torch.full((96, 96), 4.0, requires_grad=True)
sigma = torch.zeros_like(epsilon, requires_grad=True)
mu = torch.ones_like(epsilon)
optimizer = torch.optim.Adam([epsilon, sigma], lr=1e-2)

for _ in range(10):
    optimizer.zero_grad()
    pred = tide.maxwelltm(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=src,
        source_location=src_loc,
        receiver_location=rec_loc,
        pml_width=10,
    )[-1]
    loss = (pred - observed).pow(2).mean()
    loss.backward()
    optimizer.step()
```

See `docs/examples/example_multiscale_joint_eps_sigma.md` for a larger staged workflow and `docs/examples/example_multiscale_crosscorr_wrong_wavelet.md` for a wavelet-mismatch-aware objective.

## Objective Functions

Common choices:
- L2 trace misfit: mean((pred - obs)^2)
- Huber/robust losses for outlier-heavy data
- Normalized misfit by trace energy to balance shot amplitudes

## Optimization

Common strategy:
- AdamW in early stages for stability
- LBFGS in late stages for sharper convergence

Multi-stage schedule usually progresses from low to high bandwidth data.

## Regularization

Useful controls:
- spatial smoothing on gradient or model update
- positivity constraints on epsilon and mu
- clipping for sigma range
- region masks to focus updates

## Practical Tips

- Start with smaller models and shot counts to validate gradients.
- Tune model_gradient_sampling_interval and storage_mode to prevent OOM.
- Validate source/receiver indexing before long inversion runs.
- Read `guides/configuration.md`, `guides/limitations.md`, and `guides/verification.md` before enabling advanced modes broadly.
