# Inversion Workflows

Typical inversion loop:

1. Build initial epsilon/sigma model tensors with requires_grad=True.
2. Run forward propagation to generate synthetic receiver data.
3. Compute data misfit between synthetic and observed traces.
4. Backpropagate and update model with optimizer.
5. Repeat over iterations and frequency stages.

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
