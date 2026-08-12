---
title: "Inversion Workflows"
description: "Build constrained, testable electromagnetic inversion loops with TIDE and PyTorch."
---

An inversion searches for material parameters whose modeled receiver data match observations. TIDE supplies the differentiable Maxwell operator. User code owns preprocessing, parameterization, objective design, regularization, batching, optimization, and stopping decisions.

## Inverse problem

For observed data $d_{obs}$ and predicted data $F(m)$, a basic objective is

$$
\Phi(m)=\frac{1}{N}\lVert F(m)-d_{obs}\rVert_2^2 + \lambda R(m).
$$

The data term alone rarely defines a unique material model. Acquisition aperture, bandwidth, noise, parameter coupling, and regularization all influence what can be recovered.

## Minimal PyTorch loop

```python
import torch
import tide

epsilon = torch.full((96, 96), 4.0, requires_grad=True)
sigma = torch.zeros_like(epsilon, requires_grad=True)
mu = torch.ones_like(epsilon)

operator = tide.MaxwellTM(
    tide.Discretization(0.02, 4.0e-11, boundary=tide.CPML(10)),
    tide.Experiment(
        tide.Acquisition(source_location, receiver_location),
        source_amplitude,
    ),
    execution=tide.ExecutionOptions(
        fallback=tide.FallbackPolicy.REFERENCE,
    ),
)

optimizer = torch.optim.Adam([epsilon, sigma], lr=1.0e-2)

for iteration in range(50):
    optimizer.zero_grad()
    predicted = operator(tide.EMModel(epsilon, sigma, mu)).receiver_data
    residual = predicted - observed
    loss = residual.square().mean()
    loss.backward()
    optimizer.step()
```

This loop is intentionally incomplete. Direct updates can make `epsilon` non-positive or `sigma` negative, and the two fields may require very different scaling. Production code should parameterize and scale each material field deliberately.

## Bounded material parameterization

```python
class MaterialModel(torch.nn.Module):
    def __init__(self, epsilon0, sigma0):
        super().__init__()
        self.epsilon_raw = torch.nn.Parameter(
            torch.logit((epsilon0 - 1.0) / 11.0)
        )
        self.sigma_raw = torch.nn.Parameter(
            torch.logit((sigma0 + 1.0e-6) / 0.101)
        )

    def forward(self):
        epsilon = 1.0 + 11.0 * torch.sigmoid(self.epsilon_raw)
        sigma = 0.1 * torch.sigmoid(self.sigma_raw)
        return epsilon, sigma
```

The bounds above are example values, not universal physical limits. Choose them from prior information and verify the initial tensors lie strictly inside the interval before applying `logit`.

## Match data before optimizing

Predicted and observed traces must share:

- Time sampling interval and number of samples.
- Shot and receiver ordering.
- Component and polarity convention.
- Source signature and timing convention.
- Preprocessing, filtering, gain, and normalization.

A mismatch in any of these can produce a smooth, finite loss and a misleading gradient. Plot or summarize several colocated traces before the first optimizer step.

## Objective choices

Mean squared error is a useful baseline because its derivative is simple and derivative tests are easy to interpret. It can be sensitive to phase error and outliers. Other supported workflow losses include trace-wise Sinkhorn and graph-space optimal transport variants. These add assumptions and computation, so compare them against MSE on a controlled example before using them as a remedy for cycle skipping.

When normalizing traces, decide whether each trace, each shot, or the full dataset should contribute equally. Per-trace normalization suppresses amplitude information. Global normalization lets energetic shots dominate. Neither choice is neutral.

## Frequency continuation

A common strategy begins with low-frequency content and gradually increases bandwidth. For each stage:

1. Apply the same filter to observed and predicted data.
2. Reset optimizer history when the objective changes substantially.
3. Re-evaluate learning rates and stopping thresholds.
4. Preserve the current model as the initial model for the next stage.

Filtering only the receiver traces is convenient. Filtering the source can also permit a coarser grid at early stages, but then the numerical discretization changes and must be handled consistently.

## Shot batching

For many shots, accumulate gradients over mini-batches:

```python
optimizer.zero_grad()
for shot_indices in tide.workflow.split_shots(n_shots, batch_size=4):
    batch_loss = loss_for_shots(shot_indices)
    batch_loss.backward()
optimizer.step()
```

Scale each batch loss so the accumulated gradient matches the intended full-dataset normalization. `tide.workflow.backward_shot_batches` centralizes this pattern and can run a callback after each backward pass for curvature accumulation.

## Regularization and masks

Useful regularization patterns include:

- Smoothness penalties on `epsilon` or `sigma`.
- Total variation for piecewise-smooth models.
- Reference-model penalties in poorly illuminated regions.
- Fixed masks for air, known infrastructure, or boundaries.
- Cross-parameter constraints when permittivity and conductivity are coupled by prior knowledge.

Apply a mask to the parameterization or gradient consistently. Setting only the displayed update to zero while leaving optimizer momentum active can still change a supposedly fixed cell.

## Gradient checks before inversion

For a direction $v$, verify

$$
\frac{\Phi(m+h v)-\Phi(m-h v)}{2h}
\approx \nabla\Phi(m)^\top v.
$$

Test several decreasing values of $h$. The error should first decrease, then flatten or rise as floating-point cancellation dominates. Run the test with the same backend, dtype, storage policy, component selection, and preprocessing used by the inversion.

## Optimization strategy

Adam or AdamW tolerates noisy mini-batch gradients and is useful while building a workflow. LBFGS can converge quickly for smooth deterministic objectives, but it reevaluates the objective inside a closure and retains history that becomes stale when the loss definition changes. TIDE's `tide.optim` package also provides bounded LBFGS, nonlinear conjugate gradient, steepest descent, CGNR, and truncated Newton routines.

Choose the method after the objective contract is stable. An optimizer cannot correct inaccurate physics, inconsistent traces, or a failed gradient test.

## Monitor what matters

Record at least:

- Objective value and data residual norm.
- Gradient norm for each material field.
- Step norm and accepted step length.
- Minimum and maximum physical parameter values.
- Representative predicted and observed traces.
- Runtime and memory behavior per shot batch.

Stop when validation behavior, gradient size, or model change indicates no useful progress. Reaching an iteration limit is not evidence of convergence.

## Failure modes

### Loss decreases but the model is wrong

The inverse problem may be non-unique, over-parameterized, or fitting preprocessing artifacts. Inspect withheld shots, alternative source-receiver offsets, and model regularity.

### A few cells dominate the gradient

Near-source and near-receiver sensitivity can be much larger than the rest of the domain. Use a physical mask, robust parameterization, gradient clipping, or a tested preconditioner. Do not hide the symptom without identifying its geometric cause.

### Updates become unstable

Check material bounds, optimizer scaling, CFL ratio after each model update, and whether the maximum velocity supplied to the operator remains valid.

### Permittivity and conductivity trade off

The fields affect phase and attenuation differently, but limited bandwidth can still couple them strongly. Start with one parameter, inspect the block curvature, and add the second only when data sensitivity supports it.

Continue with [workflow and optimizers](/tide-GPR/guides/workflow-optim/), [storage](/tide-GPR/guides/storage/), and [verification](/tide-GPR/guides/verification/).
