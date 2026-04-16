# TIDE Documentation

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **Getting Started**

    ---

    Install TIDE and run your first forward simulation in five minutes.

    [:octicons-arrow-right-24: Getting Started](getting-started.md)

-   :material-map-outline:{ .lg .middle } **API Orientation**

    ---

    Understand the difference between functional and module APIs and choose the right style.

    [:octicons-arrow-right-24: API Orientation](guides/api-orientation.md)

-   :material-chart-line:{ .lg .middle } **Forward & Inversion**

    ---

    Common modeling workflows and inversion loop examples.

    [:octicons-arrow-right-24: Modeling](guides/modeling.md) · [:octicons-arrow-right-24: Inversion](guides/inversion.md)

-   :material-cog-outline:{ .lg .middle } **Configuration Reference**

    ---

    Full reference for storage, callbacks, backend mode, and CFL settings.

    [:octicons-arrow-right-24: Configuration](guides/configuration.md)

</div>

---

## Recommended Reading Order

1. [Getting Started](getting-started.md) — installation and first forward run
2. [API Orientation](guides/api-orientation.md) — functional vs module API choice
3. [Modeling](guides/modeling.md) and [Inversion Workflow](guides/inversion.md) — common workflows
4. [Configuration Reference](guides/configuration.md) — tuning storage, callbacks, backend mode, or CFL settings
5. [Limitations](guides/limitations.md) and [Verification](guides/verification.md) — read before relying on advanced combinations

!!! tip "Version Requirements"
    TIDE requires Python ≥ 3.12 and PyTorch ≥ 2.9.1. For GPU support, install a CUDA-enabled PyTorch build first.
