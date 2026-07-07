# TIDE Documentation

<div class="tide-home">
  <p class="tide-intro">
    TIDE is a PyTorch-first electromagnetic FDTD library for forward modeling, inversion workflows, and CUDA-backed Maxwell solvers.
  </p>

  <section class="tide-path">
    <a class="tide-path__item" href="getting-started/">
      <span class="tide-path__step">01</span>
      <strong>Run the first simulation</strong>
      <span>Install TIDE, verify the backend, and produce one 2D receiver trace.</span>
    </a>
    <a class="tide-path__item" href="guides/api-orientation/">
      <span class="tide-path__step">02</span>
      <strong>Pick the right API layer</strong>
      <span>Choose functional calls for scripts or module APIs for reusable inversion loops.</span>
    </a>
    <a class="tide-path__item" href="guides/modeling/">
      <span class="tide-path__step">03</span>
      <strong>Build modeling workflows</strong>
      <span>Set up sources, receivers, boundaries, and tensor layouts for 2D or 3D runs.</span>
    </a>
  </section>

  <section class="tide-section">
    <div class="tide-section__heading">
      <p class="tide-eyebrow">Documentation map</p>
      <h2>Read by task, not by file name.</h2>
    </div>

    <div class="tide-doc-grid">
      <a class="tide-doc-card" href="overview/">
        <span class="tide-doc-card__icon">2D</span>
        <strong>Project overview</strong>
        <span>Capabilities, tensor conventions, coordinate layout, and the end-to-end workflow.</span>
      </a>
      <a class="tide-doc-card" href="guides/inversion/">
        <span class="tide-doc-card__icon">grad</span>
        <strong>Inversion workflow</strong>
        <span>Connect forward traces, losses, backpropagation, and optimizer updates.</span>
      </a>
      <a class="tide-doc-card" href="guides/configuration/">
        <span class="tide-doc-card__icon">cfg</span>
        <strong>Configuration reference</strong>
        <span>Tune storage, callbacks, backend mode, CFL behavior, and runtime controls.</span>
      </a>
      <a class="tide-doc-card" href="api/">
        <span class="tide-doc-card__icon">api</span>
        <strong>API reference</strong>
        <span>Top-level functions, Maxwell modules, wavelets, callbacks, utilities, and C/CUDA notes.</span>
      </a>
    </div>
  </section>

  <section class="tide-section tide-split">
    <div>
      <p class="tide-eyebrow">Before scaling up</p>
      <h2>Check the constraints that affect correctness.</h2>
      <p>
        Advanced electromagnetic runs are sensitive to layout, stability, backend availability, and memory policy. These pages are the checkpoints to read before relying on larger 3D or inversion jobs.
      </p>
    </div>
    <div class="tide-checklist">
      <ul>
        <li><a href="guides/validation/">Stability and validation</a>: CFL behavior, resampling, and runtime checks</li>
        <li><a href="guides/storage/">Storage and checkpointing</a>: memory tradeoffs for forward and backward paths</li>
        <li><a href="guides/performance/">Performance tips</a>: backend selection and scaling guidance</li>
        <li><a href="guides/limitations/">Limitations</a>: supported combinations and known constraints</li>
        <li><a href="guides/verification/">Verification</a>: tests to run before trusting advanced setups</li>
      </ul>
    </div>
  </section>

  <div class="admonition tip tide-version">
    <p class="admonition-title">Version requirements</p>
    <p>TIDE requires Python >= 3.12 and PyTorch >= 2.12. For GPU support, install a CUDA-enabled PyTorch build first.</p>
  </div>
</div>
