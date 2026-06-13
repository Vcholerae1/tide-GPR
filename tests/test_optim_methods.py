import numpy as np

import tide


def _quadratic_problem():
    target = np.array([1.0, -2.0, 0.25], dtype=np.float32)
    scale = np.array([3.0, 0.5, 2.0], dtype=np.float32)

    def objective(x: np.ndarray, grad: np.ndarray) -> float:
        residual = x - target
        np.multiply(scale, residual, out=grad)
        return 0.5 * float(np.dot(grad, residual))

    def preconditioner(
        _x: np.ndarray,
        vector: np.ndarray,
        out: np.ndarray,
    ) -> None:
        np.divide(vector, scale, out=out)

    def hessian_vector(
        _x: np.ndarray,
        vector: np.ndarray,
        out: np.ndarray,
    ) -> None:
        np.multiply(scale, vector, out=out)

    x0 = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    return objective, preconditioner, hessian_vector, x0, target


def test_public_api_represents_six_reference_methods_with_generic_preconditioning():
    assert hasattr(tide.optim, "cgnr_solve")
    assert hasattr(tide.optim, "steepest_descent_minimize")
    assert hasattr(tide.optim, "nlcg_minimize")
    assert hasattr(tide.optim, "lbfgs_minimize")
    assert hasattr(tide.optim, "truncated_newton_minimize")
    assert not hasattr(tide.optim, "plbfgs_minimize")
    assert not hasattr(tide.optim, "PLBFGSOptions")


def test_steepest_descent_can_use_optional_preconditioner():
    objective, preconditioner, _, x0, target = _quadratic_problem()

    plain = tide.optim.steepest_descent_minimize(
        objective,
        x0,
        options=tide.optim.SteepestDescentOptions(
            max_iter=40,
            tolerance=1e-8,
            max_evaluations=400,
        ),
    )
    preconditioned = tide.optim.steepest_descent_minimize(
        objective,
        x0,
        preconditioner=preconditioner,
        options=tide.optim.SteepestDescentOptions(
            max_iter=40,
            tolerance=1e-8,
            max_evaluations=400,
        ),
    )

    assert plain.success, plain.status
    assert preconditioned.success, preconditioned.status
    assert plain.n_prec == 0
    assert preconditioned.n_prec > 0
    np.testing.assert_allclose(preconditioned.x, target, atol=1e-6)


def test_cgnr_can_use_optional_preconditioner():
    matrix = np.array(
        [
            [2.0, 0.0, 1.0],
            [0.0, 1.5, -0.5],
            [1.0, -1.0, 2.0],
            [0.5, 0.25, 1.0],
        ],
        dtype=np.float32,
    )
    target = np.array([1.0, -2.0, 0.25], dtype=np.float32)
    data = matrix @ target
    x0 = np.zeros_like(target)
    normal_diag = np.sum(matrix * matrix, axis=0).astype(np.float32)

    def forward(x: np.ndarray, out: np.ndarray) -> None:
        out[:] = matrix @ x

    def adjoint(residual: np.ndarray, out: np.ndarray) -> None:
        out[:] = matrix.T @ residual

    def preconditioner(
        _x: np.ndarray,
        vector: np.ndarray,
        out: np.ndarray,
    ) -> None:
        np.divide(vector, normal_diag, out=out)

    plain = tide.optim.cgnr_solve(
        forward,
        adjoint,
        data,
        x0,
        options=tide.optim.CGNROptions(max_iter=20, tolerance=1e-10),
    )
    preconditioned = tide.optim.cgnr_solve(
        forward,
        adjoint,
        data,
        x0,
        preconditioner=preconditioner,
        options=tide.optim.CGNROptions(max_iter=20, tolerance=1e-10),
    )

    assert plain.success, plain.status
    assert preconditioned.success, preconditioned.status
    assert plain.n_prec == 0
    assert preconditioned.n_prec > 0
    assert plain.n_forward == plain.n_iter + 1
    assert plain.n_adjoint == plain.n_iter + 1
    np.testing.assert_allclose(plain.x, target, atol=2e-5)
    np.testing.assert_allclose(preconditioned.x, target, atol=2e-5)


def test_nlcg_can_use_optional_preconditioner():
    objective, preconditioner, _, x0, target = _quadratic_problem()

    result = tide.optim.nlcg_minimize(
        objective,
        x0,
        preconditioner=preconditioner,
        options=tide.optim.NLCGOptions(
            max_iter=20,
            tolerance=1e-8,
            max_evaluations=400,
        ),
    )

    assert result.success, result.status
    assert result.n_prec > 0
    np.testing.assert_allclose(result.x, target, atol=1e-6)


def test_truncated_newton_can_use_optional_preconditioner():
    objective, preconditioner, hessian_vector, x0, target = _quadratic_problem()

    plain = tide.optim.truncated_newton_minimize(
        objective,
        hessian_vector,
        x0,
        options=tide.optim.TruncatedNewtonOptions(
            max_iter=20,
            max_cg_iter=8,
            tolerance=1e-8,
            max_evaluations=400,
        ),
    )
    preconditioned = tide.optim.truncated_newton_minimize(
        objective,
        hessian_vector,
        x0,
        preconditioner=preconditioner,
        options=tide.optim.TruncatedNewtonOptions(
            max_iter=20,
            max_cg_iter=8,
            tolerance=1e-8,
            max_evaluations=400,
        ),
    )

    assert plain.success, plain.status
    assert preconditioned.success, preconditioned.status
    assert plain.n_prec == 0
    assert plain.n_hess > 0
    assert preconditioned.n_prec > 0
    assert preconditioned.n_hess > 0
    np.testing.assert_allclose(preconditioned.x, target, atol=1e-6)
