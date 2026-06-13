"""Conv-only inversion entry.

与 `different_wavelet_type_norm_only.py` 共享同一套参数与流程，
这里只固定使用“只卷积、不归一化”的目标函数。

目标函数（conv-only）:
    L = - < d_obs * s_pred, d_pred * s_obs >
其中:
    s_obs = sum_i d_obs_i
    s_pred = sum_i d_pred_i
"""

from different_wavelet_type_norm_only import run_inversion as _run_inversion


def run_inversion_conv_only() -> None:
    """Run inversion with conv-only objective."""
    _run_inversion("conv_only")


def main() -> None:
    run_inversion_conv_only()


if __name__ == "__main__":
    main()
