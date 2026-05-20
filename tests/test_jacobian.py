import numpy as np

from dvf_qa.jacobian import jacobian_determinant


def test_identity_displacement_has_unit_jacobian():
    dvf = np.zeros((8, 9, 10, 3), dtype=np.float32)
    jac = jacobian_determinant(dvf, (1.2, 1.3, 2.0))
    np.testing.assert_allclose(jac, 1.0, atol=1e-6)


def test_uniform_x_expansion_matches_scale_factor():
    z, y, x = np.meshgrid(np.arange(8), np.arange(9), np.arange(10), indexing="ij")
    spacing = (2.0, 1.0, 1.0)
    dvf = np.zeros((8, 9, 10, 3), dtype=np.float64)
    dvf[..., 0] = 0.1 * x * spacing[0]
    jac = jacobian_determinant(dvf, spacing)
    np.testing.assert_allclose(jac[1:-1, 1:-1, 1:-1], 1.1, atol=1e-12)

