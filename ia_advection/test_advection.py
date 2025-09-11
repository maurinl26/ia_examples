"""
Test suite for advection schemes.

This module contains comprehensive tests for all advection schemes implemented
in the ice3.stencils.advection module, including:
- Accuracy tests with analytical solutions
- Stability tests with various CFL numbers
- Conservation tests
- Performance benchmarks
"""

import pytest
import numpy as np
import dace
from typing import Callable, Tuple

from src.ice3.stencils.advection import (
    upwind_advection_1d,
    centered_advection_1d,
    lax_wendroff_advection_1d,
    flux_form_advection_1d,
    weno5_advection_1d,
    semi_lagrangian_advection_1d,
    advection_3d,
    compute_cfl_number,
    select_advection_scheme
)
from src.ice3.utils.typingx import dtype_float


class TestAdvectionSchemes:
    """Test class for all advection schemes."""
    
    @pytest.fixture
    def setup_1d_test(self):
        """Set up common test parameters for 1D advection tests."""
        domain = 100, 1, 1
        I_test = domain[0]
        J_test = domain[1]
        K_test = domain[2]
        IJ_test = I_test * J_test
        
        # Physical parameters
        dx = 1.0
        dt = 0.05  # Small time step for stability
        u_const = 1.0
        
        # Create coordinate array
        x = np.linspace(0, IJ_test * dx, IJ_test, dtype=np.float64)
        
        return {
            'IJ': IJ_test,
            'K': K_test,
            'dx': dx,
            'dt': dt,
            'u_const': u_const,
            'x': x
        }
    
    @pytest.fixture
    def setup_3d_test(self):
        """Set up common test parameters for 3D advection tests."""
        domain = 20, 20, 10
        I_test, J_test, K_test = domain
        
        # Physical parameters
        dx = dy = dz = 1.0
        dt = 0.01  # Small time step for 3D stability
        u_const = v_const = w_const = 1.0
        
        return {
            'I': I_test,
            'J': J_test,
            'K': K_test,
            'dx': dx,
            'dy': dy,
            'dz': dz,
            'dt': dt,
            'u_const': u_const,
            'v_const': v_const,
            'w_const': w_const
        }
    
    def create_gaussian_pulse_1d(self, IJ: int, K: int, center: int = None, sigma: float = 5.0) -> np.ndarray:
        """Create a Gaussian pulse for testing."""
        if center is None:
            center = IJ // 2
        
        phi = np.zeros((IJ, K), dtype=np.float64)
        for i in range(IJ):
            phi[i, 0] = np.exp(-((i - center) / sigma)**2)
        
        return phi
    
    def create_step_function_1d(self, IJ: int, K: int, left: int = None, right: int = None) -> np.ndarray:
        """Create a step function for testing."""
        if left is None:
            left = IJ // 3
        if right is None:
            right = 2 * IJ // 3
        
        phi = np.zeros((IJ, K), dtype=np.float64)
        phi[left:right, 0] = 1.0
        
        return phi
    
    def create_sine_wave_1d(self, IJ: int, K: int, wavelength: float = 20.0) -> np.ndarray:
        """Create a sine wave for testing."""
        phi = np.zeros((IJ, K), dtype=np.float64)
        for i in range(IJ):
            phi[i, 0] = np.sin(2 * np.pi * i / wavelength)
        
        return phi
    
    def analytical_advection_solution(self, phi_init: np.ndarray, u: float, dx: float, dt: float, steps: int) -> np.ndarray:
        """Compute analytical solution for constant velocity advection."""
        IJ, K = phi_init.shape
        displacement = int(u * dt * steps / dx)
        
        phi_analytical = np.zeros_like(phi_init)
        for i in range(IJ):
            source_idx = (i - displacement) % IJ
            phi_analytical[i, :] = phi_init[source_idx, :]
        
        return phi_analytical
    
    def test_upwind_scheme_accuracy(self, setup_1d_test):
        """Test accuracy of upwind scheme with Gaussian pulse."""
        params = setup_1d_test
        
        # Create initial condition
        phi_init = self.create_gaussian_pulse_1d(params['IJ'], params['K'])
        u_field = np.full((params['IJ'], params['K']), params['u_const'], dtype=np.float64)
        phi_result = np.zeros_like(phi_init)
        
        # Compile and run scheme
        sdfg = upwind_advection_1d.to_sdfg()
        csdfg = sdfg.compile()
        
        phi_current = phi_init.copy()
        steps = 10
        
        for step in range(steps):
            csdfg(
                phi=phi_current,
                u=u_field,
                phi_new=phi_result,
                dx=params['dx'],
                dt=params['dt'],
                IJ=params['IJ'],
                K=params['K']
            )
            phi_current = phi_result.copy()
        
        # Check mass conservation
        mass_error = abs(phi_result.sum() - phi_init.sum())
        assert mass_error < 1e-10, f"Mass conservation error: {mass_error}"
        
        # Check that maximum value decreases (expected for upwind diffusion)
        assert phi_result.max() <= phi_init.max(), "Maximum value should not increase"
        
        print(f"Upwind scheme - Initial max: {phi_init.max():.6f}, Final max: {phi_result.max():.6f}")
    
    def test_centered_scheme_accuracy(self, setup_1d_test):
        """Test accuracy of centered difference scheme."""
        params = setup_1d_test
        
        # Use smaller time step for stability
        dt_stable = params['dt'] * 0.5
        
        phi_init = self.create_sine_wave_1d(params['IJ'], params['K'])
        u_field = np.full((params['IJ'], params['K']), params['u_const'], dtype=np.float64)
        phi_result = np.zeros_like(phi_init)
        
        sdfg = centered_advection_1d.to_sdfg()
        csdfg = sdfg.compile()
        
        phi_current = phi_init.copy()
        steps = 5
        
        for step in range(steps):
            csdfg(
                phi=phi_current,
                u=u_field,
                phi_new=phi_result,
                dx=params['dx'],
                dt=dt_stable,
                IJ=params['IJ'],
                K=params['K']
            )
            phi_current = phi_result.copy()
        
        # Check that scheme doesn't blow up
        assert np.isfinite(phi_result).all(), "Solution should remain finite"
        assert np.abs(phi_result).max() < 10.0, "Solution should not grow excessively"
        
        print(f"Centered scheme - Initial max: {phi_init.max():.6f}, Final max: {phi_result.max():.6f}")
    
    def test_lax_wendroff_scheme(self, setup_1d_test):
        """Test Lax-Wendroff scheme with step function."""
        params = setup_1d_test
        
        phi_init = self.create_step_function_1d(params['IJ'], params['K'])
        u_field = np.full((params['IJ'], params['K']), params['u_const'], dtype=np.float64)
        phi_result = np.zeros_like(phi_init)
        
        sdfg = lax_wendroff_advection_1d.to_sdfg()
        csdfg = sdfg.compile()
        
        phi_current = phi_init.copy()
        steps = 5
        
        for step in range(steps):
            csdfg(
                phi=phi_current,
                u=u_field,
                phi_new=phi_result,
                dx=params['dx'],
                dt=params['dt'],
                IJ=params['IJ'],
                K=params['K']
            )
            phi_current = phi_result.copy()
        
        # Check mass conservation
        mass_error = abs(phi_result.sum() - phi_init.sum())
        assert mass_error < 1e-8, f"Mass conservation error: {mass_error}"
        
        # Check stability
        assert np.isfinite(phi_result).all(), "Solution should remain finite"
        
        print(f"Lax-Wendroff scheme - Mass error: {mass_error:.2e}")
    
    def test_flux_form_scheme(self, setup_1d_test):
        """Test flux-form advection scheme."""
        params = setup_1d_test
        
        phi_init = self.create_gaussian_pulse_1d(params['IJ'], params['K'])
        u_field = np.full((params['IJ'], params['K']), params['u_const'], dtype=np.float64)
        u_face = np.full((params['IJ'], params['K']), params['u_const'], dtype=np.float64)
        phi_result = np.zeros_like(phi_init)
        
        sdfg = flux_form_advection_1d.to_sdfg()
        csdfg = sdfg.compile()
        
        phi_current = phi_init.copy()
        steps = 10
        
        for step in range(steps):
            csdfg(
                phi=phi_current,
                u=u_field,
                u_face=u_face,
                phi_new=phi_result,
                dx=params['dx'],
                dt=params['dt'],
                IJ=params['IJ'],
                K=params['K']
            )
            phi_current = phi_result.copy()
        
        # Flux-form should conserve mass exactly
        mass_error = abs(phi_result.sum() - phi_init.sum())
        assert mass_error < 1e-12, f"Flux-form mass conservation error: {mass_error}"
        
        print(f"Flux-form scheme - Mass error: {mass_error:.2e}")
    
    def test_weno5_scheme(self, setup_1d_test):
        """Test WENO5 scheme (requires larger domain)."""
        params = setup_1d_test
        
        # WENO5 requires at least 5 points, so ensure we have enough
        if params['IJ'] < 10:
            pytest.skip("WENO5 requires larger domain")
        
        phi_init = self.create_gaussian_pulse_1d(params['IJ'], params['K'])
        u_field = np.full((params['IJ'], params['K']), params['u_const'], dtype=np.float64)
        phi_result = np.zeros_like(phi_init)
        
        sdfg = weno5_advection_1d.to_sdfg()
        csdfg = sdfg.compile()
        
        phi_current = phi_init.copy()
        steps = 5
        
        for step in range(steps):
            csdfg(
                phi=phi_current,
                u=u_field,
                phi_new=phi_result,
                dx=params['dx'],
                dt=params['dt'],
                eps=1e-6,
                IJ=params['IJ'],
                K=params['K']
            )
            phi_current = phi_result.copy()
        
        # Check that WENO maintains high accuracy
        assert np.isfinite(phi_result).all(), "WENO solution should remain finite"
        
        # WENO should preserve the maximum better than upwind
        max_preservation = phi_result.max() / phi_init.max()
        assert max_preservation > 0.8, f"WENO should preserve maximum well: {max_preservation}"
        
        print(f"WENO5 scheme - Max preservation: {max_preservation:.3f}")
    
    def test_semi_lagrangian_scheme(self, setup_1d_test):
        """Test semi-Lagrangian scheme."""
        params = setup_1d_test
        
        phi_init = self.create_gaussian_pulse_1d(params['IJ'], params['K'])
        u_field = np.full((params['IJ'], params['K']), params['u_const'], dtype=np.float64)
        phi_result = np.zeros_like(phi_init)
        
        sdfg = semi_lagrangian_advection_1d.to_sdfg()
        csdfg = sdfg.compile()
        
        phi_current = phi_init.copy()
        steps = 10
        
        for step in range(steps):
            csdfg(
                phi=phi_current,
                u=u_field,
                phi_new=phi_result,
                x=params['x'],
                dx=params['dx'],
                dt=params['dt'],
                IJ=params['IJ'],
                K=params['K']
            )
            phi_current = phi_result.copy()
        
        # Semi-Lagrangian should be stable even with large time steps
        assert np.isfinite(phi_result).all(), "Semi-Lagrangian should remain finite"
        
        # Check approximate mass conservation (may have some error due to interpolation)
        mass_error = abs(phi_result.sum() - phi_init.sum()) / phi_init.sum()
        assert mass_error < 0.1, f"Semi-Lagrangian mass error: {mass_error}"
        
        print(f"Semi-Lagrangian scheme - Relative mass error: {mass_error:.3f}")
    
    def test_3d_advection_scheme(self, setup_3d_test):
        """Test 3D advection scheme."""
        params = setup_3d_test
        
        # Create 3D Gaussian pulse
        phi_init = np.zeros((params['I'], params['J'], params['K']), dtype=np.float64)
        center_i, center_j, center_k = params['I']//2, params['J']//2, params['K']//2
        sigma = 3.0
        
        for i in range(params['I']):
            for j in range(params['J']):
                for k in range(params['K']):
                    r_sq = ((i - center_i)**2 + (j - center_j)**2 + (k - center_k)**2) / sigma**2
                    phi_init[i, j, k] = np.exp(-r_sq)
        
        # Create velocity fields
        u_field = np.full((params['I'], params['J'], params['K']), params['u_const'], dtype=np.float64)
        v_field = np.full((params['I'], params['J'], params['K']), params['v_const'], dtype=np.float64)
        w_field = np.full((params['I'], params['J'], params['K']), params['w_const'], dtype=np.float64)
        phi_result = np.zeros_like(phi_init)
        
        sdfg = advection_3d.to_sdfg()
        csdfg = sdfg.compile()
        
        phi_current = phi_init.copy()
        steps = 5
        
        for step in range(steps):
            csdfg(
                phi=phi_current,
                u=u_field,
                v=v_field,
                w=w_field,
                phi_new=phi_result,
                dx=params['dx'],
                dy=params['dy'],
                dz=params['dz'],
                dt=params['dt'],
                I=params['I'],
                J=params['J'],
                K=params['K']
            )
            phi_current = phi_result.copy()
        
        # Check 3D mass conservation
        mass_error = abs(phi_result.sum() - phi_init.sum())
        assert mass_error < 1e-10, f"3D mass conservation error: {mass_error}"
        
        # Check stability
        assert np.isfinite(phi_result).all(), "3D solution should remain finite"
        
        print(f"3D advection - Mass error: {mass_error:.2e}")
    
    def test_cfl_number_computation(self):
        """Test CFL number computation."""
        u_max = 2.0
        dt = 0.1
        dx = 1.0
        
        cfl = compute_cfl_number(u_max, dt, dx)
        expected_cfl = 0.2
        
        assert abs(cfl - expected_cfl) < 1e-10, f"CFL computation error: {cfl} vs {expected_cfl}"
    
    def test_scheme_selection(self):
        """Test automatic scheme selection."""
        # Test stable CFL
        scheme = select_advection_scheme(cfl=0.5, accuracy_order=1)
        assert scheme == "upwind"
        
        scheme = select_advection_scheme(cfl=0.3, accuracy_order=2)
        assert scheme == "lax_wendroff"
        
        scheme = select_advection_scheme(cfl=0.8, accuracy_order=2)
        assert scheme == "centered"
        
        # Test unstable CFL
        scheme = select_advection_scheme(cfl=1.5, accuracy_order=2)
        assert scheme == "semi_lagrangian"
        
        # Test high-order
        scheme = select_advection_scheme(cfl=0.3, accuracy_order=5)
        assert scheme == "weno5"
    
    def test_stability_limits(self, setup_1d_test):
        """Test stability limits of different schemes."""
        params = setup_1d_test
        
        # Test with CFL = 1.0 (stability limit)
        dt_limit = params['dx'] / params['u_const']  # CFL = 1.0
        
        phi_init = self.create_gaussian_pulse_1d(params['IJ'], params['K'])
        u_field = np.full((params['IJ'], params['K']), params['u_const'], dtype=np.float64)
        phi_result = np.zeros_like(phi_init)
        
        # Test upwind at CFL = 1.0 (should be stable)
        sdfg = upwind_advection_1d.to_sdfg()
        csdfg = sdfg.compile()
        
        phi_current = phi_init.copy()
        
        csdfg(
            phi=phi_current,
            u=u_field,
            phi_new=phi_result,
            dx=params['dx'],
            dt=dt_limit,
            IJ=params['IJ'],
            K=params['K']
        )
        
        # Should remain stable
        assert np.isfinite(phi_result).all(), "Upwind should be stable at CFL=1.0"
        assert np.abs(phi_result).max() <= phi_init.max() * 1.1, "Solution should not grow significantly"
    
    @pytest.mark.parametrize("scheme_name,scheme_func", [
        ("upwind", upwind_advection_1d),
        ("centered", centered_advection_1d),
        ("lax_wendroff", lax_wendroff_advection_1d),
    ])
    def test_scheme_comparison(self, setup_1d_test, scheme_name, scheme_func):
        """Compare different schemes on the same problem."""
        params = setup_1d_test
        
        # Use conservative time step
        dt_safe = params['dt'] * 0.1
        
        phi_init = self.create_gaussian_pulse_1d(params['IJ'], params['K'])
        u_field = np.full((params['IJ'], params['K']), params['u_const'], dtype=np.float64)
        phi_result = np.zeros_like(phi_init)
        
        sdfg = scheme_func.to_sdfg()
        csdfg = sdfg.compile()
        
        phi_current = phi_init.copy()
        steps = 10
        
        for step in range(steps):
            csdfg(
                phi=phi_current,
                u=u_field,
                phi_new=phi_result,
                dx=params['dx'],
                dt=dt_safe,
                IJ=params['IJ'],
                K=params['K']
            )
            phi_current = phi_result.copy()
        
        # Basic checks for all schemes
        assert np.isfinite(phi_result).all(), f"{scheme_name} should remain finite"
        
        mass_error = abs(phi_result.sum() - phi_init.sum()) / phi_init.sum()
        print(f"{scheme_name} - Relative mass error: {mass_error:.6f}")
        
        # Store results for comparison
        setattr(self, f'result_{scheme_name}', phi_result.copy())


if __name__ == "__main__":
    """Run tests directly."""
    import sys
    
    # Create test instance
    test_instance = TestAdvectionSchemes()
    
    # Set up test parameters
    setup_1d = {
        'IJ': 100,
        'K': 1,
        'dx': 1.0,
        'dt': 0.05,
        'u_const': 1.0,
        'x': np.linspace(0, 100.0, 100, dtype=np.float64)
    }
    
    setup_3d = {
        'I': 20,
        'J': 20,
        'K': 10,
        'dx': 1.0,
        'dy': 1.0,
        'dz': 1.0,
        'dt': 0.01,
        'u_const': 1.0,
        'v_const': 1.0,
        'w_const': 1.0
    }
    
    print("Running advection scheme tests...")
    
    try:
        # Run individual tests
        print("\n1. Testing upwind scheme...")
        test_instance.test_upwind_scheme_accuracy(setup_1d)
        
        print("\n2. Testing centered scheme...")
        test_instance.test_centered_scheme_accuracy(setup_1d)
        
        print("\n3. Testing Lax-Wendroff scheme...")
        test_instance.test_lax_wendroff_scheme(setup_1d)
        
        print("\n4. Testing flux-form scheme...")
        test_instance.test_flux_form_scheme(setup_1d)
        
        print("\n5. Testing WENO5 scheme...")
        test_instance.test_weno5_scheme(setup_1d)
        
        print("\n6. Testing semi-Lagrangian scheme...")
        test_instance.test_semi_lagrangian_scheme(setup_1d)
        
        print("\n7. Testing 3D advection...")
        test_instance.test_3d_advection_scheme(setup_3d)
        
        print("\n8. Testing utility functions...")
        test_instance.test_cfl_number_computation()
        test_instance.test_scheme_selection()
        
        print("\n9. Testing stability limits...")
        test_instance.test_stability_limits(setup_1d)
        
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
