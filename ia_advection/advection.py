"""
Advection schemes for atmospheric modeling.

This module implements various advection schemes commonly used in atmospheric models:
- Upwind scheme (1st order)
- Centered difference scheme (2nd order)
- Lax-Wendroff scheme (2nd order)
- Semi-Lagrangian scheme
- WENO scheme (5th order)
- Flux-form advection schemes

Author: Generated for atmospheric modeling applications
"""

import dace
import numpy as np
from typing import Optional

from ice3.utils.dims import I, J, K, IJ
from ice3.utils.typingx import dtype_float, dtype_int


@dace.program
def upwind_advection_1d(
    phi: dtype_float[IJ, K],          # Scalar field to advect
    u: dtype_float[IJ, K],            # Velocity field
    phi_new: dtype_float[IJ, K],      # Output advected field
    dx: dtype_float,                  # Grid spacing
    dt: dtype_float                   # Time step
):
    """
    First-order upwind advection scheme in 1D.
    
    Stable but diffusive scheme suitable for positive-definite quantities.
    CFL condition: |u|*dt/dx <= 1
    """
    for ij, k in dace.map[0:IJ, 0:K]:
        # Compute CFL number
        cfl = u[ij, k] * dt / dx
        
        # Upwind differencing based on wind direction
        if u[ij, k] >= 0.0:
            # Forward difference (upwind for positive velocity)
            if ij < IJ - 1:
                phi_new[ij, k] = phi[ij, k] - cfl * (phi[ij, k] - phi[ij-1 if ij > 0 else ij, k])
            else:
                phi_new[ij, k] = phi[ij, k]
        else:
            # Backward difference (upwind for negative velocity)
            if ij > 0:
                phi_new[ij, k] = phi[ij, k] - cfl * (phi[ij+1 if ij < IJ-1 else ij, k] - phi[ij, k])
            else:
                phi_new[ij, k] = phi[ij, k]


@dace.program
def centered_advection_1d(
    phi: dtype_float[IJ, K],          # Scalar field to advect
    u: dtype_float[IJ, K],            # Velocity field
    phi_new: dtype_float[IJ, K],      # Output advected field
    dx: dtype_float,                  # Grid spacing
    dt: dtype_float                   # Time step
):
    """
    Second-order centered difference advection scheme in 1D.
    
    More accurate but can be unstable without proper treatment.
    CFL condition: |u|*dt/dx <= 1
    """
    for ij, k in dace.map[0:IJ, 0:K]:
        cfl = u[ij, k] * dt / dx
        
        # Centered difference with boundary handling
        if ij > 0 and ij < IJ - 1:
            phi_new[ij, k] = phi[ij, k] - 0.5 * cfl * (phi[ij+1, k] - phi[ij-1, k])
        else:
            # Use upwind at boundaries
            if u[ij, k] >= 0.0 and ij > 0:
                phi_new[ij, k] = phi[ij, k] - cfl * (phi[ij, k] - phi[ij-1, k])
            elif u[ij, k] < 0.0 and ij < IJ - 1:
                phi_new[ij, k] = phi[ij, k] - cfl * (phi[ij+1, k] - phi[ij, k])
            else:
                phi_new[ij, k] = phi[ij, k]


@dace.program
def lax_wendroff_advection_1d(
    phi: dtype_float[IJ, K],          # Scalar field to advect
    u: dtype_float[IJ, K],            # Velocity field
    phi_new: dtype_float[IJ, K],      # Output advected field
    dx: dtype_float,                  # Grid spacing
    dt: dtype_float                   # Time step
):
    """
    Lax-Wendroff advection scheme (2nd order in space and time).
    
    Includes both advection and numerical diffusion terms.
    CFL condition: |u|*dt/dx <= 1
    """
    for ij, k in dace.map[0:IJ, 0:K]:
        cfl = u[ij, k] * dt / dx
        
        if ij > 0 and ij < IJ - 1:
            # Lax-Wendroff formula
            phi_new[ij, k] = phi[ij, k] - 0.5 * cfl * (phi[ij+1, k] - phi[ij-1, k]) + \
                           0.5 * cfl * cfl * (phi[ij+1, k] - 2.0 * phi[ij, k] + phi[ij-1, k])
        else:
            # Fallback to upwind at boundaries
            if u[ij, k] >= 0.0 and ij > 0:
                phi_new[ij, k] = phi[ij, k] - cfl * (phi[ij, k] - phi[ij-1, k])
            elif u[ij, k] < 0.0 and ij < IJ - 1:
                phi_new[ij, k] = phi[ij, k] - cfl * (phi[ij+1, k] - phi[ij, k])
            else:
                phi_new[ij, k] = phi[ij, k]


@dace.program
def flux_form_advection_1d(
    phi: dtype_float[IJ, K],          # Scalar field to advect
    u: dtype_float[IJ, K],            # Velocity field at cell centers
    u_face: dtype_float[IJ, K],       # Velocity field at cell faces
    phi_new: dtype_float[IJ, K],      # Output advected field
    dx: dtype_float,                  # Grid spacing
    dt: dtype_float                   # Time step
):
    """
    Flux-form advection scheme using upwind interpolation.
    
    Conserves mass exactly and is suitable for atmospheric applications.
    """
    for ij, k in dace.map[0:IJ, 0:K]:
        flux_left = dtype_float(0.0)
        flux_right = dtype_float(0.0)
        
        # Left face flux
        if ij > 0:
            if u_face[ij-1, k] >= 0.0:
                flux_left = u_face[ij-1, k] * phi[ij-1, k]
            else:
                flux_left = u_face[ij-1, k] * phi[ij, k]
        
        # Right face flux
        if ij < IJ - 1:
            if u_face[ij, k] >= 0.0:
                flux_right = u_face[ij, k] * phi[ij, k]
            else:
                flux_right = u_face[ij, k] * phi[ij+1, k]
        
        # Update using flux difference
        phi_new[ij, k] = phi[ij, k] - (dt / dx) * (flux_right - flux_left)


@dace.program
def weno5_advection_1d(
    phi: dtype_float[IJ, K],          # Scalar field to advect
    u: dtype_float[IJ, K],            # Velocity field
    phi_new: dtype_float[IJ, K],      # Output advected field
    dx: dtype_float,                  # Grid spacing
    dt: dtype_float,                  # Time step
    eps: dtype_float = 1e-6           # Small parameter for WENO weights
):
    """
    Fifth-order WENO (Weighted Essentially Non-Oscillatory) scheme.
    
    High-order accurate scheme that maintains monotonicity.
    Requires at least 5 grid points.
    """
    for ij, k in dace.map[2:IJ-2, 0:K]:
        cfl = u[ij, k] * dt / dx
        
        if u[ij, k] >= 0.0:
            # Positive velocity - use left-biased stencils
            # Three candidate stencils
            q1 = (2.0 * phi[ij-2, k] - 7.0 * phi[ij-1, k] + 11.0 * phi[ij, k]) / 6.0
            q2 = (-phi[ij-1, k] + 5.0 * phi[ij, k] + 2.0 * phi[ij+1, k]) / 6.0
            q3 = (2.0 * phi[ij, k] + 5.0 * phi[ij+1, k] - phi[ij+2, k]) / 6.0
            
            # Smoothness indicators
            beta1 = (13.0/12.0) * (phi[ij-2, k] - 2.0*phi[ij-1, k] + phi[ij, k])**2 + \
                    (1.0/4.0) * (phi[ij-2, k] - 4.0*phi[ij-1, k] + 3.0*phi[ij, k])**2
            beta2 = (13.0/12.0) * (phi[ij-1, k] - 2.0*phi[ij, k] + phi[ij+1, k])**2 + \
                    (1.0/4.0) * (phi[ij-1, k] - phi[ij+1, k])**2
            beta3 = (13.0/12.0) * (phi[ij, k] - 2.0*phi[ij+1, k] + phi[ij+2, k])**2 + \
                    (1.0/4.0) * (3.0*phi[ij, k] - 4.0*phi[ij+1, k] + phi[ij+2, k])**2
            
            # WENO weights
            alpha1 = 0.1 / (eps + beta1)**2
            alpha2 = 0.6 / (eps + beta2)**2
            alpha3 = 0.3 / (eps + beta3)**2
            alpha_sum = alpha1 + alpha2 + alpha3
            
            w1 = alpha1 / alpha_sum
            w2 = alpha2 / alpha_sum
            w3 = alpha3 / alpha_sum
            
            # WENO reconstruction
            phi_face = w1 * q1 + w2 * q2 + w3 * q3
            
        else:
            # Negative velocity - use right-biased stencils
            q1 = (2.0 * phi[ij+2, k] - 7.0 * phi[ij+1, k] + 11.0 * phi[ij, k]) / 6.0
            q2 = (-phi[ij+1, k] + 5.0 * phi[ij, k] + 2.0 * phi[ij-1, k]) / 6.0
            q3 = (2.0 * phi[ij, k] + 5.0 * phi[ij-1, k] - phi[ij-2, k]) / 6.0
            
            # Smoothness indicators (mirrored)
            beta1 = (13.0/12.0) * (phi[ij+2, k] - 2.0*phi[ij+1, k] + phi[ij, k])**2 + \
                    (1.0/4.0) * (phi[ij+2, k] - 4.0*phi[ij+1, k] + 3.0*phi[ij, k])**2
            beta2 = (13.0/12.0) * (phi[ij+1, k] - 2.0*phi[ij, k] + phi[ij-1, k])**2 + \
                    (1.0/4.0) * (phi[ij+1, k] - phi[ij-1, k])**2
            beta3 = (13.0/12.0) * (phi[ij, k] - 2.0*phi[ij-1, k] + phi[ij-2, k])**2 + \
                    (1.0/4.0) * (3.0*phi[ij, k] - 4.0*phi[ij-1, k] + phi[ij-2, k])**2
            
            # WENO weights
            alpha1 = 0.1 / (eps + beta1)**2
            alpha2 = 0.6 / (eps + beta2)**2
            alpha3 = 0.3 / (eps + beta3)**2
            alpha_sum = alpha1 + alpha2 + alpha3
            
            w1 = alpha1 / alpha_sum
            w2 = alpha2 / alpha_sum
            w3 = alpha3 / alpha_sum
            
            # WENO reconstruction
            phi_face = w1 * q1 + w2 * q2 + w3 * q3
        
        # Update using reconstructed value
        phi_new[ij, k] = phi[ij, k] - cfl * (phi_face - phi[ij, k])


@dace.program
def semi_lagrangian_advection_1d(
    phi: dtype_float[IJ, K],          # Scalar field to advect
    u: dtype_float[IJ, K],            # Velocity field
    phi_new: dtype_float[IJ, K],      # Output advected field
    x: dtype_float[IJ],               # Grid coordinates
    dx: dtype_float,                  # Grid spacing
    dt: dtype_float                   # Time step
):
    """
    Semi-Lagrangian advection scheme.
    
    Traces particles backward in time and interpolates.
    Unconditionally stable but requires interpolation.
    """
    for ij, k in dace.map[0:IJ, 0:K]:
        # Departure point (backward trajectory)
        x_dep = x[ij] - u[ij, k] * dt
        
        # Find grid indices for interpolation
        i_dep = dace.int32((x_dep - x[0]) / dx)
        
        # Ensure indices are within bounds
        if i_dep < 0:
            i_dep = 0
        elif i_dep >= IJ - 1:
            i_dep = IJ - 2
        
        # Linear interpolation weight
        alpha = (x_dep - x[i_dep]) / dx
        
        # Interpolate value at departure point
        if i_dep < IJ - 1:
            phi_new[ij, k] = (1.0 - alpha) * phi[i_dep, k] + alpha * phi[i_dep + 1, k]
        else:
            phi_new[ij, k] = phi[i_dep, k]


@dace.program
def advection_3d(
    phi: dtype_float[I, J, K],        # 3D scalar field to advect
    u: dtype_float[I, J, K],          # U-velocity component
    v: dtype_float[I, J, K],          # V-velocity component
    w: dtype_float[I, J, K],          # W-velocity component
    phi_new: dtype_float[I, J, K],    # Output advected field
    dx: dtype_float,                  # Grid spacing in x
    dy: dtype_float,                  # Grid spacing in y
    dz: dtype_float,                  # Grid spacing in z
    dt: dtype_float                   # Time step
):
    """
    3D upwind advection scheme.
    
    Applies upwind differencing in all three spatial dimensions.
    """
    for i, j, k in dace.map[0:I, 0:J, 0:K]:
        # X-direction advection
        if u[i, j, k] >= 0.0 and i > 0:
            flux_x = u[i, j, k] * (phi[i, j, k] - phi[i-1, j, k]) / dx
        elif u[i, j, k] < 0.0 and i < I - 1:
            flux_x = u[i, j, k] * (phi[i+1, j, k] - phi[i, j, k]) / dx
        else:
            flux_x = 0.0
        
        # Y-direction advection
        if v[i, j, k] >= 0.0 and j > 0:
            flux_y = v[i, j, k] * (phi[i, j, k] - phi[i, j-1, k]) / dy
        elif v[i, j, k] < 0.0 and j < J - 1:
            flux_y = v[i, j, k] * (phi[i, j+1, k] - phi[i, j, k]) / dy
        else:
            flux_y = 0.0
        
        # Z-direction advection
        if w[i, j, k] >= 0.0 and k > 0:
            flux_z = w[i, j, k] * (phi[i, j, k] - phi[i, j, k-1]) / dz
        elif w[i, j, k] < 0.0 and k < K - 1:
            flux_z = w[i, j, k] * (phi[i, j, k+1] - phi[i, j, k]) / dz
        else:
            flux_z = 0.0
        
        # Update field
        phi_new[i, j, k] = phi[i, j, k] - dt * (flux_x + flux_y + flux_z)


# Utility functions for CFL condition checking and scheme selection
def compute_cfl_number(u_max: float, dt: float, dx: float) -> float:
    """Compute the CFL number for stability analysis."""
    return abs(u_max) * dt / dx


def select_advection_scheme(cfl: float, accuracy_order: int = 1) -> str:
    """
    Select appropriate advection scheme based on CFL number and desired accuracy.
    
    Args:
        cfl: CFL number
        accuracy_order: Desired order of accuracy (1, 2, or 5)
    
    Returns:
        Recommended scheme name
    """
    if cfl > 1.0:
        return "semi_lagrangian"  # Unconditionally stable
    elif accuracy_order == 1:
        return "upwind"
    elif accuracy_order == 2:
        if cfl < 0.5:
            return "lax_wendroff"
        else:
            return "centered"
    elif accuracy_order >= 5:
        return "weno5"
    else:
        return "upwind"  # Safe fallback


if __name__ == "__main__":
    """
    Example usage and testing of advection schemes.
    """
    import matplotlib.pyplot as plt
    
    # Test parameters
    domain = 100, 1, 1  # 1D test case
    I_test = domain[0]
    J_test = domain[1] 
    K_test = domain[2]
    IJ_test = I_test * J_test
    
    # Physical parameters
    dx = 1.0
    dt = 0.1
    u_const = 1.0
    
    # Check CFL condition
    cfl = compute_cfl_number(u_const, dt, dx)
    print(f"CFL number: {cfl}")
    print(f"Recommended scheme: {select_advection_scheme(cfl, accuracy_order=2)}")
    
    # Create test data
    phi_init = np.zeros((IJ_test, K_test), dtype=np.float64)
    u_field = np.full((IJ_test, K_test), u_const, dtype=np.float64)
    phi_result = np.zeros((IJ_test, K_test), dtype=np.float64)
    
    # Initialize with a Gaussian pulse
    x_center = IJ_test // 2
    sigma = 5.0
    for i in range(IJ_test):
        phi_init[i, 0] = np.exp(-((i - x_center) / sigma)**2)
    
    # Test upwind scheme
    print("Testing upwind advection scheme...")
    sdfg_upwind = upwind_advection_1d.to_sdfg()
    csdfg_upwind = sdfg_upwind.compile()
    
    phi_upwind = phi_init.copy()
    csdfg_upwind(
        phi=phi_upwind,
        u=u_field,
        phi_new=phi_result,
        dx=dx,
        dt=dt,
        IJ=IJ_test,
        K=K_test
    )
    
    print(f"Initial max: {phi_init.max():.6f}")
    print(f"Final max: {phi_result.max():.6f}")
    print(f"Mass conservation error: {abs(phi_result.sum() - phi_init.sum()):.6e}")
    
    # Save SDFG for visualization
    sdfg_upwind.save("sdfg/advection_upwind.sdfg")
    
    print("Advection schemes implemented successfully!")
