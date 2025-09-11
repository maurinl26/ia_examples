# Advection Schemes for Atmospheric Modeling

This document provides comprehensive documentation for the advection schemes implemented in the `src/ice3/stencils/advection.py` module. These schemes are designed for atmospheric modeling applications and are optimized using the DaCe (Data-Centric Parallel Programming) framework.

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Background](#mathematical-background)
3. [Implemented Schemes](#implemented-schemes)
4. [Usage Guide](#usage-guide)
5. [Performance Considerations](#performance-considerations)
6. [Stability and Accuracy](#stability-and-accuracy)
7. [Examples](#examples)
8. [References](#references)

## Overview

Advection is a fundamental process in atmospheric modeling that describes the transport of scalar quantities (such as temperature, humidity, or chemical species) by the wind field. The advection equation is:

```
∂φ/∂t + u·∇φ = 0
```

where:
- `φ` is the scalar field being advected
- `u` is the velocity field
- `t` is time

This module implements several numerical schemes to solve this equation, each with different characteristics regarding accuracy, stability, and computational cost.

## Mathematical Background

### The 1D Advection Equation

In one dimension, the advection equation becomes:

```
∂φ/∂t + u ∂φ/∂x = 0
```

### Discretization

Using finite differences, we discretize space and time:
- Spatial grid: `x_i = i·Δx` for `i = 0, 1, ..., N-1`
- Time levels: `t_n = n·Δt` for `n = 0, 1, 2, ...`
- Grid function: `φ_i^n ≈ φ(x_i, t_n)`

### CFL Condition

The Courant-Friedrichs-Lewy (CFL) condition is crucial for stability:

```
CFL = |u|·Δt/Δx ≤ CFL_max
```

Different schemes have different maximum stable CFL numbers.

## Implemented Schemes

### 1. Upwind Scheme (1st Order)

**Function**: `upwind_advection_1d`

**Mathematical Formula**:
```
φ_i^{n+1} = φ_i^n - CFL·(φ_i^n - φ_{i-1}^n)  [if u ≥ 0]
φ_i^{n+1} = φ_i^n - CFL·(φ_{i+1}^n - φ_i^n)  [if u < 0]
```

**Characteristics**:
- **Order**: 1st order accurate in space and time
- **Stability**: Stable for CFL ≤ 1
- **Properties**: Monotonic, diffusive
- **Best for**: Positive-definite quantities, when stability is more important than accuracy

**Advantages**:
- Unconditionally stable for CFL ≤ 1
- Preserves monotonicity
- Simple to implement

**Disadvantages**:
- High numerical diffusion
- Low accuracy

### 2. Centered Difference Scheme (2nd Order)

**Function**: `centered_advection_1d`

**Mathematical Formula**:
```
φ_i^{n+1} = φ_i^n - 0.5·CFL·(φ_{i+1}^n - φ_{i-1}^n)
```

**Characteristics**:
- **Order**: 2nd order accurate in space, 1st order in time
- **Stability**: Conditionally stable, can be unstable
- **Properties**: Non-diffusive but dispersive

**Advantages**:
- Higher accuracy than upwind
- No numerical diffusion

**Disadvantages**:
- Can produce oscillations
- Stability issues without additional treatment

### 3. Lax-Wendroff Scheme (2nd Order)

**Function**: `lax_wendroff_advection_1d`

**Mathematical Formula**:
```
φ_i^{n+1} = φ_i^n - 0.5·CFL·(φ_{i+1}^n - φ_{i-1}^n) + 0.5·CFL²·(φ_{i+1}^n - 2φ_i^n + φ_{i-1}^n)
```

**Characteristics**:
- **Order**: 2nd order accurate in space and time
- **Stability**: Stable for CFL ≤ 1
- **Properties**: Includes both advection and diffusion terms

**Advantages**:
- 2nd order accuracy in both space and time
- Stable for reasonable CFL numbers
- Good balance of accuracy and stability

**Disadvantages**:
- Can produce oscillations near discontinuities
- More complex than upwind

### 4. Flux-Form Advection

**Function**: `flux_form_advection_1d`

**Mathematical Formula**:
```
φ_i^{n+1} = φ_i^n - (Δt/Δx)·(F_{i+1/2} - F_{i-1/2})
```

where `F_{i+1/2}` is the flux at the cell face.

**Characteristics**:
- **Order**: Depends on flux reconstruction (1st order with upwind)
- **Stability**: Stable for CFL ≤ 1
- **Properties**: Exactly conserves mass

**Advantages**:
- Perfect mass conservation
- Suitable for atmospheric applications
- Can be extended to higher orders

**Disadvantages**:
- Requires careful flux reconstruction
- More complex implementation

### 5. WENO5 Scheme (5th Order)

**Function**: `weno5_advection_1d`

**Mathematical Formula**: Complex weighted reconstruction using multiple stencils.

**Characteristics**:
- **Order**: 5th order accurate in smooth regions
- **Stability**: Stable for appropriate CFL numbers
- **Properties**: High accuracy with shock-capturing capability

**Advantages**:
- Very high accuracy
- Handles discontinuities well
- Maintains monotonicity

**Disadvantages**:
- Computationally expensive
- Complex implementation
- Requires larger stencil (5 points)

### 6. Semi-Lagrangian Scheme

**Function**: `semi_lagrangian_advection_1d`

**Mathematical Formula**: Traces particle trajectories backward in time.

**Characteristics**:
- **Order**: Depends on interpolation method
- **Stability**: Unconditionally stable
- **Properties**: Allows large time steps

**Advantages**:
- Unconditionally stable
- Can use large time steps
- Good for operational weather models

**Disadvantages**:
- Requires interpolation
- Can have mass conservation issues
- More complex for irregular grids

### 7. 3D Advection Scheme

**Function**: `advection_3d`

Extends the upwind scheme to three dimensions with separate treatment of each direction.

## Usage Guide

### Basic Usage

```python
import numpy as np
import dace
from src.ice3.stencils.advection import upwind_advection_1d

# Set up domain
IJ, K = 100, 1
dx, dt = 1.0, 0.05
u_const = 1.0

# Create arrays
phi = np.zeros((IJ, K), dtype=np.float64)
u = np.full((IJ, K), u_const, dtype=np.float64)
phi_new = np.zeros((IJ, K), dtype=np.float64)

# Initialize with Gaussian pulse
center = IJ // 2
sigma = 5.0
for i in range(IJ):
    phi[i, 0] = np.exp(-((i - center) / sigma)**2)

# Compile and run
sdfg = upwind_advection_1d.to_sdfg()
csdfg = sdfg.compile()

csdfg(
    phi=phi,
    u=u,
    phi_new=phi_new,
    dx=dx,
    dt=dt,
    IJ=IJ,
    K=K
)
```

### Scheme Selection

Use the utility functions to select appropriate schemes:

```python
from src.ice3.stencils.advection import compute_cfl_number, select_advection_scheme

# Compute CFL number
cfl = compute_cfl_number(u_max=2.0, dt=0.1, dx=1.0)

# Get recommended scheme
scheme_name = select_advection_scheme(cfl=cfl, accuracy_order=2)
print(f"Recommended scheme: {scheme_name}")
```

### Time Stepping Loop

```python
# Time integration loop
n_steps = 100
phi_current = phi_initial.copy()

for step in range(n_steps):
    # Check CFL condition
    u_max = np.abs(u_field).max()
    cfl = compute_cfl_number(u_max, dt, dx)
    
    if cfl > 1.0:
        print(f"Warning: CFL = {cfl:.3f} > 1.0")
    
    # Apply advection scheme
    csdfg(
        phi=phi_current,
        u=u_field,
        phi_new=phi_result,
        dx=dx,
        dt=dt,
        IJ=IJ,
        K=K
    )
    
    # Update for next step
    phi_current = phi_result.copy()
```

## Performance Considerations

### DaCe Optimization

All schemes are implemented as DaCe programs, which provides:
- Automatic parallelization
- GPU acceleration capability
- Memory optimization
- Code generation for different architectures

### Memory Usage

- **1D schemes**: O(IJ × K) memory for each field
- **3D schemes**: O(I × J × K) memory for each field
- **WENO5**: Requires additional temporary arrays

### Computational Complexity

| Scheme | Operations per grid point | Memory accesses |
|--------|---------------------------|-----------------|
| Upwind | ~5 | 3-4 |
| Centered | ~4 | 3 |
| Lax-Wendroff | ~7 | 3 |
| Flux-form | ~8 | 4-5 |
| WENO5 | ~50 | 5 |
| Semi-Lagrangian | ~10 | Variable |

## Stability and Accuracy

### CFL Limits

| Scheme | Maximum CFL | Typical CFL |
|--------|-------------|-------------|
| Upwind | 1.0 | 0.8 |
| Centered | ~0.5 | 0.3 |
| Lax-Wendroff | 1.0 | 0.8 |
| Flux-form | 1.0 | 0.8 |
| WENO5 | ~0.4 | 0.3 |
| Semi-Lagrangian | ∞ | Any |

### Accuracy Comparison

For smooth solutions:
1. **WENO5**: Highest accuracy (5th order)
2. **Lax-Wendroff**: Good accuracy (2nd order)
3. **Centered**: Moderate accuracy (2nd order, but dispersive)
4. **Flux-form**: Depends on reconstruction
5. **Semi-Lagrangian**: Depends on interpolation
6. **Upwind**: Lowest accuracy (1st order, diffusive)

### Conservation Properties

| Scheme | Mass Conservation | Energy Conservation |
|--------|-------------------|-------------------|
| Upwind | Good | Poor (diffusive) |
| Centered | Good | Good |
| Lax-Wendroff | Good | Moderate |
| Flux-form | Exact | Good |
| WENO5 | Good | Good |
| Semi-Lagrangian | Approximate | Moderate |

## Examples

### Example 1: Gaussian Pulse Advection

```python
# Test advection of a Gaussian pulse
def test_gaussian_advection():
    # Parameters
    IJ, K = 200, 1
    dx, dt = 0.5, 0.01
    u_const = 10.0  # m/s
    
    # Check CFL
    cfl = compute_cfl_number(u_const, dt, dx)
    print(f"CFL number: {cfl}")
    
    # Initial condition: Gaussian pulse
    phi_init = np.zeros((IJ, K))
    x_center = IJ // 2
    sigma = 10.0
    
    for i in range(IJ):
        phi_init[i, 0] = np.exp(-((i - x_center) / sigma)**2)
    
    # Run different schemes and compare
    schemes = [
        ("Upwind", upwind_advection_1d),
        ("Lax-Wendroff", lax_wendroff_advection_1d),
        ("WENO5", weno5_advection_1d)
    ]
    
    results = {}
    
    for name, scheme_func in schemes:
        phi_result = run_advection_scheme(
            scheme_func, phi_init, u_const, dx, dt, steps=100
        )
        results[name] = phi_result
        
        # Analyze results
        max_val = phi_result.max()
        mass = phi_result.sum() * dx
        print(f"{name}: Max = {max_val:.4f}, Mass = {mass:.4f}")
```

### Example 2: 3D Atmospheric Transport

```python
def test_3d_atmospheric_transport():
    # Atmospheric domain
    I, J, K = 100, 100, 50  # 100x100x50 grid
    dx = dy = 1000.0  # 1 km horizontal resolution
    dz = 100.0        # 100 m vertical resolution
    dt = 60.0         # 1 minute time step
    
    # Wind field (simplified)
    u = np.full((I, J, K), 10.0)  # 10 m/s eastward
    v = np.full((I, J, K), 5.0)   # 5 m/s northward
    w = np.zeros((I, J, K))       # No vertical motion
    
    # Initial tracer concentration (point source)
    phi = np.zeros((I, J, K))
    phi[I//2, J//2, K//4] = 1000.0  # kg/m³
    
    # Check 3D CFL condition
    u_max = max(np.abs(u).max(), np.abs(v).max(), np.abs(w).max())
    dx_min = min(dx, dy, dz)
    cfl_3d = compute_cfl_number(u_max, dt, dx_min)
    print(f"3D CFL number: {cfl_3d}")
    
    # Run 3D advection
    sdfg = advection_3d.to_sdfg()
    csdfg = sdfg.compile()
    
    phi_new = np.zeros_like(phi)
    
    for step in range(360):  # 6 hours
        csdfg(
            phi=phi,
            u=u, v=v, w=w,
            phi_new=phi_new,
            dx=dx, dy=dy, dz=dz, dt=dt,
            I=I, J=J, K=K
        )
        phi = phi_new.copy()
        
        if step % 60 == 0:  # Every hour
            total_mass = phi.sum() * dx * dy * dz
            max_conc = phi.max()
            print(f"Hour {step//60}: Mass = {total_mass:.2e}, Max = {max_conc:.2e}")
```

### Example 3: Adaptive Scheme Selection

```python
def adaptive_advection_step(phi, u, dx, dt, target_accuracy=2):
    """
    Automatically select and apply the best advection scheme
    based on local conditions.
    """
    # Compute local CFL number
    u_max = np.abs(u).max()
    cfl = compute_cfl_number(u_max, dt, dx)
    
    # Select scheme based on CFL and accuracy requirements
    scheme_name = select_advection_scheme(cfl, target_accuracy)
    
    # Map scheme names to functions
    scheme_map = {
        "upwind": upwind_advection_1d,
        "centered": centered_advection_1d,
        "lax_wendroff": lax_wendroff_advection_1d,
        "weno5": weno5_advection_1d,
        "semi_lagrangian": semi_lagrangian_advection_1d
    }
    
    # Get the appropriate scheme
    scheme_func = scheme_map[scheme_name]
    
    # Compile and run
    sdfg = scheme_func.to_sdfg()
    csdfg = sdfg.compile()
    
    phi_new = np.zeros_like(phi)
    
    # Handle special cases
    if scheme_name == "semi_lagrangian":
        x = np.arange(phi.shape[0], dtype=np.float64) * dx
        csdfg(phi=phi, u=u, phi_new=phi_new, x=x, dx=dx, dt=dt,
              IJ=phi.shape[0], K=phi.shape[1])
    else:
        csdfg(phi=phi, u=u, phi_new=phi_new, dx=dx, dt=dt,
              IJ=phi.shape[0], K=phi.shape[1])
    
    return phi_new, scheme_name
```

## Testing and Validation

The module includes comprehensive tests in `tests/test_advection.py`:

- **Accuracy tests**: Compare with analytical solutions
- **Stability tests**: Verify CFL limits
- **Conservation tests**: Check mass and energy conservation
- **Performance benchmarks**: Compare computational efficiency

Run tests with:
```bash
python -m pytest tests/test_advection.py -v
```

Or run directly:
```bash
python tests/test_advection.py
```

## References

1. **Durran, D. R.** (2010). *Numerical Methods for Fluid Dynamics: With Applications to Geophysics*. Springer.

2. **Strikwerda, J. C.** (2004). *Finite Difference Schemes and Partial Differential Equations*. SIAM.

3. **LeVeque, R. J.** (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.

4. **Shu, C.-W.** (2009). High order weighted essentially non-oscillatory schemes for convection dominated problems. *SIAM Review*, 51(1), 82-126.

5. **Staniforth, A., & Côté, J.** (1991). Semi-Lagrangian integration schemes for atmospheric models—A review. *Monthly Weather Review*, 119(9), 2206-2223.

6. **Williamson, D. L., et al.** (1992). A standard test set for numerical approximations to the shallow water equations in spherical geometry. *Journal of Computational Physics*, 102(1), 211-224.

## Contributing

When adding new advection schemes:

1. Follow the existing function signature patterns
2. Use DaCe decorators (`@dace.program`)
3. Include comprehensive docstrings
4. Add corresponding tests
5. Update this documentation
6. Consider stability and accuracy properties

## License

This code is part of the ICE3-DaCe project and follows the project's licensing terms.
