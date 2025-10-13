"""
JAX translation of ice_adjust.F90 and its dependencies

This module provides a JAX-based implementation of the ICE_ADJUST subroutine
from the PHYEX atmospheric physics package, optimized for automatic differentiation
and GPU acceleration.

Author: Translated from Fortran to Python/JAX
Original Fortran authors: J.-P. Pinty and others (see original headers)
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Optional, Tuple, Dict, Any, NamedTuple
from dataclasses import dataclass
import warnings
import numpy as np

# Import the NumPy version for comparison and testing
from ice_adjust_numpy import (
    PhysicalConstants, DimensionConfig, IceParameters, 
    NebulosityParameters, TurbulenceParameters, create_test_case
)

# JAX configuration for better performance
jax.config.update("jax_enable_x64", True)  # Use double precision

class IceAdjustState(NamedTuple):
    """State variables for ice adjustment"""
    prv: jnp.ndarray  # Water vapor mixing ratio
    prc: jnp.ndarray  # Cloud water mixing ratio
    pri: jnp.ndarray  # Cloud ice mixing ratio
    pth: jnp.ndarray  # Potential temperature
    temperature: jnp.ndarray  # Temperature

class IceAdjustOutput(NamedTuple):
    """Output from ice adjustment"""
    prvs: jnp.ndarray  # Water vapor source
    prcs: jnp.ndarray  # Cloud water source
    pris: jnp.ndarray  # Cloud ice source
    pths: jnp.ndarray  # Theta source
    pcldfr: jnp.ndarray  # Cloud fraction
    picldfr: jnp.ndarray  # Ice cloud fraction
    pwcldfr: jnp.ndarray  # Water cloud fraction
    pssio: jnp.ndarray  # Super-saturation over ice
    pssiu: jnp.ndarray  # Sub-saturation over ice
    pifr: jnp.ndarray  # Ice fraction ratio
    psigrc: jnp.ndarray  # Sigma_rc
    pout_rv: jnp.ndarray  # Adjusted water vapor
    pout_rc: jnp.ndarray  # Adjusted cloud water
    pout_ri: jnp.ndarray  # Adjusted cloud ice
    pout_th: jnp.ndarray  # Adjusted theta

@jit
def esatw_jax(temperature: jnp.ndarray, cst: PhysicalConstants) -> jnp.ndarray:
    """
    Compute saturation vapor pressure over liquid water using JAX
    
    Args:
        temperature: Temperature in Kelvin
        cst: Physical constants
        
    Returns:
        Saturation vapor pressure in Pa
    """
    return jnp.exp(cst.XALPW - cst.XBETAW / temperature - cst.XGAMW * jnp.log(temperature))

@jit
def esati_jax(temperature: jnp.ndarray, cst: PhysicalConstants) -> jnp.ndarray:
    """
    Compute saturation vapor pressure over ice using JAX
    
    Args:
        temperature: Temperature in Kelvin
        cst: Physical constants
        
    Returns:
        Saturation vapor pressure in Pa
    """
    return jnp.exp(cst.XALPI - cst.XBETAI / temperature - cst.XGAMI * jnp.log(temperature))

@jit
def compute_frac_ice_jax(
    frac_ice_scheme: str,
    neb_params: NebulosityParameters,
    pfrac_ice: jnp.ndarray,
    temperature: jnp.ndarray,
    cst: PhysicalConstants
) -> jnp.ndarray:
    """
    Compute ice fraction based on temperature using JAX
    
    Args:
        frac_ice_scheme: Scheme to use ('T', 'O', 'N', 'S')
        neb_params: Nebulosity parameters
        pfrac_ice: Current ice fraction
        temperature: Temperature in Kelvin
        cst: Physical constants
        
    Returns:
        Updated ice fraction
    """
    if frac_ice_scheme == 'T':  # Using Temperature
        return jnp.clip((neb_params.XTMAXMIX - temperature) / (neb_params.XTMAXMIX - neb_params.XTMINMIX), 0.0, 1.0)
    elif frac_ice_scheme == 'O':  # Using Temperature with old formulae
        return jnp.clip((cst.XTT - temperature) / 40.0, 0.0, 1.0)
    elif frac_ice_scheme == 'N':  # No ice
        return jnp.zeros_like(temperature)
    elif frac_ice_scheme == 'S':  # Same as previous
        return jnp.clip(pfrac_ice, 0.0, 1.0)
    else:
        warnings.warn(f"Unknown ice fraction scheme: {frac_ice_scheme}, using 'T'")
        return jnp.clip((neb_params.XTMAXMIX - temperature) / (neb_params.XTMAXMIX - neb_params.XTMINMIX), 0.0, 1.0)

@jit
def compute_latent_heats_jax(
    temperature: jnp.ndarray,
    cst: PhysicalConstants
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute latent heats of vaporization and sublimation using JAX
    
    Args:
        temperature: Temperature in Kelvin
        cst: Physical constants
        
    Returns:
        Tuple of (latent heat of vaporization, latent heat of sublimation)
    """
    lv = cst.XLVTT + (cst.XCPV - cst.XCL) * (temperature - cst.XTT)
    ls = cst.XLSTT + (cst.XCPV - cst.XCI) * (temperature - cst.XTT)
    return lv, ls

@jit
def compute_specific_heat_jax(
    prv: jnp.ndarray,
    prc: jnp.ndarray,
    pri: jnp.ndarray,
    prr: jnp.ndarray,
    prs: jnp.ndarray,
    prg: jnp.ndarray,
    prh: jnp.ndarray,
    cst: PhysicalConstants,
    krr: int
) -> jnp.ndarray:
    """
    Compute specific heat for moist air using JAX
    
    Args:
        prv, prc, pri, prr, prs, prg, prh: Mixing ratios
        cst: Physical constants
        krr: Number of moist variables
        
    Returns:
        Specific heat
    """
    if krr == 7:
        return (cst.XCPD + cst.XCPV * prv + cst.XCL * (prc + prr) + 
                cst.XCI * (pri + prs + prg + prh))
    elif krr == 6:
        return (cst.XCPD + cst.XCPV * prv + cst.XCL * (prc + prr) + 
                cst.XCI * (pri + prs + prg))
    elif krr == 5:
        return (cst.XCPD + cst.XCPV * prv + cst.XCL * (prc + prr) + 
                cst.XCI * (pri + prs))
    elif krr == 3:
        return cst.XCPD + cst.XCPV * prv + cst.XCL * (prc + prr)
    elif krr == 2:
        return cst.XCPD + cst.XCPV * prv + cst.XCL * prc
    else:
        return cst.XCPD + cst.XCPV * prv + cst.XCL * prc

@jit
def gaussian_condensation_jax(
    zsbar: jnp.ndarray,
    zsigma: jnp.ndarray,
    cst: PhysicalConstants
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Gaussian PDF condensation scheme using JAX
    
    Args:
        zsbar: Normalized saturation deficit
        zsigma: Standard deviation
        cst: Physical constants
        
    Returns:
        Tuple of (cloud fraction, condensate)
    """
    # Normalized saturation deficit
    zq1 = zsbar / zsigma
    
    # Gaussian PDF
    zgcond = -zq1 / jnp.sqrt(2.0)
    
    # Use JAX's erf function
    from jax.scipy.special import erf
    zgauv = 1.0 + erf(-zgcond)
    
    # Cloud fraction
    pcldfr = jnp.clip(0.5 * zgauv, 0.0, 1.0)
    
    # Condensate
    exp_term = jnp.exp(-zgcond**2)
    sqrt_pi = jnp.sqrt(cst.XPI)
    sqrt_2pi = jnp.sqrt(2.0 * cst.XPI)
    zcond = (exp_term - zgcond * sqrt_pi * zgauv) * zsigma / sqrt_2pi
    zcond = jnp.maximum(zcond, 0.0)
    
    # Remove very small values
    mask = (zcond < 1e-12) | (pcldfr == 0.0)
    pcldfr = jnp.where(mask, 0.0, pcldfr)
    zcond = jnp.where(mask, 0.0, zcond)
    
    return pcldfr, zcond

@jit
def cb02_condensation_jax(
    zsbar: jnp.ndarray,
    zsigma: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    CB02 condensation scheme using JAX
    
    Args:
        zsbar: Normalized saturation deficit
        zsigma: Standard deviation
        
    Returns:
        Tuple of (cloud fraction, condensate, psigrc)
    """
    # Normalized saturation deficit
    zq1 = zsbar / zsigma
    
    # Total condensate
    zcond = jnp.where(
        zq1 > 0.0,
        jnp.where(
            zq1 <= 2.0,
            jnp.minimum(jnp.exp(-1.0) + 0.66*zq1 + 0.086*zq1**2, 2.0),
            zq1
        ),
        jnp.exp(1.2*zq1 - 1.0)
    )
    zcond = zcond * zsigma
    
    # Cloud fraction
    pcldfr = jnp.where(
        zcond < 1e-12,
        0.0,
        jnp.clip(0.5 + 0.36 * jnp.arctan(1.55 * zq1), 0.0, 1.0)
    )
    
    zcond = jnp.where(pcldfr == 0.0, 0.0, zcond)
    
    # Simplified psigrc computation using lookup table approximation
    # Linear interpolation of the lookup table
    zsrc_1d = jnp.array([
        0.0, 0.0, 2.0094444e-04, 0.316670e-03,
        4.9965648e-04, 0.785956e-03, 1.2341294e-03, 0.193327e-02,
        3.0190963e-03, 0.470144e-02, 7.2950651e-03, 0.112759e-01,
        1.7350994e-02, 0.265640e-01, 4.0427860e-02, 0.610997e-01,
        9.1578111e-02, 0.135888e+00, 0.1991484, 0.230756e+00,
        0.2850565, 0.375050e+00, 0.5000000, 0.691489e+00,
        0.8413813, 0.933222e+00, 0.9772662, 0.993797e+00,
        0.9986521, 0.999768e+00, 0.9999684, 0.999997e+00,
        1.0000000, 1.000000
    ])
    
    # Compute indices for lookup table
    inq1 = jnp.clip(jnp.floor(jnp.clip(2.0 * zq1, -100.0, 100.0)).astype(int), -22, 10)
    zinc = 2.0 * zq1 - inq1
    
    # Offset for 0-based indexing
    idx = inq1 + 22
    idx = jnp.clip(idx, 0, 32)
    
    # Linear interpolation
    psigrc = jnp.minimum(1.0, (1.0 - zinc) * zsrc_1d[idx] + zinc * zsrc_1d[idx + 1])
    
    return pcldfr, zcond, psigrc

@jit
def condensation_jax(
    state: IceAdjustState,
    ppabs: jnp.ndarray,
    pzz: jnp.ndarray,
    prhodref: jnp.ndarray,
    prr: jnp.ndarray,
    prs: jnp.ndarray,
    prg: jnp.ndarray,
    prh: jnp.ndarray,
    psigs: jnp.ndarray,
    cst: PhysicalConstants,
    neb_params: NebulosityParameters,
    krr: int,
    lmfconv: bool = False,
    pmfconv: Optional[jnp.ndarray] = None,
    psigqsat: Optional[jnp.ndarray] = None,
    plv: Optional[jnp.ndarray] = None,
    pls: Optional[jnp.ndarray] = None,
    pcph: Optional[jnp.ndarray] = None
) -> Tuple[IceAdjustState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Main condensation routine using JAX
    
    Args:
        state: Current state variables
        ppabs: Absolute pressure
        pzz: Height
        prhodref: Reference density
        prr, prs, prg, prh: Precipitation mixing ratios
        psigs: Sigma_s from turbulence scheme
        cst: Physical constants
        neb_params: Nebulosity parameters
        krr: Number of moist variables
        lmfconv: Convective mass flux flag
        pmfconv: Convective mass flux
        psigqsat: Qsat variance contribution
        plv, pls, pcph: Optional precomputed values
        
    Returns:
        Tuple of (updated state, cloud fraction, psigrc, ice cloud fraction, 
                 water cloud fraction, ice fraction ratio)
    """
    # Total water mixing ratio
    zrt = state.prv + state.prc + state.pri
    
    # Compute latent heats if not provided
    if plv is None or pls is None:
        zlv, zls = compute_latent_heats_jax(state.temperature, cst)
    else:
        zlv, zls = plv, pls
    
    # Compute specific heat if not provided
    if pcph is None:
        zcph = compute_specific_heat_jax(state.prv, state.prc, state.pri, 
                                       prr, prs, prg, prh, cst, krr)
    else:
        zcph = pcph
    
    # Compute saturation vapor pressures
    zpv = jnp.minimum(esatw_jax(state.temperature, cst), 0.99 * ppabs)
    zpiv = jnp.minimum(esati_jax(state.temperature, cst), 0.99 * ppabs)
    
    # Compute saturation mixing ratios
    zqsl = cst.XRD / cst.XRV * zpv / (ppabs - zpv)
    zqsi = cst.XRD / cst.XRV * zpiv / (ppabs - zpiv)
    
    # Compute ice fraction
    zfrac = jnp.zeros_like(state.temperature)
    if neb_params.CFRAC_ICE_ADJUST == 'S':
        # Use existing fraction
        mask = (state.prc + state.pri) > 1e-20
        zfrac = jnp.where(mask, state.pri / (state.prc + state.pri), 0.0)
    
    zfrac = compute_frac_ice_jax(neb_params.CFRAC_ICE_ADJUST, neb_params, zfrac, state.temperature, cst)
    
    # Interpolate between liquid and solid saturation
    zqsl_mixed = (1.0 - zfrac) * zqsl + zfrac * zqsi
    zlvs = (1.0 - zfrac) * zlv + zfrac * zls
    
    # Compute coefficients a and b
    zah = zlvs * zqsl_mixed / (cst.XRV * state.temperature**2) * (cst.XRV * zqsl_mixed / cst.XRD + 1.0)
    za = 1.0 / (1.0 + zlvs / zcph * zah)
    zb = zah * za
    
    # Compute normalized saturation deficit
    zsbar = za * (zrt - zqsl_mixed + zah * zlvs * (state.prc + state.pri) / zcph)
    
    # Compute sigma
    if neb_params.LSUBG_COND:
        # Use provided sigma_s
        zsigma = jnp.maximum(1e-10, psigs)
        if psigqsat is not None:
            # Add qsat variance contribution
            zsigma_qsat = psigqsat * zqsl_mixed * za
            if neb_params.LSTATNW:
                zsigma = jnp.sqrt(psigs**2 + zsigma_qsat**2)
            else:
                zsigma = jnp.sqrt((2*psigs)**2 + zsigma_qsat**2)
    else:
        # Simplified turbulent variance computation
        zsigma = jnp.full_like(state.temperature, 1e-4)
    
    zsigma = jnp.maximum(zsigma, 1e-10)
    
    # Condensation calculation
    if neb_params.CCONDENS == 'GAUS':
        pcldfr, zcond = gaussian_condensation_jax(zsbar, zsigma, cst)
        psigrc = pcldfr
    elif neb_params.CCONDENS == 'CB02':
        pcldfr, zcond, psigrc = cb02_condensation_jax(zsbar, zsigma)
    else:
        # Default to Gaussian
        pcldfr, zcond = gaussian_condensation_jax(zsbar, zsigma, cst)
        psigrc = pcldfr
    
    # Distribute condensate between liquid and ice
    prc_out = (1.0 - zfrac) * zcond
    pri_out = zfrac * zcond
    
    # Update temperature
    temperature_out = state.temperature + ((prc_out - state.prc) * zlv + 
                                         (pri_out - state.pri) * zls) / zcph
    
    # Update water vapor
    prv_out = zrt - prc_out - pri_out
    
    # Update potential temperature
    pth_out = temperature_out / (ppabs / cst.XP00)**(cst.XRD / cst.XCPD)
    
    # Create updated state
    new_state = IceAdjustState(
        prv=prv_out,
        prc=prc_out,
        pri=pri_out,
        pth=pth_out,
        temperature=temperature_out
    )
    
    # Simplified cloud fractions for now
    picldfr = jnp.zeros_like(pcldfr)
    pwcldfr = pcldfr
    pifr = jnp.full_like(pcldfr, 10.0)
    
    # Lambda3 coefficient
    if neb_params.CLAMBDA3 == 'CB':
        zq1 = zsbar / zsigma
        psigrc = psigrc * jnp.clip(1.0 - zq1, 1.0, 3.0)
    
    return new_state, pcldfr, psigrc, picldfr, pwcldfr, pifr

@jit
def ice_adjust_jax(
    # Input state variables
    prv: jnp.ndarray,
    prc: jnp.ndarray,
    pri: jnp.ndarray,
    pth: jnp.ndarray,
    # Source terms
    prvs: jnp.ndarray,
    prcs: jnp.ndarray,
    pris: jnp.ndarray,
    pths: jnp.ndarray,
    # Precipitation mixing ratios
    prr: jnp.ndarray,
    prs: jnp.ndarray,
    prg: jnp.ndarray,
    prh: jnp.ndarray,
    # Atmospheric state
    ppabs: jnp.ndarray,
    pzz: jnp.ndarray,
    pexn: jnp.ndarray,
    pexnref: jnp.ndarray,
    prhodref: jnp.ndarray,
    psigs: jnp.ndarray,
    # Mass flux parameters
    pcf_mf: jnp.ndarray,
    prc_mf: jnp.ndarray,
    pri_mf: jnp.ndarray,
    pweight_mf_cloud: jnp.ndarray,
    # Control parameters
    ptstep: float,
    krr: int,
    lmfconv: bool,
    # Physical constants and parameters
    cst: PhysicalConstants,
    neb_params: NebulosityParameters,
    # Optional parameters
    psigqsat: Optional[jnp.ndarray] = None,
    pmfconv: Optional[jnp.ndarray] = None
) -> IceAdjustOutput:
    """
    Main ice adjustment routine using JAX
    
    Args:
        prv, prc, pri, pth: State variables to adjust
        prvs, prcs, pris, pths: Source terms
        prr, prs, prg, prh: Precipitation mixing ratios
        ppabs, pzz, pexn, pexnref, prhodref: Atmospheric state
        psigs: Sigma_s from turbulence scheme
        pcf_mf, prc_mf, pri_mf, pweight_mf_cloud: Mass flux parameters
        ptstep: Time step
        krr: Number of moist variables
        lmfconv: Convective mass flux flag
        cst: Physical constants
        neb_params: Nebulosity parameters
        psigqsat: Qsat variance contribution
        pmfconv: Convective mass flux
        
    Returns:
        IceAdjustOutput with all computed fields
    """
    # Convert potential temperature to temperature
    temperature = pth * pexn
    
    # Create initial state
    state = IceAdjustState(
        prv=prv,
        prc=prc,
        pri=pri,
        pth=pth,
        temperature=temperature
    )
    
    # Maximum iterations for adjustment
    itermax = 1
    
    # Iterative adjustment loop
    for jiter in range(itermax):
        # Call condensation routine
        new_state, pcldfr, psigrc, picldfr, pwcldfr, pifr = condensation_jax(
            state, ppabs, pzz, prhodref, prr, prs, prg, prh, psigs,
            cst, neb_params, krr, lmfconv, pmfconv, psigqsat
        )
        state = new_state
    
    # Apply mass flux cloud weighting
    zrc = state.prc * (1.0 - pweight_mf_cloud)
    zri = state.pri * (1.0 - pweight_mf_cloud)
    pcldfr = pcldfr * (1.0 - pweight_mf_cloud)
    psigrc = psigrc * (1.0 - pweight_mf_cloud)
    
    # Compute latent heats for source computation
    zlv, zls = compute_latent_heats_jax(state.temperature, cst)
    
    # Compute specific heat for source computation
    zcph = compute_specific_heat_jax(prv, prc, pri, prr, prs, prg, prh, cst, krr)
    
    # Initialize output source arrays
    prvs_out = prvs
    prcs_out = prcs
    pris_out = pris
    pths_out = pths
    
    # Compute liquid water source
    zw1 = (zrc - prc) / ptstep
    zw1 = jnp.where(zw1 < 0.0, jnp.maximum(zw1, -prcs_out), jnp.minimum(zw1, prvs_out))
    prvs_out = prvs_out - zw1
    prcs_out = prcs_out + zw1
    pths_out = pths_out + zw1 * zlv / (zcph * pexnref)
    
    # Compute ice source
    zw2 = (zri - pri) / ptstep
    zw2 = jnp.where(zw2 < 0.0, jnp.maximum(zw2, -pris_out), jnp.minimum(zw2, prvs_out))
    prvs_out = prvs_out - zw2
    pris_out = pris_out + zw2
    pths_out = pths_out + zw2 * zls / (zcph * pexnref)
    
    # Handle mass flux contributions if present
    if lmfconv:
        # Add mass flux contributions
        zw1_mf = prc_mf / ptstep
        zw2_mf = pri_mf / ptstep
        
        # Limit to available water vapor
        total_mf = zw1_mf + zw2_mf
        mask = total_mf > prvs_out
        scale_factor = jnp.where(mask, prvs_out / total_mf, 1.0)
        zw1_mf = zw1_mf * scale_factor
        zw2_mf = jnp.where(mask, prvs_out - zw1_mf, zw2_mf)
        
        pcldfr = jnp.minimum(1.0, pcldfr + pcf_mf)
        prcs_out = prcs_out + zw1_mf
        pris_out = pris_out + zw2_mf
        prvs_out = prvs_out - (zw1_mf + zw2_mf)
        pths_out = pths_out + (zw1_mf * zlv + zw2_mf * zls) / (zcph * pexnref)
    
    # Compute final cloud fractions
    if not neb_params.LSUBG_COND:
        # All-or-nothing scheme
        mask = (prcs_out + pris_out) > 1e-12 / ptstep
        pcldfr = jnp.where(mask, 1.0, 0.0)
        psigrc = pcldfr
    
    # Simplified additional outputs
    pssio = jnp.zeros_like(pcldfr)
    pssiu = jnp.zeros_like(pcldfr)
    
    return IceAdjustOutput(
        prvs=prvs_out,
        prcs=prcs_out,
        pris=pris_out,
        pths=pths_out,
        pcldfr=pcldfr,
        picldfr=picldfr,
        pwcldfr=pwcldfr,
        pssio=pssio,
        pssiu=pssiu,
        pifr=pifr,
        psigrc=psigrc,
        pout_rv=state.prv,
        pout_rc=state.prc,
        pout_ri=state.pri,
        pout_th=state.pth
    )

class IceAdjustJAX:
    """
    JAX-based ice adjustment class that provides a similar interface to the NumPy version
    """
    
    def __init__(self):
        # Pre-compile the main function for better performance
        self._compiled_ice_adjust = jit(ice_adjust_jax, static_argnums=(10, 11, 12))
    
    def ice_adjust(
        self,
        dim_config: DimensionConfig,
        cst: PhysicalConstants,
        ice_params: IceParameters,
        neb_params: NebulosityParameters,
        turb_params: TurbulenceParameters,
        krr: int,
        ptstep: float,
        psigqsat: jnp.ndarray,
        prhodj: jnp.ndarray,
        pexnref: jnp.ndarray,
        prhodref: jnp.ndarray,
        psigs: Optional[jnp.ndarray],
        lmfconv: bool,
        pmfconv: Optional[jnp.ndarray],
        ppabst: jnp.ndarray,
        pzz: jnp.ndarray,
        pexn: jnp.ndarray,
        pcf_mf: jnp.ndarray,
        prc_mf: jnp.ndarray,
        pri_mf: jnp.ndarray,
        pweight_mf_cloud: jnp.ndarray,
        prv: jnp.ndarray,
        prc: jnp.ndarray,
        prvs: jnp.ndarray,
        prcs: jnp.ndarray,
        pth: jnp.ndarray,
        pths: jnp.ndarray,
        ocompute_src: bool,
        prr: jnp.ndarray,
        pri: jnp.ndarray,
        pris: jnp.ndarray,
        prs: jnp.ndarray,
        prg: jnp.ndarray,
        pice_cld_wgt: Optional[jnp.ndarray] = None,
        prh: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, 
               jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
               Optional[jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray], 
               Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        """
        JAX version of ice_adjust with the same interface as the NumPy version
        
        Args:
            dim_config: Dimension configuration
            cst: Physical constants
            ice_params: Ice microphysics parameters
            neb_params: Nebulosity parameters
            turb_params: Turbulence parameters
            krr: Number of moist variables
            ptstep: Time step
            psigqsat: Coefficient applied to qsat variance contribution
            prhodj: Dry density * Jacobian
            pexnref: Reference Exner function
            prhodref: Reference density
            psigs: Sigma_s from turbulence scheme
            lmfconv: Convective mass flux flag
            pmfconv: Convective mass flux
            ppabst: Absolute pressure
            pzz: Height of model levels
            pexn: Exner function
            pcf_mf: Convective mass flux cloud fraction
            prc_mf: Convective mass flux liquid mixing ratio
            pri_mf: Convective mass flux ice mixing ratio
            pweight_mf_cloud: Weight coefficient for mass-flux cloud
            prv: Water vapor mixing ratio to adjust
            prc: Cloud water mixing ratio to adjust
            prvs: Water vapor mixing ratio source
            prcs: Cloud water mixing ratio source
            pth: Theta to adjust
            pths: Theta source
            ocompute_src: Logical switch to compute sources
            prr: Rain water mixing ratio
            pri: Cloud ice mixing ratio to adjust
            pris: Cloud ice mixing ratio source
            prs: Aggregate mixing ratio
            prg: Graupel mixing ratio
            pice_cld_wgt: Ice cloud weight (optional)
            prh: Hail mixing ratio (optional)
            
        Returns:
            Tuple of updated arrays and diagnostics
        """
        
        # Handle optional arrays
        if prh is None:
            prh = jnp.zeros_like(prr)
        if psigs is None:
            psigs = jnp.full_like(pth, 1e-4)
        
        # Convert NumPy arrays to JAX arrays if needed
        prv_jax = jnp.asarray(prv)
        prc_jax = jnp.asarray(prc)
        pri_jax = jnp.asarray(pri)
        pth_jax = jnp.asarray(pth)
        prvs_jax = jnp.asarray(prvs)
        prcs_jax = jnp.asarray(prcs)
        pris_jax = jnp.asarray(pris)
        pths_jax = jnp.asarray(pths)
        prr_jax = jnp.asarray(prr)
        prs_jax = jnp.asarray(prs)
        prg_jax = jnp.asarray(prg)
        prh_jax = jnp.asarray(prh)
        ppabst_jax = jnp.asarray(ppabst)
        pzz_jax = jnp.asarray(pzz)
        pexn_jax = jnp.asarray(pexn)
        pexnref_jax = jnp.asarray(pexnref)
        prhodref_jax = jnp.asarray(prhodref)
        psigs_jax = jnp.asarray(psigs)
        pcf_mf_jax = jnp.asarray(pcf_mf)
        prc_mf_jax = jnp.asarray(prc_mf)
        pri_mf_jax = jnp.asarray(pri_mf)
        pweight_mf_cloud_jax = jnp.asarray(pweight_mf_cloud)
        psigqsat_jax = jnp.asarray(psigqsat) if psigqsat is not None else None
        pmfconv_jax = jnp.asarray(pmfconv) if pmfconv is not None else None
        
        # Call the compiled JAX function
        result = ice_adjust_jax(
            prv_jax, prc_jax, pri_jax, pth_jax,
            prvs_jax, prcs_jax, pris_jax, pths_jax,
            prr_jax, prs_jax, prg_jax, prh_jax,
            ppabst_jax, pzz_jax, pexn_jax, pexnref_jax, prhodref_jax, psigs_jax,
            pcf_mf_jax, prc_mf_jax, pri_mf_jax, pweight_mf_cloud_jax,
            ptstep, krr, lmfconv,
            cst, neb_params,
            psigqsat_jax, pmfconv_jax
        )
        
        # Convert back to NumPy arrays for compatibility
        return (
            np.asarray(result.prvs),
            np.asarray(result.prcs),
            np.asarray(result.pris),
            np.asarray(result.pths),
            np.asarray(result.pcldfr),
            np.asarray(result.picldfr),
            np.asarray(result.pwcldfr),
            np.asarray(result.pssio),
            np.asarray(result.pssiu),
            np.asarray(result.pifr),
            np.asarray(result.psigrc) if ocompute_src else None,
            np.asarray(result.pout_rv),
            np.asarray(result.pout_rc),
            np.asarray(result.pout_ri),
            np.asarray(result.pout_th)
        )


def create_jax_test_case(nx: int = 10, ny: int = 1, nz: int = 20) -> Dict[str, Any]:
    """
    Create a test case for JAX ice_adjust using the same data as NumPy version
    """
    test_data = create_test_case(nx, ny, nz)
    
    # Convert NumPy arrays to JAX arrays
    jax_test_data = {}
    for key, value in test_data.items():
        if isinstance(value, np.ndarray):
            jax_test_data[key] = jnp.asarray(value)
        else:
            jax_test_data[key] = value
    
    return jax_test_data


if __name__ == "__main__":
    print("Ice Adjust JAX module loaded successfully")
    print("Use IceAdjustJAX().ice_adjust() to run the JAX version")
    print("Use create_jax_test_case() to generate test data")
