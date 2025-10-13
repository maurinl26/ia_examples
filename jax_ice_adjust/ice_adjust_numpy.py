"""
Python/NumPy translation of ice_adjust.F90 and its dependencies

This module provides a NumPy-based implementation of the ICE_ADJUST subroutine
from the PHYEX atmospheric physics package.

Author: Translated from Fortran to Python
Original Fortran authors: J.-P. Pinty and others (see original headers)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import warnings

def esatw(temperature: np.ndarray) -> np.ndarray:
    """
    Compute saturation vapor pressure over liquid water
    
    Args:
        temperature: Temperature in Kelvin
        
    Returns:
        Saturation vapor pressure in Pa
    """
    cst = PhysicalConstants()
    return np.exp(cst.XALPW - cst.XBETAW / temperature - cst.XGAMW * np.log(temperature))

def esati(temperature: np.ndarray) -> np.ndarray:
    """
    Compute saturation vapor pressure over ice
    
    Args:
        temperature: Temperature in Kelvin
        
    Returns:
        Saturation vapor pressure in Pa
    """
    cst = PhysicalConstants()
    return np.exp(cst.XALPI - cst.XBETAI / temperature - cst.XGAMI * np.log(temperature))

def compute_frac_ice(
    frac_ice_scheme: str,
    neb_params: NebulosityParameters,
    pfrac_ice: np.ndarray,
    temperature: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ice fraction based on temperature
    
    Args:
        frac_ice_scheme: Scheme to use ('T', 'O', 'N', 'S')
        neb_params: Nebulosity parameters
        pfrac_ice: Current ice fraction
        temperature: Temperature in Kelvin
        
    Returns:
        Tuple of (updated ice fraction, error codes)
    """
    cst = PhysicalConstants()
    error_codes = np.zeros_like(temperature, dtype=int)
    
    if frac_ice_scheme == 'T':  # Using Temperature
        pfrac_ice = np.maximum(0.0, np.minimum(1.0, 
            (neb_params.XTMAXMIX - temperature) / (neb_params.XTMAXMIX - neb_params.XTMINMIX)))
    elif frac_ice_scheme == 'O':  # Using Temperature with old formulae
        pfrac_ice = np.maximum(0.0, np.minimum(1.0, (cst.XTT - temperature) / 40.0))
    elif frac_ice_scheme == 'N':  # No ice
        pfrac_ice = np.zeros_like(temperature)
    elif frac_ice_scheme == 'S':  # Same as previous
        pfrac_ice = np.maximum(0.0, np.minimum(1.0, pfrac_ice))
    else:
        error_codes[:] = 1
        warnings.warn(f"Unknown ice fraction scheme: {frac_ice_scheme}")
    
    return pfrac_ice, error_codes

def condensation(
    dim_config: DimensionConfig,
    cst: PhysicalConstants,
    ice_params: IceParameters,
    neb_params: NebulosityParameters,
    turb_params: TurbulenceParameters,
    hfrac_ice: str,
    hcondens: str,
    hlambda3: str,
    ppabs: np.ndarray,
    pzz: np.ndarray,
    prhodref: np.ndarray,
    pt: np.ndarray,
    prv_in: np.ndarray,
    prc_in: np.ndarray,
    pri_in: np.ndarray,
    prr: np.ndarray,
    prs: np.ndarray,
    prg: np.ndarray,
    psigs: np.ndarray,
    lmfconv: bool,
    pmfconv: Optional[np.ndarray],
    ouseri: bool,
    osigmas: bool,
    ocnd2: bool,
    psigqsat: np.ndarray,
    plv: Optional[np.ndarray] = None,
    pls: Optional[np.ndarray] = None,
    pcph: Optional[np.ndarray] = None,
    pice_cld_wgt: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Main condensation routine
    
    This is a simplified version focusing on the core condensation physics.
    
    Returns:
        Tuple of (prv_out, prc_out, pri_out, pt_out, pcldfr, psigrc, 
                 picldfr, pwcldfr, pssio, pssiu, pifr)
    """
    
    # Get array dimensions
    nijt, nkt = pt.shape
    iktb = dim_config.NKTB - 1  # Convert to 0-based indexing
    ikte = dim_config.NKTE - 1
    iijb = dim_config.NIJB - 1
    iije = dim_config.NIJE - 1
    
    # Initialize output arrays
    prv_out = np.zeros_like(prv_in)
    prc_out = np.zeros_like(prc_in)
    pri_out = np.zeros_like(pri_in)
    pt_out = pt.copy()
    pcldfr = np.zeros_like(pt)
    psigrc = np.zeros_like(pt)
    picldfr = np.zeros_like(pt)
    pwcldfr = np.zeros_like(pt)
    pssio = np.zeros_like(pt)
    pssiu = np.zeros_like(pt)
    pifr = np.full_like(pt, 10.0)
    
    # Compute total water mixing ratio
    zrt = prv_in + prc_in + pri_in
    
    # Compute latent heats if not provided
    if plv is None:
        zlv = cst.XLVTT + (cst.XCPV - cst.XCL) * (pt_out - cst.XTT)
    else:
        zlv = plv.copy()
        
    if pls is None:
        zls = cst.XLSTT + (cst.XCPV - cst.XCI) * (pt_out - cst.XTT)
    else:
        zls = pls.copy()
    
    # Compute specific heat if not provided
    if pcph is None:
        zcpd = (cst.XCPD + cst.XCPV * prv_in + cst.XCL * prc_in + 
                cst.XCI * pri_in + cst.XCL * prr + cst.XCI * (prs + prg))
    else:
        zcpd = pcph.copy()
    
    # Compute saturation vapor pressures
    zpv = np.minimum(esatw(pt_out), 0.99 * ppabs)
    zpiv = np.minimum(esati(pt_out), 0.99 * ppabs)
    
    # Compute saturation mixing ratios
    zqsl = cst.XRD / cst.XRV * zpv / (ppabs - zpv)
    zqsi = cst.XRD / cst.XRV * zpiv / (ppabs - zpiv)
    
    # Compute ice fraction
    zfrac = np.zeros_like(pt)
    if ouseri and not ocnd2:
        mask = (prc_in + pri_in) > 1e-20
        zfrac[mask] = pri_in[mask] / (prc_in[mask] + pri_in[mask])
        zfrac, _ = compute_frac_ice(hfrac_ice, neb_params, zfrac, pt_out)
    
    # Interpolate between liquid and solid saturation
    zqsl = (1.0 - zfrac) * zqsl + zfrac * zqsi
    zlvs = (1.0 - zfrac) * zlv + zfrac * zls
    
    # Compute coefficients a and b
    zah = zlvs * zqsl / (cst.XRV * pt_out**2) * (cst.XRV * zqsl / cst.XRD + 1.0)
    za = 1.0 / (1.0 + zlvs / zcpd * zah)
    zb = zah * za
    
    # Compute normalized saturation deficit
    zsbar = za * (zrt - zqsl + zah * zlvs * (prc_in + pri_in) / zcpd)
    
    # Compute sigma (simplified version)
    if osigmas:
        zsigma = np.maximum(1e-10, psigs)
    else:
        # Simplified turbulent variance computation
        zsigma = np.full_like(pt, 1e-4)
    
    # Normalized saturation deficit
    zq1 = zsbar / zsigma
    
    # Condensation scheme
    if hcondens == 'GAUS':
        # Gaussian PDF
        zgcond = -zq1 / np.sqrt(2.0)
        from scipy.special import erf
        zgauv = 1.0 + erf(-zgcond)
        
        # Cloud fraction
        pcldfr = np.maximum(0.0, np.minimum(1.0, 0.5 * zgauv))
        
        # Condensate
        zcond = ((np.exp(-zgcond**2) - zgcond * np.sqrt(cst.XPI) * zgauv) * 
                zsigma / np.sqrt(2.0 * cst.XPI))
        zcond = np.maximum(zcond, 0.0)
        
        # Remove very small values
        mask = (zcond < 1e-12) | (pcldfr == 0.0)
        pcldfr[mask] = 0.0
        zcond[mask] = 0.0
        
        psigrc = pcldfr.copy()
        
    elif hcondens == 'CB02':
        # CB02 scheme (simplified)
        zcond = np.where(zq1 > 0.0, 
                        np.where(zq1 <= 2.0,
                                np.minimum(np.exp(-1.0) + 0.66*zq1 + 0.086*zq1**2, 2.0),
                                zq1),
                        np.exp(1.2*zq1 - 1.0))
        zcond *= zsigma
        
        # Cloud fraction
        pcldfr = np.where(zcond < 1e-12, 0.0,
                         np.maximum(0.0, np.minimum(1.0, 0.5 + 0.36*np.arctan(1.55*zq1))))
        
        zcond[pcldfr == 0.0] = 0.0
        
        # Simplified psigrc computation
        psigrc = np.minimum(1.0, np.maximum(0.0, 0.5 + 0.36*np.arctan(1.55*zq1)))
    
    # Distribute condensate between liquid and ice
    if not ocnd2:
        prc_out = (1.0 - zfrac) * zcond
        pri_out = zfrac * zcond
        
        # Update temperature
        pt_out += ((prc_out - prc_in) * zlv + (pri_out - pri_in) * zls) / zcpd
        
        # Update water vapor
        prv_out = zrt - prc_out - pri_out
    else:
        # OCND2 scheme (simplified)
        prc_out = (1.0 - zfrac) * zcond
        pri_out = pri_in.copy()  # Keep ice unchanged in OCND2
        
        pt_out += (prc_out - prc_in) * zlv / zcpd
        prv_out = zrt - prc_out - pri_out
        
        # Simplified cloud fractions for OCND2
        pwcldfr = pcldfr.copy()
        picldfr = np.zeros_like(pcldfr)  # Simplified
    
    # Lambda3 coefficient
    if hlambda3 == 'CB':
        psigrc *= np.minimum(3.0, np.maximum(1.0, 1.0 - zq1))
    
    return (prv_out, prc_out, pri_out, pt_out, pcldfr, psigrc, 
            picldfr, pwcldfr, pssio, pssiu, pifr)

def ice_adjust(
    dim_config: DimensionConfig,
    cst: PhysicalConstants,
    ice_params: IceParameters,
    neb_params: NebulosityParameters,
    turb_params: TurbulenceParameters,
    krr: int,
    ptstep: float,
    psigqsat: np.ndarray,
    prhodj: np.ndarray,
    pexnref: np.ndarray,
    prhodref: np.ndarray,
    psigs: Optional[np.ndarray],
    lmfconv: bool,
    pmfconv: Optional[np.ndarray],
    ppabst: np.ndarray,
    pzz: np.ndarray,
    pexn: np.ndarray,
    pcf_mf: np.ndarray,
    prc_mf: np.ndarray,
    pri_mf: np.ndarray,
    pweight_mf_cloud: np.ndarray,
    prv: np.ndarray,
    prc: np.ndarray,
    prvs: np.ndarray,
    prcs: np.ndarray,
    pth: np.ndarray,
    pths: np.ndarray,
    ocompute_src: bool,
    prr: np.ndarray,
    pri: np.ndarray,
    pris: np.ndarray,
    prs: np.ndarray,
    prg: np.ndarray,
    pice_cld_wgt: Optional[np.ndarray] = None,
    prh: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], 
           Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Main ice adjustment routine
    
    This routine computes the adjustment of water vapor in mixed-phase clouds
    through a saturation adjustment procedure.
    
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
    
    # Get dimensions
    nijt, nkt = pth.shape
    iktb = dim_config.NKTB - 1  # Convert to 0-based indexing
    ikte = dim_config.NKTE - 1
    iijb = dim_config.NIJB - 1
    iije = dim_config.NIJE - 1
    
    # Maximum iterations for adjustment
    itermax = 1
    
    # Initialize working arrays
    zt = pth * pexn  # Temperature
    zrv = prv.copy()
    zrc = prc.copy()
    zri = pri.copy()
    
    # Compute latent heats
    zlv = cst.XLVTT + (cst.XCPV - cst.XCL) * (zt - cst.XTT)
    zls = cst.XLSTT + (cst.XCPV - cst.XCI) * (zt - cst.XTT)
    
    # Compute specific heat
    if krr == 7:
        zcph = (cst.XCPD + cst.XCPV * prv + cst.XCL * (prc + prr) + 
                cst.XCI * (pri + prs + prg + (prh if prh is not None else 0)))
    elif krr == 6:
        zcph = (cst.XCPD + cst.XCPV * prv + cst.XCL * (prc + prr) + 
                cst.XCI * (pri + prs + prg))
    elif krr == 5:
        zcph = (cst.XCPD + cst.XCPV * prv + cst.XCL * (prc + prr) + 
                cst.XCI * (pri + prs))
    elif krr == 3:
        zcph = cst.XCPD + cst.XCPV * prv + cst.XCL * (prc + prr)
    elif krr == 2:
        zcph = cst.XCPD + cst.XCPV * prv + cst.XCL * prc
    else:
        zcph = cst.XCPD + cst.XCPV * prv + cst.XCL * prc
    
    # Iterative adjustment loop
    for jiter in range(itermax):
        if jiter == 0:
            # First iteration: use input values
            prv_in, prc_in, pri_in = prv, prc, pri
        else:
            # Subsequent iterations: use updated values
            prv_in, prc_in, pri_in = zrv, zrc, zri
        
        # Call condensation routine
        if neb_params.LSUBG_COND:
            # Subgrid condensation
            if psigs is None:
                psigs_use = np.zeros_like(pth)
            else:
                psigs_use = psigs
        else:
            # All-or-nothing condensation
            psigs_use = np.zeros_like(pth)
        
        # Call condensation
        (zrv, zrc, zri, zt, pcldfr, psigrc, picldfr, pwcldfr, 
         pssio, pssiu, pifr) = condensation(
            dim_config, cst, ice_params, neb_params, turb_params,
            neb_params.CFRAC_ICE_ADJUST, neb_params.CCONDENS, neb_params.CLAMBDA3,
            ppabst, pzz, prhodref, zt, prv_in, prc_in, pri_in,
            prr, prs, prg, psigs_use, lmfconv, pmfconv,
            True, neb_params.LSIGMAS, False, psigqsat,
            zlv, zls, zcph, pice_cld_wgt
        )
    
    # Apply mass flux cloud weighting
    zrc *= (1.0 - pweight_mf_cloud)
    zri *= (1.0 - pweight_mf_cloud)
    pcldfr *= (1.0 - pweight_mf_cloud)
    psigrc *= (1.0 - pweight_mf_cloud)
    
    # Compute sources
    prvs_out = prvs.copy()
    prcs_out = prcs.copy()
    pris_out = pris.copy()
    pths_out = pths.copy()
    
    # Liquid water source
    zw1 = (zrc - prc) / ptstep
    zw1 = np.where(zw1 < 0.0, np.maximum(zw1, -prcs_out), np.minimum(zw1, prvs_out))
    prvs_out -= zw1
    prcs_out += zw1
    pths_out += zw1 * zlv / (zcph * pexnref)
    
    # Ice source
    zw2 = (zri - pri) / ptstep
    zw2 = np.where(zw2 < 0.0, np.maximum(zw2, -pris_out), np.minimum(zw2, prvs_out))
    prvs_out -= zw2
    pris_out += zw2
    pths_out += zw2 * zls / (zcph * pexnref)
    
    # Handle mass flux contributions if present
    if lmfconv and pmfconv is not None:
        # Add mass flux contributions (simplified)
        zw1_mf = prc_mf / ptstep
        zw2_mf = pri_mf / ptstep
        
        # Limit to available water vapor
        total_mf = zw1_mf + zw2_mf
        mask = total_mf > prvs_out
        zw1_mf[mask] *= prvs_out[mask] / total_mf[mask]
        zw2_mf[mask] = prvs_out[mask] - zw1_mf[mask]
        
        pcldfr = np.minimum(1.0, pcldfr + pcf_mf)
        prcs_out += zw1_mf
        pris_out += zw2_mf
        prvs_out -= (zw1_mf + zw2_mf)
        pths_out += (zw1_mf * zlv + zw2_mf * zls) / zcph / pexnref
    
    # Compute final cloud fractions
    if not neb_params.LSUBG_COND:
        # All-or-nothing scheme
        mask = (prcs_out + pris_out) > 1e-12 / ptstep
        pcldfr = np.where(mask, 1.0, 0.0)
        psigrc = pcldfr.copy()
    
    # Compute optional outputs
    psrcs = psigrc if ocompute_src else None
    pout_rv = zrv
    pout_rc = zrc
    pout_ri = zri
    pout_th = zt / pexn
    
    return (prvs_out, prcs_out, pris_out, pths_out, pcldfr, 
            picldfr, pwcldfr, pssio, pssiu, pifr,
            psrcs, pout_rv, pout_rc, pout_ri, pout_th)


def create_test_case(nx: int = 10, ny: int = 1, nz: int = 20) -> Dict[str, Any]:
    """
    Create a test case for ice_adjust
    
    Args:
        nx: Number of grid points in x direction
        ny: Number of grid points in y direction  
        nz: Number of grid points in z direction
        
    Returns:
        Dictionary containing test data
    """
    
    # Create dimension configuration
    dim_config = DimensionConfig(
        NIT=nx, NIB=1, NIE=nx,
        NJT=ny, NJB=1, NJE=ny,
        NKL=1, NKT=nz, NKLES=nz-2,
        NKA=1, NKU=nz, NKB=2, NKE=nz-1,
        NKTB=2, NKTE=nz-1,
        NIBC=1, NJBC=1, NIEC=nx, NJEC=ny,
        NIJT=nx*ny, NIJB=1, NIJE=nx*ny
    )
    
    # Create parameter objects
    cst = PhysicalConstants()
    ice_params = IceParameters()
    neb_params = NebulosityParameters()
    turb_params = TurbulenceParameters()
    
    # Create test atmospheric profile
    nijt = nx * ny
    
    # Pressure profile (Pa) - typical atmospheric profile
    pressure_levels = np.linspace(100000, 10000, nz)  # 1000 to 100 hPa
    ppabst = np.tile(pressure_levels, (nijt, 1))
    
    # Height profile (m)
    height_levels = np.linspace(0, 16000, nz)  # 0 to 16 km
    pzz = np.tile(height_levels, (nijt, 1))
    
    # Temperature profile (K) - typical atmospheric profile
    temp_surface = 288.0  # K
    lapse_rate = 0.0065  # K/m
    temp_levels = temp_surface - lapse_rate * height_levels
    temp_levels = np.maximum(temp_levels, 200.0)  # Minimum temperature
    
    # Exner function
    pexn = (ppabst / cst.XP00) ** (cst.XRD / cst.XCPD)
    pexnref = pexn.copy()
    
    # Potential temperature
    pth = np.tile(temp_levels, (nijt, 1)) / pexn
    
    # Density
    prhodref = ppabst / (cst.XRD * np.tile(temp_levels, (nijt, 1)))
    prhodj = prhodref.copy()
    
    # Water vapor mixing ratio (kg/kg) - decreasing with height
    prv = np.zeros((nijt, nz))
    for k in range(nz):
        # Typical water vapor profile
        if height_levels[k] < 2000:
            prv[:, k] = 0.015 * np.exp(-height_levels[k] / 2000)
        else:
            prv[:, k] = 0.015 * np.exp(-1.0) * np.exp(-(height_levels[k] - 2000) / 8000)
    
    # Cloud water and ice mixing ratios (kg/kg) - small initial values
    prc = np.full((nijt, nz), 1e-6)
    pri = np.full((nijt, nz), 1e-7)
    
    # Precipitation mixing ratios
    prr = np.full((nijt, nz), 1e-8)
    prs = np.full((nijt, nz), 1e-8)
    prg = np.full((nijt, nz), 1e-9)
    
    # Source terms (initialized to zero)
    prvs = np.zeros((nijt, nz))
    prcs = np.zeros((nijt, nz))
    pris = np.zeros((nijt, nz))
    pths = np.zeros((nijt, nz))
    
    # Mass flux parameters (simplified - no convection)
    pcf_mf = np.zeros((nijt, nz))
    prc_mf = np.zeros((nijt, nz))
    pri_mf = np.zeros((nijt, nz))
    pweight_mf_cloud = np.zeros((nijt, nz))
    
    # Other parameters
    psigqsat = np.ones(nijt) * 0.1
    psigs = np.full((nijt, nz), 1e-4)
    
    return {
        'dim_config': dim_config,
        'cst': cst,
        'ice_params': ice_params,
        'neb_params': neb_params,
        'turb_params': turb_params,
        'krr': 5,  # Number of moist variables
        'ptstep': 300.0,  # 5 minute time step
        'psigqsat': psigqsat,
        'prhodj': prhodj,
        'pexnref': pexnref,
        'prhodref': prhodref,
        'psigs': psigs,
        'lmfconv': False,
        'pmfconv': None,
        'ppabst': ppabst,
        'pzz': pzz,
        'pexn': pexn,
        'pcf_mf': pcf_mf,
        'prc_mf': prc_mf,
        'pri_mf': pri_mf,
        'pweight_mf_cloud': pweight_mf_cloud,
        'prv': prv,
        'prc': prc,
        'prvs': prvs,
        'prcs': prcs,
        'pth': pth,
        'pths': pths,
        'ocompute_src': True,
        'prr': prr,
        'pri': pri,
        'pris': pris,
        'prs': prs,
        'prg': prg
    }


if __name__ == "__main__":
    # This section will be used for testing
    print("Ice Adjust NumPy module loaded successfully")
    print("Use create_test_case() to generate test data")
    print("Use ice_adjust() to run the main routine")
