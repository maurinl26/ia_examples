"""
DaCe translation of ice_adjust.F90 and its dependencies

This module provides a DaCe-based implementation of the ICE_ADJUST subroutine
from the PHYEX atmospheric physics package, optimized for high-performance computing.

Author: Translated from Fortran to Python/DaCe
Original Fortran authors: J.-P. Pinty and others (see original headers)
"""

import numpy as np
import dace
from typing import Optional, Tuple, Dict, Any

# Import the NumPy version for comparison and testing
from jax_ice_adjust.ice_adjust_numpy import create_test_case
from ice_adjust_numpy import (
    PhysicalConstants, DimensionConfig, IceParameters, 
    NebulosityParameters, TurbulenceParameters
)

# DaCe data types
float64 = dace.float64
int32 = dace.int32

@dace.program
def compute_saturation_pressures(temperature: dace.float64[dace.symbol('N'), dace.symbol('K')],
                                alpw: dace.float64, betaw: dace.float64, gamw: dace.float64,
                                alpi: dace.float64, betai: dace.float64, gami: dace.float64,
                                pressure: dace.float64[dace.symbol('N'), dace.symbol('K')]) -> Tuple[
    dace.float64[dace.symbol('N'), dace.symbol('K')], 
    dace.float64[dace.symbol('N'), dace.symbol('K')]]:
    """
    Compute saturation vapor pressures over liquid water and ice
    """
    N, K = temperature.shape
    
    # Output arrays
    esatw = np.zeros((N, K), dtype=np.float64)
    esati = np.zeros((N, K), dtype=np.float64)
    
    for i, j in dace.map[0:N, 0:K]:
        # Saturation pressure over liquid water
        esatw_val = dace.math.exp(alpw - betaw / temperature[i, j] - gamw * dace.math.log(temperature[i, j]))
        esatw[i, j] = min(esatw_val, 0.99 * pressure[i, j])
        
        # Saturation pressure over ice
        esati_val = dace.math.exp(alpi - betai / temperature[i, j] - gami * dace.math.log(temperature[i, j]))
        esati[i, j] = min(esati_val, 0.99 * pressure[i, j])
    
    return esatw, esati

@dace.program
def compute_ice_fraction(temperature: dace.float64[dace.symbol('N'), dace.symbol('K')],
                        prc_in: dace.float64[dace.symbol('N'), dace.symbol('K')],
                        pri_in: dace.float64[dace.symbol('N'), dace.symbol('K')],
                        scheme: dace.int32,  # 0=T, 1=O, 2=N, 3=S
                        xtt: dace.float64,
                        xtmaxmix: dace.float64,
                        xtminmix: dace.float64) -> dace.float64[dace.symbol('N'), dace.symbol('K')]:
    """
    Compute ice fraction based on temperature and scheme
    """
    N, K = temperature.shape
    frac_ice = np.zeros((N, K), dtype=np.float64)
    
    for i, j in dace.map[0:N, 0:K]:
        if scheme == 0:  # 'T' - using Temperature
            frac_ice[i, j] = max(0.0, min(1.0, (xtmaxmix - temperature[i, j]) / (xtmaxmix - xtminmix)))
        elif scheme == 1:  # 'O' - using Temperature with old formulae
            frac_ice[i, j] = max(0.0, min(1.0, (xtt - temperature[i, j]) / 40.0))
        elif scheme == 2:  # 'N' - No ice
            frac_ice[i, j] = 0.0
        elif scheme == 3:  # 'S' - Same as previous
            if prc_in[i, j] + pri_in[i, j] > 1e-20:
                current_frac = pri_in[i, j] / (prc_in[i, j] + pri_in[i, j])
                frac_ice[i, j] = max(0.0, min(1.0, current_frac))
            else:
                frac_ice[i, j] = 0.0
    
    return frac_ice

@dace.program
def compute_latent_heats(temperature: dace.float64[dace.symbol('N'), dace.symbol('K')],
                        xlvtt: dace.float64, xcpv: dace.float64, xcl: dace.float64,
                        xlstt: dace.float64, xci: dace.float64, xtt: dace.float64) -> Tuple[
    dace.float64[dace.symbol('N'), dace.symbol('K')], 
    dace.float64[dace.symbol('N'), dace.symbol('K')]]:
    """
    Compute latent heats of vaporization and sublimation
    """
    N, K = temperature.shape
    
    lv = np.zeros((N, K), dtype=np.float64)
    ls = np.zeros((N, K), dtype=np.float64)
    
    for i, j in dace.map[0:N, 0:K]:
        lv[i, j] = xlvtt + (xcpv - xcl) * (temperature[i, j] - xtt)
        ls[i, j] = xlstt + (xcpv - xci) * (temperature[i, j] - xtt)
    
    return lv, ls

@dace.program
def compute_specific_heat(prv: dace.float64[dace.symbol('N'), dace.symbol('K')],
                         prc: dace.float64[dace.symbol('N'), dace.symbol('K')],
                         pri: dace.float64[dace.symbol('N'), dace.symbol('K')],
                         prr: dace.float64[dace.symbol('N'), dace.symbol('K')],
                         prs: dace.float64[dace.symbol('N'), dace.symbol('K')],
                         prg: dace.float64[dace.symbol('N'), dace.symbol('K')],
                         prh: dace.float64[dace.symbol('N'), dace.symbol('K')],
                         xcpd: dace.float64, xcpv: dace.float64, 
                         xcl: dace.float64, xci: dace.float64,
                         krr: dace.int32) -> dace.float64[dace.symbol('N'), dace.symbol('K')]:
    """
    Compute specific heat for moist air
    """
    N, K = prv.shape
    cph = np.zeros((N, K), dtype=np.float64)
    
    for i, j in dace.map[0:N, 0:K]:
        if krr == 7:
            cph[i, j] = (xcpd + xcpv * prv[i, j] + xcl * (prc[i, j] + prr[i, j]) + 
                        xci * (pri[i, j] + prs[i, j] + prg[i, j] + prh[i, j]))
        elif krr == 6:
            cph[i, j] = (xcpd + xcpv * prv[i, j] + xcl * (prc[i, j] + prr[i, j]) + 
                        xci * (pri[i, j] + prs[i, j] + prg[i, j]))
        elif krr == 5:
            cph[i, j] = (xcpd + xcpv * prv[i, j] + xcl * (prc[i, j] + prr[i, j]) + 
                        xci * (pri[i, j] + prs[i, j]))
        elif krr == 3:
            cph[i, j] = xcpd + xcpv * prv[i, j] + xcl * (prc[i, j] + prr[i, j])
        elif krr == 2:
            cph[i, j] = xcpd + xcpv * prv[i, j] + xcl * prc[i, j]
        else:
            cph[i, j] = xcpd + xcpv * prv[i, j] + xcl * prc[i, j]
    
    return cph

@dace.program
def gaussian_condensation(zsbar: dace.float64[dace.symbol('N'), dace.symbol('K')],
                         zsigma: dace.float64[dace.symbol('N'), dace.symbol('K')],
                         zrt: dace.float64[dace.symbol('N'), dace.symbol('K')],
                         pi: dace.float64) -> Tuple[
    dace.float64[dace.symbol('N'), dace.symbol('K')],  # cloud fraction
    dace.float64[dace.symbol('N'), dace.symbol('K')]   # condensate
]:
    """
    Gaussian PDF condensation scheme
    """
    N, K = zsbar.shape
    
    pcldfr = np.zeros((N, K), dtype=np.float64)
    zcond = np.zeros((N, K), dtype=np.float64)
    
    sqrt_2 = dace.math.sqrt(2.0)
    sqrt_2pi = dace.math.sqrt(2.0 * pi)
    
    for i, j in dace.map[0:N, 0:K]:
        # Normalized saturation deficit
        zq1 = zsbar[i, j] / zsigma[i, j]
        
        # Gaussian PDF
        zgcond = -zq1 / sqrt_2
        
        # Compute error function approximation (simplified for DaCe)
        # Using approximation: erf(x) ≈ sign(x) * sqrt(1 - exp(-x^2 * (4/π + ax^2) / (1 + ax^2)))
        # where a = 8(π-3)/(3π(4-π)) ≈ 0.147
        a = 0.147
        x = -zgcond
        if x >= 0:
            erf_val = dace.math.sqrt(1.0 - dace.math.exp(-x*x * (4.0/pi + a*x*x) / (1.0 + a*x*x)))
        else:
            erf_val = -dace.math.sqrt(1.0 - dace.math.exp(-x*x * (4.0/pi + a*x*x) / (1.0 + a*x*x)))
        
        zgauv = 1.0 + erf_val
        
        # Cloud fraction
        pcldfr[i, j] = max(0.0, min(1.0, 0.5 * zgauv))
        
        # Condensate
        exp_term = dace.math.exp(-zgcond * zgcond)
        sqrt_pi = dace.math.sqrt(pi)
        zcond_val = (exp_term - zgcond * sqrt_pi * zgauv) * zsigma[i, j] / sqrt_2pi
        zcond[i, j] = max(zcond_val, 0.0)
        
        # Remove very small values
        if zcond[i, j] < 1e-12 or pcldfr[i, j] == 0.0:
            pcldfr[i, j] = 0.0
            zcond[i, j] = 0.0
    
    return pcldfr, zcond

@dace.program
def cb02_condensation(zsbar: dace.float64[dace.symbol('N'), dace.symbol('K')],
                     zsigma: dace.float64[dace.symbol('N'), dace.symbol('K')]) -> Tuple[
    dace.float64[dace.symbol('N'), dace.symbol('K')],  # cloud fraction
    dace.float64[dace.symbol('N'), dace.symbol('K')],  # condensate
    dace.float64[dace.symbol('N'), dace.symbol('K')]   # psigrc
]:
    """
    CB02 condensation scheme
    """
    N, K = zsbar.shape
    
    pcldfr = np.zeros((N, K), dtype=np.float64)
    zcond = np.zeros((N, K), dtype=np.float64)
    psigrc = np.zeros((N, K), dtype=np.float64)
    
    # Precomputed lookup table for CB02 scheme
    zsrc_1d = np.array([
        0.0, 0.0, 2.0094444e-04, 0.316670e-03,
        4.9965648e-04, 0.785956e-03, 1.2341294e-03, 0.193327e-02,
        3.0190963e-03, 0.470144e-02, 7.2950651e-03, 0.112759e-01,
        1.7350994e-02, 0.265640e-01, 4.0427860e-02, 0.610997e-01,
        9.1578111e-02, 0.135888e+00, 0.1991484, 0.230756e+00,
        0.2850565, 0.375050e+00, 0.5000000, 0.691489e+00,
        0.8413813, 0.933222e+00, 0.9772662, 0.993797e+00,
        0.9986521, 0.999768e+00, 0.9999684, 0.999997e+00,
        1.0000000, 1.000000
    ], dtype=np.float64)
    
    for i, j in dace.map[0:N, 0:K]:
        # Normalized saturation deficit
        zq1 = zsbar[i, j] / zsigma[i, j]
        
        # Total condensate
        if zq1 > 0.0 and zq1 <= 2.0:
            zcond_val = min(dace.math.exp(-1.0) + 0.66*zq1 + 0.086*zq1*zq1, 2.0)
        elif zq1 > 2.0:
            zcond_val = zq1
        else:
            zcond_val = dace.math.exp(1.2*zq1 - 1.0)
        
        zcond[i, j] = zcond_val * zsigma[i, j]
        
        # Cloud fraction
        if zcond[i, j] < 1e-12:
            pcldfr[i, j] = 0.0
        else:
            pcldfr[i, j] = max(0.0, min(1.0, 0.5 + 0.36 * dace.math.atan(1.55 * zq1)))
        
        if pcldfr[i, j] == 0.0:
            zcond[i, j] = 0.0
        
        # Compute psigrc using lookup table
        inq1 = max(-22, min(10, int(dace.math.floor(min(100.0, max(-100.0, 2.0*zq1))))))
        zinc = 2.0*zq1 - float(inq1)
        
        # Linear interpolation in lookup table
        idx = inq1 + 22  # Offset for 0-based indexing
        if idx >= 0 and idx < 33:
            psigrc[i, j] = min(1.0, (1.0 - zinc) * zsrc_1d[idx] + zinc * zsrc_1d[idx + 1])
        else:
            psigrc[i, j] = 0.0
    
    return pcldfr, zcond, psigrc

@dace.program
def condensation_dace(
    temperature: dace.float64[dace.symbol('N'), dace.symbol('K')],
    prv_in: dace.float64[dace.symbol('N'), dace.symbol('K')],
    prc_in: dace.float64[dace.symbol('N'), dace.symbol('K')],
    pri_in: dace.float64[dace.symbol('N'), dace.symbol('K')],
    prr: dace.float64[dace.symbol('N'), dace.symbol('K')],
    prs: dace.float64[dace.symbol('N'), dace.symbol('K')],
    prg: dace.float64[dace.symbol('N'), dace.symbol('K')],
    prh: dace.float64[dace.symbol('N'), dace.symbol('K')],
    ppabs: dace.float64[dace.symbol('N'), dace.symbol('K')],
    psigs: dace.float64[dace.symbol('N'), dace.symbol('K')],
    # Physical constants
    xrd: dace.float64, xrv: dace.float64, xcpd: dace.float64, xcpv: dace.float64,
    xcl: dace.float64, xci: dace.float64, xtt: dace.float64,
    xlvtt: dace.float64, xlstt: dace.float64, xpi: dace.float64,
    alpw: dace.float64, betaw: dace.float64, gamw: dace.float64,
    alpi: dace.float64, betai: dace.float64, gami: dace.float64,
    xtmaxmix: dace.float64, xtminmix: dace.float64,
    # Control parameters
    krr: dace.int32, ice_scheme: dace.int32, cond_scheme: dace.int32,
    use_subgrid: dace.int32
) -> Tuple[
    dace.float64[dace.symbol('N'), dace.symbol('K')],  # prv_out
    dace.float64[dace.symbol('N'), dace.symbol('K')],  # prc_out
    dace.float64[dace.symbol('N'), dace.symbol('K')],  # pri_out
    dace.float64[dace.symbol('N'), dace.symbol('K')],  # pt_out
    dace.float64[dace.symbol('N'), dace.symbol('K')],  # pcldfr
    dace.float64[dace.symbol('N'), dace.symbol('K')]   # psigrc
]:
    """
    Main condensation routine in DaCe
    """
    N, K = temperature.shape
    
    # Initialize output arrays
    prv_out = np.zeros((N, K), dtype=np.float64)
    prc_out = np.zeros((N, K), dtype=np.float64)
    pri_out = np.zeros((N, K), dtype=np.float64)
    pt_out = np.copy(temperature)
    pcldfr = np.zeros((N, K), dtype=np.float64)
    psigrc = np.zeros((N, K), dtype=np.float64)
    
    # Compute saturation pressures
    esatw, esati = compute_saturation_pressures(
        temperature, alpw, betaw, gamw, alpi, betai, gami, ppabs)
    
    # Compute latent heats
    lv, ls = compute_latent_heats(temperature, xlvtt, xcpv, xcl, xlstt, xci, xtt)
    
    # Compute specific heat
    cph = compute_specific_heat(prv_in, prc_in, pri_in, prr, prs, prg, prh,
                               xcpd, xcpv, xcl, xci, krr)
    
    # Compute ice fraction
    frac_ice = compute_ice_fraction(temperature, prc_in, pri_in, ice_scheme,
                                   xtt, xtmaxmix, xtminmix)
    
    # Main computation loop
    for i, j in dace.map[0:N, 0:K]:
        # Total water mixing ratio
        zrt = prv_in[i, j] + prc_in[i, j] + pri_in[i, j]
        
        # Saturation mixing ratios
        zqsl = xrd / xrv * esatw[i, j] / (ppabs[i, j] - esatw[i, j])
        zqsi = xrd / xrv * esati[i, j] / (ppabs[i, j] - esati[i, j])
        
        # Interpolate between liquid and solid saturation
        zqsl_mixed = (1.0 - frac_ice[i, j]) * zqsl + frac_ice[i, j] * zqsi
        zlvs = (1.0 - frac_ice[i, j]) * lv[i, j] + frac_ice[i, j] * ls[i, j]
        
        # Coefficients a and b
        zah = zlvs * zqsl_mixed / (xrv * temperature[i, j]**2) * (xrv * zqsl_mixed / xrd + 1.0)
        za = 1.0 / (1.0 + zlvs / cph[i, j] * zah)
        zb = zah * za
        
        # Normalized saturation deficit
        zsbar_val = za * (zrt - zqsl_mixed + zah * zlvs * (prc_in[i, j] + pri_in[i, j]) / cph[i, j])
        
        # Sigma (simplified - use provided psigs or default)
        zsigma_val = max(1e-10, psigs[i, j] if use_subgrid == 1 else 1e-4)
        
        # Condensation calculation
        if cond_scheme == 0:  # Gaussian
            # Normalized saturation deficit
            zq1 = zsbar_val / zsigma_val
            
            # Gaussian PDF
            zgcond = -zq1 / dace.math.sqrt(2.0)
            
            # Error function approximation
            a = 0.147
            x = -zgcond
            if x >= 0:
                erf_val = dace.math.sqrt(1.0 - dace.math.exp(-x*x * (4.0/xpi + a*x*x) / (1.0 + a*x*x)))
            else:
                erf_val = -dace.math.sqrt(1.0 - dace.math.exp(-x*x * (4.0/xpi + a*x*x) / (1.0 + a*x*x)))
            
            zgauv = 1.0 + erf_val
            
            # Cloud fraction
            pcldfr[i, j] = max(0.0, min(1.0, 0.5 * zgauv))
            
            # Condensate
            exp_term = dace.math.exp(-zgcond * zgcond)
            sqrt_pi = dace.math.sqrt(xpi)
            sqrt_2pi = dace.math.sqrt(2.0 * xpi)
            zcond = (exp_term - zgcond * sqrt_pi * zgauv) * zsigma_val / sqrt_2pi
            zcond = max(zcond, 0.0)
            
            # Remove very small values
            if zcond < 1e-12 or pcldfr[i, j] == 0.0:
                pcldfr[i, j] = 0.0
                zcond = 0.0
            
            psigrc[i, j] = pcldfr[i, j]
            
        else:  # CB02 scheme
            zq1 = zsbar_val / zsigma_val
            
            # Total condensate
            if zq1 > 0.0 and zq1 <= 2.0:
                zcond = min(dace.math.exp(-1.0) + 0.66*zq1 + 0.086*zq1*zq1, 2.0)
            elif zq1 > 2.0:
                zcond = zq1
            else:
                zcond = dace.math.exp(1.2*zq1 - 1.0)
            
            zcond = zcond * zsigma_val
            
            # Cloud fraction
            if zcond < 1e-12:
                pcldfr[i, j] = 0.0
            else:
                pcldfr[i, j] = max(0.0, min(1.0, 0.5 + 0.36 * dace.math.atan(1.55 * zq1)))
            
            if pcldfr[i, j] == 0.0:
                zcond = 0.0
            
            # Simple psigrc for CB02
            psigrc[i, j] = min(1.0, max(0.0, 0.5 + 0.36 * dace.math.atan(1.55 * zq1)))
        
        # Distribute condensate between liquid and ice
        prc_out[i, j] = (1.0 - frac_ice[i, j]) * zcond
        pri_out[i, j] = frac_ice[i, j] * zcond
        
        # Update temperature
        pt_out[i, j] = temperature[i, j] + ((prc_out[i, j] - prc_in[i, j]) * lv[i, j] + 
                                           (pri_out[i, j] - pri_in[i, j]) * ls[i, j]) / cph[i, j]
        
        # Update water vapor
        prv_out[i, j] = zrt - prc_out[i, j] - pri_out[i, j]
    
    return prv_out, prc_out, pri_out, pt_out, pcldfr, psigrc

@dace.program
def ice_adjust_dace(
    # Input state variables
    prv: dace.float64[dace.symbol('N'), dace.symbol('K')],
    prc: dace.float64[dace.symbol('N'), dace.symbol('K')],
    pri: dace.float64[dace.symbol('N'), dace.symbol('K')],
    pth: dace.float64[dace.symbol('N'), dace.symbol('K')],
    prr: dace.float64[dace.symbol('N'), dace.symbol('K')],
    prs: dace.float64[dace.symbol('N'), dace.symbol('K')],
    prg: dace.float64[dace.symbol('N'), dace.symbol('K')],
    prh: dace.float64[dace.symbol('N'), dace.symbol('K')],
    # Atmospheric state
    ppabs: dace.float64[dace.symbol('N'), dace.symbol('K')],
    pexn: dace.float64[dace.symbol('N'), dace.symbol('K')],
    pexnref: dace.float64[dace.symbol('N'), dace.symbol('K')],
    psigs: dace.float64[dace.symbol('N'), dace.symbol('K')],
    # Mass flux parameters
    pcf_mf: dace.float64[dace.symbol('N'), dace.symbol('K')],
    prc_mf: dace.float64[dace.symbol('N'), dace.symbol('K')],
    pri_mf: dace.float64[dace.symbol('N'), dace.symbol('K')],
    pweight_mf_cloud: dace.float64[dace.symbol('N'), dace.symbol('K')],
    # Physical constants
    xrd: dace.float64, xrv: dace.float64, xcpd: dace.float64, xcpv: dace.float64,
    xcl: dace.float64, xci: dace.float64, xtt: dace.float64,
    xlvtt: dace.float64, xlstt: dace.float64, xpi: dace.float64,
    alpw: dace.float64, betaw: dace.float64, gamw: dace.float64,
    alpi: dace.float64, betai: dace.float64, gami: dace.float64,
    xtmaxmix: dace.float64, xtminmix: dace.float64,
    # Control parameters
    ptstep: dace.float64, krr: dace.int32, ice_scheme: dace.int32, 
    cond_scheme: dace.int32, use_subgrid: dace.int32, lmfconv: dace.int32
) -> Tuple[
    dace.float64[dace.symbol('N'), dace.symbol('K')],  # prvs_out
    dace.float64[dace.symbol('N'), dace.symbol('K')],  # prcs_out
    dace.float64[dace.symbol('N'), dace.symbol('K')],  # pris_out
    dace.float64[dace.symbol('N'), dace.symbol('K')],  # pths_out
    dace.float64[dace.symbol('N'), dace.symbol('K')],  # pcldfr
    dace.float64[dace.symbol('N'), dace.symbol('K')]   # psigrc
]:
    """
    Main ice adjustment routine in DaCe
    """
    N, K = prv.shape
    
    # Initialize source arrays
    prvs_out = np.zeros((N, K), dtype=np.float64)
    prcs_out = np.zeros((N, K), dtype=np.float64)
    pris_out = np.zeros((N, K), dtype=np.float64)
    pths_out = np.zeros((N, K), dtype=np.float64)
    
    # Convert potential temperature to temperature
    temperature = np.zeros((N, K), dtype=np.float64)
    for i, j in dace.map[0:N, 0:K]:
        temperature[i, j] = pth[i, j] * pexn[i, j]
    
    # Call condensation routine
    prv_out, prc_out, pri_out, pt_out, pcldfr, psigrc = condensation_dace(
        temperature, prv, prc, pri, prr, prs, prg, prh, ppabs, psigs,
        xrd, xrv, xcpd, xcpv, xcl, xci, xtt, xlvtt, xlstt, xpi,
        alpw, betaw, gamw, alpi, betai, gami, xtmaxmix, xtminmix,
        krr, ice_scheme, cond_scheme, use_subgrid
    )
    
    # Compute latent heats for source computation
    lv, ls = compute_latent_heats(pt_out, xlvtt, xcpv, xcl, xlstt, xci, xtt)
    
    # Compute specific heat for source computation
    cph = compute_specific_heat(prv, prc, pri, prr, prs, prg, prh,
                               xcpd, xcpv, xcl, xci, krr)
    
    # Apply mass flux cloud weighting and compute sources
    for i, j in dace.map[0:N, 0:K]:
        # Apply weighting
        prc_adj = prc_out[i, j] * (1.0 - pweight_mf_cloud[i, j])
        pri_adj = pri_out[i, j] * (1.0 - pweight_mf_cloud[i, j])
        pcldfr[i, j] = pcldfr[i, j] * (1.0 - pweight_mf_cloud[i, j])
        psigrc[i, j] = psigrc[i, j] * (1.0 - pweight_mf_cloud[i, j])
        
        # Compute liquid water source
        zw1 = (prc_adj - prc[i, j]) / ptstep
        if zw1 < 0.0:
            zw1 = max(zw1, -prcs_out[i, j])
        else:
            zw1 = min(zw1, prvs_out[i, j])
        
        prvs_out[i, j] = prvs_out[i, j] - zw1
        prcs_out[i, j] = prcs_out[i, j] + zw1
        pths_out[i, j] = pths_out[i, j] + zw1 * lv[i, j] / (cph[i, j] * pexnref[i, j])
        
        # Compute ice source
        zw2 = (pri_adj - pri[i, j]) / ptstep
        if zw2 < 0.0:
            zw2 = max(zw2, -pris_out[i, j])
        else:
            zw2 = min(zw2, prvs_out[i, j])
        
        prvs_out[i, j] = prvs_out[i, j] - zw2
        pris_out[i, j] = pris_out[i, j] + zw2
        pths_out[i, j] = pths_out[i, j] + zw2 * ls[i, j] / (cph[i, j] * pexnref[i, j])
        
        # Handle mass flux contributions
        if lmfconv == 1:
            # Add mass flux contributions
            zw1_mf = prc_mf[i, j] / ptstep
            zw2_mf = pri_mf[i, j] / ptstep
            
            # Limit to available water vapor
            total_mf = zw1_mf + zw2_mf
            if total_mf > prvs_out[i, j]:
                if total_mf > 0:
                    zw1_mf = zw1_mf * prvs_out[i, j] / total_mf
                    zw2_mf = prvs_out[i, j] - zw1_mf
                else:
                    zw1_mf = 0.0
                    zw2_mf = 0.0
            
            pcldfr[i, j] = min(1.0, pcldfr[i, j] + pcf_mf[i, j])
            prcs_out[i, j] = prcs_out[i, j] + zw1_mf
            pris_out[i, j] = pris_out[i, j] + zw2_mf
            prvs_out[i, j] = prvs_out[i, j] - (zw1_mf + zw2_mf)
            pths_out[i, j] = pths_out[i, j] + (zw1_mf * lv[i, j] + zw2_mf * ls[i, j]) / (cph[i, j] * pexnref[i, j])
        
        # All-or-nothing scheme adjustment
        if use_subgrid == 0:
            if prcs_out[i, j] + pris_out[i, j] > 1e-12 / ptstep:
                pcldfr[i, j] = 1.0
                psigrc[i, j] = 1.0
            else:
                pcldfr[i, j] = 0.0
                psigrc[i, j] = 0.0
    
    return prvs_out, prcs_out, pris_out, pths_out, pcldfr, psigrc

# Wrapper functions for easier integration with existing code
class IceAdjustDaCe:
    """
    DaCe-based ice adjustment class that provides a similar interface to the NumPy version
    """
    
    def __init__(self):
        self.compiled_functions = {}
    
    def ice_adjust(
        self,
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
        DaCe version of ice_adjust with the same interface as the NumPy version
        """
        
        # Handle optional arrays
        if prh is None:
            prh = np.zeros_like(prr)
        if psigs is None:
            psigs = np.full_like(pth, 1e-4)
        
        # Convert scheme names to integers
        ice_scheme = {'T': 0, 'O': 1, 'N': 2, 'S': 3}.get(neb_params.CFRAC_ICE_ADJUST, 0)
        cond_scheme = {'GAUS': 0, 'CB02': 1}.get(neb_params.CCONDENS, 0)
        use_subgrid = 1 if neb_params.LSUBG_COND else 0
        lmfconv_int = 1 if lmfconv else 0
        
        # Call the DaCe function
        prvs_out, prcs_out, pris_out, pths_out, pcldfr, psigrc = ice_adjust_dace(
            prv, prc, pri, pth, prr, prs, prg, prh,
            ppabst, pexn, pexnref, psigs,
            pcf_mf, prc_mf, pri_mf, pweight_mf_cloud,
            # Physical constants
            cst.XRD, cst.XRV, cst.XCPD, cst.XCPV, cst.XCL, cst.XCI, cst.XTT,
            cst.XLVTT, cst.XLSTT, cst.XPI, cst.XALPW, cst.XBETAW, cst.XGAMW,
            cst.XALPI, cst.XBETAI, cst.XGAMI, neb_params.XTMAXMIX, neb_params.XTMINMIX,
            # Control parameters
            ptstep, krr, ice_scheme, cond_scheme, use_subgrid, lmfconv_int
        )
        
        # Initialize additional outputs (simplified for DaCe version)
        picldfr = np.zeros_like(pcldfr)
        pwcldfr = pcldfr.copy()
        pssio = np.zeros_like(pcldfr)
        pssiu = np.zeros_like(pcldfr)
        pifr = np.full_like(pcldfr, 10.0)
        
        # Compute adjusted state variables
        pout_rv = prv + prvs_out * ptstep
        pout_rc = prc + prcs_out * ptstep
        pout_ri = pri + pris_out * ptstep
        pout_th = pth + pths_out * ptstep
        
        return (prvs_out, prcs_out, pris_out, pths_out, pcldfr, 
                picldfr, pwcldfr, pssio, pssiu, pifr,
                psigrc if ocompute_src else None, 
                pout_rv, pout_rc, pout_ri, pout_th)


def create_dace_test_case(nx: int = 10, ny: int = 1, nz: int = 20) -> Dict[str, Any]:
    """
    Create a test case for DaCe ice_adjust using the same data as NumPy version
    """
    return create_test_case(nx, ny, nz)


if __name__ == "__main__":
    print("Ice Adjust DaCe module loaded successfully")
    print("Use IceAdjustDaCe().ice_adjust() to run the DaCe version")
    print("Use create_dace_test_case() to generate test data")
