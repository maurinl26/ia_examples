"""
Test cases for the ice adjustment scheme.
"""

import numpy as np
import pytest
import gt4py.next as gtx
from gt4py.next import Field, float64

from gt4py_ice_adjust.constants import CST
from gt4py_ice_adjust.dimensions import DimPhyex
from gt4py_ice_adjust.ice_adjust import ice_adjust
from gt4py_ice_adjust.utils import compute_ice_fraction_simple


class TestIceAdjust:
    """Test class for ice adjustment functionality."""
    
    @pytest.fixture
    def simple_domain(self):
        """Create a simple test domain."""
        return DimPhyex.create_simple(ni=10, nj=10, nk=20)
    
    @pytest.fixture
    def test_fields(self, simple_domain):
        """Create test fields for a simple atmospheric profile."""
        dim = simple_domain
        shape = dim.get_total_shape()
        
        # Create realistic atmospheric profiles
        # Pressure decreases exponentially with height
        pressure_surface = 101325.0  # Pa
        scale_height = 8000.0  # m
        
        # Height field (simple linear spacing)
        height = np.zeros(shape)
        for k in range(shape[2]):
            height[:, :, k] = k * 500.0  # 500m spacing
        
        # Pressure field
        pressure = pressure_surface * np.exp(-height / scale_height)
        
        # Temperature field (decreases with height)
        temperature_surface = 288.0  # K
        lapse_rate = 0.0065  # K/m
        temperature = temperature_surface - lapse_rate * height
        
        # Exner function
        exner = (pressure / CST.XP00) ** (CST.XRD / CST.XCPD)
        exner_ref = exner.copy()
        
        # Potential temperature
        theta = temperature / exner
        
        # Reference density (simplified)
        rho_ref = pressure / (CST.XRD * temperature)
        rho_dry = rho_ref.copy()
        
        # Water vapor mixing ratio (decreases with height)
        rv = np.maximum(0.001 * np.exp(-height / 2000.0), 1.0e-6)
        
        # Initialize cloud water and ice to small values
        rc = np.full(shape, 1.0e-6)
        ri = np.full(shape, 1.0e-6)
        
        # Rain, snow, graupel (minimal values)
        rr = np.full(shape, 1.0e-8)
        rs = np.full(shape, 1.0e-8)
        rg = np.full(shape, 1.0e-8)
        
        # Initialize source terms to zero
        rv_source = np.zeros(shape)
        rc_source = np.zeros(shape)
        ri_source = np.zeros(shape)
        theta_source = np.zeros(shape)
        
        # Convert to GT4Py fields
        fields = {}
        for name, array in locals().items():
            if isinstance(array, np.ndarray) and array.shape == shape:
                fields[name] = gtx.as_field([gtx.Dims.I, gtx.Dims.J, gtx.Dims.K], array)
        
        return fields
    
    def test_ice_adjust_basic(self, simple_domain, test_fields):
        """Test basic ice adjustment functionality."""
        dim = simple_domain
        
        # Run ice adjustment
        result = ice_adjust(
            dim=dim,
            pressure=test_fields['pressure'],
            height=test_fields['height'],
            rho_dry=test_fields['rho_dry'],
            exner=test_fields['exner'],
            exner_ref=test_fields['exner_ref'],
            rho_ref=test_fields['rho_ref'],
            theta=test_fields['theta'],
            rv=test_fields['rv'],
            rc=test_fields['rc'],
            ri=test_fields['ri'],
            rr=test_fields['rr'],
            rs=test_fields['rs'],
            rg=test_fields['rg'],
            rv_source=test_fields['rv_source'],
            rc_source=test_fields['rc_source'],
            ri_source=test_fields['ri_source'],
            theta_source=test_fields['theta_source'],
            timestep=300.0,  # 5 minutes
            use_subgrid_condensation=False,  # Use all-or-nothing for simplicity
        )
        
        # Unpack results
        (rv_source_out, rc_source_out, ri_source_out, theta_source_out,
         cloud_fraction, ice_cloud_fraction, water_cloud_fraction) = result
        
        # Basic checks
        assert rv_source_out is not None
        assert rc_source_out is not None
        assert ri_source_out is not None
        assert theta_source_out is not None
        assert cloud_fraction is not None
        assert ice_cloud_fraction is not None
        assert water_cloud_fraction is not None
        
        # Check that cloud fractions are between 0 and 1
        cf_array = gtx.as_array(cloud_fraction)
        assert np.all(cf_array >= 0.0)
        assert np.all(cf_array <= 1.0)
        
        # Check conservation (rv_source + rc_source + ri_source should be conserved)
        rv_src_array = gtx.as_array(rv_source_out)
        rc_src_array = gtx.as_array(rc_source_out)
        ri_src_array = gtx.as_array(ri_source_out)
        
        # Total water should be conserved (within numerical precision)
        total_water_source = rv_src_array + rc_src_array + ri_src_array
        assert np.allclose(total_water_source, 0.0, atol=1.0e-12)
    
    def test_ice_adjust_supersaturated_conditions(self, simple_domain):
        """Test ice adjustment under supersaturated conditions."""
        dim = simple_domain
        shape = dim.get_total_shape()
        
        # Create supersaturated conditions
        pressure = np.full(shape, 85000.0)  # 850 hPa
        height = np.full(shape, 1500.0)     # 1.5 km
        temperature = np.full(shape, 268.0)  # -5°C
        
        exner = (pressure / CST.XP00) ** (CST.XRD / CST.XCPD)
        theta = temperature / exner
        rho_ref = pressure / (CST.XRD * temperature)
        
        # High water vapor content (supersaturated)
        rv = np.full(shape, 0.008)  # 8 g/kg
        rc = np.full(shape, 1.0e-6)
        ri = np.full(shape, 1.0e-6)
        rr = np.full(shape, 1.0e-8)
        rs = np.full(shape, 1.0e-8)
        rg = np.full(shape, 1.0e-8)
        
        # Initialize sources
        rv_source = np.zeros(shape)
        rc_source = np.zeros(shape)
        ri_source = np.zeros(shape)
        theta_source = np.zeros(shape)
        
        # Convert to GT4Py fields
        fields = {}
        for name, array in locals().items():
            if isinstance(array, np.ndarray) and array.shape == shape:
                fields[name] = gtx.as_field([gtx.Dims.I, gtx.Dims.J, gtx.Dims.K], array)
        
        # Run ice adjustment
        result = ice_adjust(
            dim=dim,
            pressure=fields['pressure'],
            height=fields['height'],
            rho_dry=fields['rho_ref'],
            exner=fields['exner'],
            exner_ref=fields['exner'],
            rho_ref=fields['rho_ref'],
            theta=fields['theta'],
            rv=fields['rv'],
            rc=fields['rc'],
            ri=fields['ri'],
            rr=fields['rr'],
            rs=fields['rs'],
            rg=fields['rg'],
            rv_source=fields['rv_source'],
            rc_source=fields['rc_source'],
            ri_source=fields['ri_source'],
            theta_source=fields['theta_source'],
            timestep=300.0,
            use_subgrid_condensation=False,
        )
        
        # Unpack results
        (rv_source_out, rc_source_out, ri_source_out, theta_source_out,
         cloud_fraction, ice_cloud_fraction, water_cloud_fraction) = result
        
        # In supersaturated conditions, we should see:
        # 1. Condensation (negative rv_source, positive rc/ri_source)
        # 2. Positive theta_source (latent heating)
        # 3. Non-zero cloud fraction
        
        rv_src_array = gtx.as_array(rv_source_out)
        rc_src_array = gtx.as_array(rc_source_out)
        ri_src_array = gtx.as_array(ri_source_out)
        theta_src_array = gtx.as_array(theta_source_out)
        cf_array = gtx.as_array(cloud_fraction)
        
        # Check for condensation
        total_condensation = rc_src_array + ri_src_array
        assert np.any(total_condensation > 0.0), "Expected condensation in supersaturated conditions"
        
        # Check for latent heating
        assert np.any(theta_src_array > 0.0), "Expected latent heating from condensation"
        
        # Check for cloud formation
        assert np.any(cf_array > 0.0), "Expected cloud formation in supersaturated conditions"
    
    def test_ice_fraction_temperature_dependence(self):
        """Test that ice fraction varies correctly with temperature."""
        # Create temperature field spanning freezing conditions
        temperatures = np.array([280.0, 273.16, 268.0, 258.0, 233.0])  # K
        temp_field = gtx.as_field([gtx.Dims.I], temperatures)
        
        # Compute ice fraction
        ice_frac_field = compute_ice_fraction_simple(temp_field)
        ice_frac = gtx.as_array(ice_frac_field)
        
        # Check expected behavior
        assert ice_frac[0] == 0.0, "Ice fraction should be 0 above freezing"
        assert ice_frac[1] == 0.0, "Ice fraction should be 0 at freezing point"
        assert 0.0 < ice_frac[2] < 1.0, "Ice fraction should be partial at -5°C"
        assert 0.0 < ice_frac[3] < 1.0, "Ice fraction should be partial at -15°C"
        assert ice_frac[4] == 1.0, "Ice fraction should be 1 at -40°C"
        
        # Check monotonic increase with decreasing temperature
        assert np.all(np.diff(ice_frac) >= 0.0), "Ice fraction should increase with decreasing temperature"
    
    def test_conservation_properties(self, simple_domain, test_fields):
        """Test that conservation properties are maintained."""
        dim = simple_domain
        
        # Store initial total water
        rv_init = gtx.as_array(test_fields['rv'])
        rc_init = gtx.as_array(test_fields['rc'])
        ri_init = gtx.as_array(test_fields['ri'])
        total_water_init = rv_init + rc_init + ri_init
        
        # Run ice adjustment
        result = ice_adjust(
            dim=dim,
            pressure=test_fields['pressure'],
            height=test_fields['height'],
            rho_dry=test_fields['rho_dry'],
            exner=test_fields['exner'],
            exner_ref=test_fields['exner_ref'],
            rho_ref=test_fields['rho_ref'],
            theta=test_fields['theta'],
            rv=test_fields['rv'],
            rc=test_fields['rc'],
            ri=test_fields['ri'],
            rr=test_fields['rr'],
            rs=test_fields['rs'],
            rg=test_fields['rg'],
            rv_source=test_fields['rv_source'],
            rc_source=test_fields['rc_source'],
            ri_source=test_fields['ri_source'],
            theta_source=test_fields['theta_source'],
            timestep=300.0,
        )
        
        # Check that total water source is zero (conservation)
        rv_src, rc_src, ri_src = result[:3]
        total_water_source = (gtx.as_array(rv_src) + 
                             gtx.as_array(rc_src) + 
                             gtx.as_array(ri_src))
        
        assert np.allclose(total_water_source, 0.0, atol=1.0e-12), \
            "Total water should be conserved"


if __name__ == "__main__":
    pytest.main([__file__])
