! Example Usage of Fortran-C Array Interface
! This demonstrates how to use the interface to pass Fortran arrays to C restricted pointers
! Based on the existing ICE3 DACE integration

program example_usage
    use fortran_c_array_interface
    use, intrinsic :: iso_c_binding
    implicit none

    ! Example 1: Basic usage with the generic interface
    call example_basic_usage()
    
    ! Example 2: ICE3-style usage matching the existing pattern
    call example_ice3_style_usage()
    
    ! Example 3: Mixed data types
    call example_mixed_types()

contains

    subroutine example_basic_usage()
        implicit none
        
        ! Fortran arrays
        real(c_double), allocatable, target :: temp_field(:,:)
        real(c_double), allocatable, target :: pressure_field(:,:)
        real(c_double), allocatable, target :: output_field(:,:)
        
        ! Dimensions
        integer, parameter :: ni = 100, nk = 50
        integer :: i, j
        
        ! C pointers
        type(c_ptr) :: temp_ptr, pressure_ptr, output_ptr
        
        print *, "=== Example 1: Basic Usage ==="
        
        ! Allocate arrays
        allocate(temp_field(ni, nk))
        allocate(pressure_field(ni, nk))
        allocate(output_field(ni, nk))
        
        ! Initialize with dummy data
        do j = 1, nk
            do i = 1, ni
                temp_field(i, j) = 273.15_c_double + real(i + j, c_double)
                pressure_field(i, j) = 101325.0_c_double + real(i * j, c_double)
                output_field(i, j) = 0.0_c_double
            end do
        end do
        
        ! Convert Fortran arrays to C pointers using the generic interface
        temp_ptr = fortran_to_c_ptr(temp_field)
        pressure_ptr = fortran_to_c_ptr(pressure_field)
        output_ptr = fortran_to_c_ptr(output_field)
        
        ! Check if pointers are valid
        if (c_associated(temp_ptr) .and. c_associated(pressure_ptr) .and. c_associated(output_ptr)) then
            print *, "Successfully converted Fortran arrays to C pointers"
            print *, "Temperature field size:", size(temp_field)
            print *, "Pressure field size:", size(pressure_field)
            print *, "Output field size:", size(output_field)
        else
            print *, "Error: Failed to convert arrays to C pointers"
        endif
        
        ! Clean up
        deallocate(temp_field, pressure_field, output_field)
        
    end subroutine example_basic_usage

    subroutine example_ice3_style_usage()
        implicit none
        
        ! Variables matching the ICE3 pattern
        integer(c_int) :: IJ, K
        real(c_double), allocatable, target :: cldfr(:,:)
        real(c_double), allocatable, target :: exn(:,:)
        real(c_double), allocatable, target :: pabs(:,:)
        real(c_double), allocatable, target :: rc0(:,:)
        real(c_double), allocatable, target :: th0(:,:)
        
        ! C pointers
        type(c_ptr) :: cldfr_ptr, exn_ptr, pabs_ptr, rc0_ptr, th0_ptr
        
        ! Handle (would come from actual initialization)
        type(c_ptr) :: handle
        
        print *, "=== Example 2: ICE3-Style Usage ==="
        
        ! Set dimensions
        IJ = 225  ! 15 x 15
        K = 90
        
        ! Allocate arrays
        allocate(cldfr(IJ, K))
        allocate(exn(IJ, K))
        allocate(pabs(IJ, K))
        allocate(rc0(IJ, K))
        allocate(th0(IJ, K))
        
        ! Initialize with dummy values
        cldfr(:,:) = 0.5_c_double
        exn(:,:) = 1.0_c_double
        pabs(:,:) = 101325.0_c_double
        rc0(:,:) = 0.001_c_double
        th0(:,:) = 300.0_c_double
        
        ! Convert arrays to C pointers
        cldfr_ptr = fortran_to_c_ptr(cldfr)
        exn_ptr = fortran_to_c_ptr(exn)
        pabs_ptr = fortran_to_c_ptr(pabs)
        rc0_ptr = fortran_to_c_ptr(rc0)
        th0_ptr = fortran_to_c_ptr(th0)
        
        ! Simulate handle (in real usage, this would come from c_dace_init_ice_adjust)
        handle = c_null_ptr
        
        print *, "ICE3-style arrays converted successfully"
        print *, "IJ =", IJ, ", K =", K
        print *, "Cloud fraction mean:", sum(cldfr) / (IJ * K)
        print *, "Exner function mean:", sum(exn) / (IJ * K)
        print *, "Pressure mean:", sum(pabs) / (IJ * K)
        
        ! In real usage, you would call the C function here:
        ! call c_program_ice_adjust(handle, cldfr_ptr, exn_ptr, pabs_ptr, ...)
        
        ! Clean up
        deallocate(cldfr, exn, pabs, rc0, th0)
        
    end subroutine example_ice3_style_usage

    subroutine example_mixed_types()
        implicit none
        
        ! Different data types
        real(c_double), allocatable, target :: double_array(:)
        real(c_float), allocatable, target :: float_array(:)
        integer(c_int), allocatable, target :: int_array(:)
        
        ! C pointers
        type(c_ptr) :: double_ptr, float_ptr, int_ptr
        
        integer, parameter :: n = 1000
        integer :: i
        
        print *, "=== Example 3: Mixed Data Types ==="
        
        ! Allocate arrays
        allocate(double_array(n))
        allocate(float_array(n))
        allocate(int_array(n))
        
        ! Initialize arrays
        do i = 1, n
            double_array(i) = real(i, c_double) * 0.1_c_double
            float_array(i) = real(i, c_float) * 0.01_c_float
            int_array(i) = i
        end do
        
        ! Convert to C pointers using the generic interface
        double_ptr = fortran_to_c_ptr(double_array)
        float_ptr = fortran_to_c_ptr(float_array)
        int_ptr = fortran_to_c_ptr(int_array)
        
        ! Verify conversions
        if (c_associated(double_ptr)) then
            print *, "Double precision array converted successfully"
        endif
        
        if (c_associated(float_ptr)) then
            print *, "Single precision array converted successfully"
        endif
        
        if (c_associated(int_ptr)) then
            print *, "Integer array converted successfully"
        endif
        
        print *, "Array sizes - Double:", size(double_array), &
                 ", Float:", size(float_array), &
                 ", Integer:", size(int_array)
        
        ! Clean up
        deallocate(double_array, float_array, int_array)
        
    end subroutine example_mixed_types

end program example_usage
