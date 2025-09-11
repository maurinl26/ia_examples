! Fortran-C Array Interface Module
! This module provides utilities to pass Fortran arrays to C restricted pointers
! Created for dwarf-p-ice3-dace project

module fortran_c_array_interface
    use, intrinsic :: iso_c_binding
    implicit none

    ! Generic interface for array conversion
    interface fortran_to_c_ptr
        module procedure :: real64_1d_to_c_ptr
        module procedure :: real64_2d_to_c_ptr
        module procedure :: real64_3d_to_c_ptr
        module procedure :: real32_1d_to_c_ptr
        module procedure :: real32_2d_to_c_ptr
        module procedure :: real32_3d_to_c_ptr
        module procedure :: int32_1d_to_c_ptr
        module procedure :: int32_2d_to_c_ptr
        module procedure :: int32_3d_to_c_ptr
    end interface fortran_to_c_ptr

    ! Interface for C functions expecting restricted pointers
    interface
        ! Example C function signature with restricted pointers
        subroutine c_function_with_restricted_arrays(handle, &
            array1, array2, array3, &
            n1, n2, n3, &
            scalar_param) bind(c, name='c_function_with_restricted_arrays')
            use, intrinsic :: iso_c_binding
            type(c_ptr), value :: handle
            type(c_ptr), value :: array1  ! double * __restrict__
            type(c_ptr), value :: array2  ! double * __restrict__
            type(c_ptr), value :: array3  ! double * __restrict__
            integer(c_int), value :: n1
            integer(c_int), value :: n2
            integer(c_int), value :: n3
            real(c_double), value :: scalar_param
        end subroutine c_function_with_restricted_arrays
    end interface

contains

    ! Convert 1D real64 array to C pointer
    function real64_1d_to_c_ptr(fortran_array) result(c_ptr_result)
        real(c_double), target, intent(in) :: fortran_array(:)
        type(c_ptr) :: c_ptr_result
        
        if (size(fortran_array) > 0) then
            c_ptr_result = c_loc(fortran_array(1))
        else
            c_ptr_result = c_null_ptr
        endif
    end function real64_1d_to_c_ptr

    ! Convert 2D real64 array to C pointer
    function real64_2d_to_c_ptr(fortran_array) result(c_ptr_result)
        real(c_double), target, intent(in) :: fortran_array(:,:)
        type(c_ptr) :: c_ptr_result
        
        if (size(fortran_array) > 0) then
            c_ptr_result = c_loc(fortran_array(1,1))
        else
            c_ptr_result = c_null_ptr
        endif
    end function real64_2d_to_c_ptr

    ! Convert 3D real64 array to C pointer
    function real64_3d_to_c_ptr(fortran_array) result(c_ptr_result)
        real(c_double), target, intent(in) :: fortran_array(:,:,:)
        type(c_ptr) :: c_ptr_result
        
        if (size(fortran_array) > 0) then
            c_ptr_result = c_loc(fortran_array(1,1,1))
        else
            c_ptr_result = c_null_ptr
        endif
    end function real64_3d_to_c_ptr

    ! Convert 1D real32 array to C pointer
    function real32_1d_to_c_ptr(fortran_array) result(c_ptr_result)
        real(c_float), target, intent(in) :: fortran_array(:)
        type(c_ptr) :: c_ptr_result
        
        if (size(fortran_array) > 0) then
            c_ptr_result = c_loc(fortran_array(1))
        else
            c_ptr_result = c_null_ptr
        endif
    end function real32_1d_to_c_ptr

    ! Convert 2D real32 array to C pointer
    function real32_2d_to_c_ptr(fortran_array) result(c_ptr_result)
        real(c_float), target, intent(in) :: fortran_array(:,:)
        type(c_ptr) :: c_ptr_result
        
        if (size(fortran_array) > 0) then
            c_ptr_result = c_loc(fortran_array(1,1))
        else
            c_ptr_result = c_null_ptr
        endif
    end function real32_2d_to_c_ptr

    ! Convert 3D real32 array to C pointer
    function real32_3d_to_c_ptr(fortran_array) result(c_ptr_result)
        real(c_float), target, intent(in) :: fortran_array(:,:,:)
        type(c_ptr) :: c_ptr_result
        
        if (size(fortran_array) > 0) then
            c_ptr_result = c_loc(fortran_array(1,1,1))
        else
            c_ptr_result = c_null_ptr
        endif
    end function real32_3d_to_c_ptr

    ! Convert 1D int32 array to C pointer
    function int32_1d_to_c_ptr(fortran_array) result(c_ptr_result)
        integer(c_int), target, intent(in) :: fortran_array(:)
        type(c_ptr) :: c_ptr_result
        
        if (size(fortran_array) > 0) then
            c_ptr_result = c_loc(fortran_array(1))
        else
            c_ptr_result = c_null_ptr
        endif
    end function int32_1d_to_c_ptr

    ! Convert 2D int32 array to C pointer
    function int32_2d_to_c_ptr(fortran_array) result(c_ptr_result)
        integer(c_int), target, intent(in) :: fortran_array(:,:)
        type(c_ptr) :: c_ptr_result
        
        if (size(fortran_array) > 0) then
            c_ptr_result = c_loc(fortran_array(1,1))
        else
            c_ptr_result = c_null_ptr
        endif
    end function int32_2d_to_c_ptr

    ! Convert 3D int32 array to C pointer
    function int32_3d_to_c_ptr(fortran_array) result(c_ptr_result)
        integer(c_int), target, intent(in) :: fortran_array(:,:,:)
        type(c_ptr) :: c_ptr_result
        
        if (size(fortran_array) > 0) then
            c_ptr_result = c_loc(fortran_array(1,1,1))
        else
            c_ptr_result = c_null_ptr
        endif
    end function int32_3d_to_c_ptr

    ! Utility subroutine to call C function with Fortran arrays
    subroutine call_c_with_fortran_arrays(handle, &
        f_array1, f_array2, f_array3, &
        scalar_param)
        
        type(c_ptr), intent(in) :: handle
        real(c_double), target, intent(inout) :: f_array1(:,:)
        real(c_double), target, intent(inout) :: f_array2(:,:)
        real(c_double), target, intent(inout) :: f_array3(:,:)
        real(c_double), intent(in) :: scalar_param
        
        ! Local variables
        type(c_ptr) :: c_ptr1, c_ptr2, c_ptr3
        integer(c_int) :: n1, n2, n3
        
        ! Convert Fortran arrays to C pointers
        c_ptr1 = fortran_to_c_ptr(f_array1)
        c_ptr2 = fortran_to_c_ptr(f_array2)
        c_ptr3 = fortran_to_c_ptr(f_array3)
        
        ! Get array dimensions
        n1 = int(size(f_array1, 1), c_int)
        n2 = int(size(f_array2, 1), c_int)
        n3 = int(size(f_array3, 1), c_int)
        
        ! Call C function
        call c_function_with_restricted_arrays(handle, &
            c_ptr1, c_ptr2, c_ptr3, &
            n1, n2, n3, &
            scalar_param)
    end subroutine call_c_with_fortran_arrays

end module fortran_c_array_interface
