# Fortran-C Array Interface for Restricted Pointers

This interface provides utilities to pass Fortran arrays to C functions that expect restricted pointers. It's designed for high-performance computing applications where C functions need direct access to Fortran array data.

## Overview

The interface consists of three main components:

1. **`fortran_c_array_interface.f90`** - Fortran module with generic conversion functions
2. **`c_restricted_interface.h`** - C header with function declarations and utilities
3. **`example_usage.f90`** - Comprehensive usage examples

## Key Features

- **Generic Interface**: Supports multiple data types (real64, real32, int32) and dimensions (1D, 2D, 3D)
- **Type Safety**: Uses ISO C binding for guaranteed interoperability
- **Memory Safety**: Includes null pointer checks and proper error handling
- **Performance Optimized**: Direct pointer access without data copying
- **DACE Compatible**: Designed to work with existing DACE-generated code

## Usage Patterns

### Basic Array Conversion

```fortran
use fortran_c_array_interface
use, intrinsic :: iso_c_binding

real(c_double), allocatable, target :: fortran_array(:,:)
type(c_ptr) :: c_pointer

! Allocate and initialize array
allocate(fortran_array(100, 50))
fortran_array = 1.0_c_double

! Convert to C pointer
c_pointer = fortran_to_c_ptr(fortran_array)

! Use c_pointer in C function calls
```

### ICE3-Style Integration

```fortran
! Following the existing ICE3 DACE pattern
type(c_ptr) :: handle
real(c_double), allocatable, target :: cldfr(:,:), exn(:,:), pabs(:,:)

! Initialize handle
handle = c_dace_init_ice_adjust(IJ, K)

! Convert arrays and call C function
call c_program_ice_adjust(handle, &
    fortran_to_c_ptr(cldfr), &
    fortran_to_c_ptr(exn), &
    fortran_to_c_ptr(pabs), &
    ! ... other parameters
)
```

## Supported Data Types

| Fortran Type | C Type | Interface Function |
|--------------|--------|-------------------|
| `real(c_double)` | `double*` | `fortran_to_c_ptr` |
| `real(c_float)` | `float*` | `fortran_to_c_ptr` |
| `integer(c_int)` | `int*` | `fortran_to_c_ptr` |

## Array Dimensions

The interface supports 1D, 2D, and 3D arrays:

```fortran
! 1D array
real(c_double), target :: array_1d(1000)
c_ptr_1d = fortran_to_c_ptr(array_1d)

! 2D array  
real(c_double), target :: array_2d(100, 50)
c_ptr_2d = fortran_to_c_ptr(array_2d)

! 3D array
real(c_double), target :: array_3d(100, 50, 25)
c_ptr_3d = fortran_to_c_ptr(array_3d)
```

## C Function Declarations

### Basic Pattern

```c
void c_function_with_restricted_arrays(
    void* handle,
    double* __restrict__ array1,
    double* __restrict__ array2,
    double* __restrict__ array3,
    int n1, int n2, int n3,
    double scalar_param
);
```

### ICE3-Style Pattern

```c
void ice3_compute_with_restricted_arrays(
    void* handle,
    double* __restrict__ temperature,
    double* __restrict__ pressure,
    double* __restrict__ humidity,
    int ni, int nk,
    double dt,
    bool enable_condensation
);
```

## Memory Layout Considerations

### Fortran vs C Array Ordering

- **Fortran**: Column-major ordering (first index varies fastest)
- **C**: Row-major ordering (last index varies fastest)

For multi-dimensional arrays, use the provided indexing macros:

```c
// 2D array access: fortran_array(i,j) -> c_array[FORTRAN_2D_INDEX(i,j,dim1)]
#define FORTRAN_2D_INDEX(i, j, dim1) ((i) + (j) * (dim1))

// 3D array access: fortran_array(i,j,k) -> c_array[FORTRAN_3D_INDEX(i,j,k,dim1,dim2)]
#define FORTRAN_3D_INDEX(i, j, k, dim1, dim2) ((i) + (j) * (dim1) + (k) * (dim1) * (dim2))
```

### Memory Alignment

For optimal performance, consider memory alignment:

```c
// Assume 32-byte alignment for vectorization
double* __restrict__ aligned_array = ASSUME_ALIGNED(input_array, 32);
```

## Error Handling

### Fortran Side

```fortran
type(c_ptr) :: c_pointer
c_pointer = fortran_to_c_ptr(fortran_array)

if (.not. c_associated(c_pointer)) then
    print *, "Error: Failed to convert array to C pointer"
    stop 1
endif
```

### C Side

```c
if (array_ptr == NULL) {
    fprintf(stderr, "Error: Received null pointer from Fortran\n");
    return -1;
}
```

## Performance Considerations

### Best Practices

1. **Use `target` attribute**: Always declare Fortran arrays with `target` attribute
2. **Minimize conversions**: Convert arrays once and reuse pointers
3. **Check alignment**: Use alignment hints for vectorization
4. **Avoid copying**: The interface provides direct pointer access without copying

### Memory Management

```fortran
! Proper memory management
real(c_double), allocatable, target :: array(:,:)
type(c_ptr) :: c_ptr

allocate(array(ni, nk))
c_ptr = fortran_to_c_ptr(array)

! Use c_ptr in C function calls...

! Clean up
deallocate(array)
! c_ptr becomes invalid after deallocation
```

## Integration with Existing Code

### Modifying Existing ICE3 Calls

**Before:**
```fortran
call c_program_ice_adjust(handle, cldfr, exn, pabs, ...)
```

**After:**
```fortran
call c_program_ice_adjust(handle, &
    fortran_to_c_ptr(cldfr), &
    fortran_to_c_ptr(exn), &
    fortran_to_c_ptr(pabs), &
    ...)
```

### Adding New C Functions

1. Declare function in C header with `__restrict__` pointers
2. Add Fortran interface declaration with `type(c_ptr), value` parameters
3. Use `fortran_to_c_ptr` for array conversion

## Testing

Compile and run the example:

```bash
# Compile the interface and example
gfortran -c interface/fortran_c_array_interface.f90
gfortran -c interface/example_usage.f90
gfortran -o example fortran_c_array_interface.o example_usage.o

# Run the example
./example
```

Expected output:
```
=== Example 1: Basic Usage ===
Successfully converted Fortran arrays to C pointers
Temperature field size: 5000
Pressure field size: 5000
Output field size: 5000

=== Example 2: ICE3-Style Usage ===
ICE3-style arrays converted successfully
IJ = 225, K = 90
Cloud fraction mean: 0.500000000000000
Exner function mean: 1.000000000000000
Pressure mean: 101325.000000000

=== Example 3: Mixed Data Types ===
Double precision array converted successfully
Single precision array converted successfully
Integer array converted successfully
Array sizes - Double: 1000, Float: 1000, Integer: 1000
```

## Troubleshooting

### Common Issues

1. **Segmentation Fault**: Check that arrays have `target` attribute
2. **Wrong Results**: Verify array indexing (Fortran column-major vs C row-major)
3. **Compilation Errors**: Ensure `iso_c_binding` is used consistently
4. **Memory Leaks**: Don't deallocate Fortran arrays while C pointers are in use

### Debug Tips

```fortran
! Check pointer validity
if (c_associated(c_ptr)) then
    print *, "Pointer is valid"
else
    print *, "Pointer is null - check array allocation and target attribute"
endif

! Check array bounds
print *, "Array shape:", shape(fortran_array)
print *, "Array size:", size(fortran_array)
```

## Compiler Compatibility

Tested with:
- GNU Fortran (gfortran) 9.0+
- Intel Fortran Compiler (ifort) 19.0+
- NVIDIA HPC SDK (nvfortran) 21.0+

## License

This interface is part of the dwarf-p-ice3-dace project and follows the same licensing terms.
