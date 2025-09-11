/*
 * C Header for Fortran-C Array Interface
 * This header demonstrates how to declare C functions with restricted pointers
 * that can receive Fortran arrays through the interface
 * Created for dwarf-p-ice3-dace project
 */

#ifndef C_RESTRICTED_INTERFACE_H
#define C_RESTRICTED_INTERFACE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Example function signature showing how to declare C functions
 * that accept restricted pointers from Fortran arrays
 */
void c_function_with_restricted_arrays(
    void* handle,                    // Handle from initialization
    double* __restrict__ array1,     // Fortran real(c_double) array
    double* __restrict__ array2,     // Fortran real(c_double) array  
    double* __restrict__ array3,     // Fortran real(c_double) array
    int n1,                         // Array dimension 1
    int n2,                         // Array dimension 2
    int n3,                         // Array dimension 3
    double scalar_param             // Scalar parameter
);

/*
 * More specific example based on the ICE3 pattern
 * This matches the pattern used in the existing DACE-generated code
 */
void ice3_compute_with_restricted_arrays(
    void* handle,
    double* __restrict__ temperature,    // Temperature field
    double* __restrict__ pressure,       // Pressure field
    double* __restrict__ humidity,       // Humidity field
    double* __restrict__ cloud_fraction, // Cloud fraction field
    double* __restrict__ ice_content,    // Ice content field
    double* __restrict__ water_content,  // Water content field
    int ni,                             // Number of horizontal points
    int nk,                             // Number of vertical levels
    double dt,                          // Time step
    bool enable_condensation,           // Condensation flag
    bool enable_ice_processes           // Ice processes flag
);

/*
 * Generic template for different data types
 */

// Single precision arrays
void process_float_arrays(
    void* handle,
    float* __restrict__ input_array,
    float* __restrict__ output_array,
    int size,
    float scaling_factor
);

// Integer arrays
void process_int_arrays(
    void* handle,
    int* __restrict__ input_array,
    int* __restrict__ output_array,
    int size,
    int offset
);

// Mixed precision example
void mixed_precision_compute(
    void* handle,
    double* __restrict__ high_precision_data,
    float* __restrict__ low_precision_data,
    int* __restrict__ index_array,
    int n_high,
    int n_low,
    int n_indices
);

/*
 * Multi-dimensional array handling
 * Note: C sees multi-dimensional Fortran arrays as 1D with column-major ordering
 */
void process_2d_array(
    void* handle,
    double* __restrict__ array_2d,  // Fortran array(:,:) -> C 1D array
    int dim1,                       // First dimension size
    int dim2,                       // Second dimension size
    double multiplier
);

void process_3d_array(
    void* handle,
    double* __restrict__ array_3d,  // Fortran array(:,:,:) -> C 1D array
    int dim1,                       // First dimension size
    int dim2,                       // Second dimension size  
    int dim3,                       // Third dimension size
    double scaling
);

/*
 * Utility macros for array indexing
 * These help convert from multi-dimensional indices to 1D indices
 * for Fortran column-major arrays
 */

// 2D array indexing: array(i,j) in Fortran -> array[FORTRAN_2D_INDEX(i,j,dim1)] in C
#define FORTRAN_2D_INDEX(i, j, dim1) ((i) + (j) * (dim1))

// 3D array indexing: array(i,j,k) in Fortran -> array[FORTRAN_3D_INDEX(i,j,k,dim1,dim2)] in C
#define FORTRAN_3D_INDEX(i, j, k, dim1, dim2) ((i) + (j) * (dim1) + (k) * (dim1) * (dim2))

/*
 * Memory alignment and optimization hints
 */
#ifdef __GNUC__
    #define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#else
    #define ASSUME_ALIGNED(ptr, alignment) (ptr)
#endif

// Example function using alignment assumptions
void optimized_compute(
    void* handle,
    double* __restrict__ aligned_array,  // Assume 32-byte alignment
    int size
);

#ifdef __cplusplus
}
#endif

#endif /* C_RESTRICTED_INTERFACE_H */
