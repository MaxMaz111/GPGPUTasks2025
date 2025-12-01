#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* input_levels,
    __global uint* output_levels,
    const uint n)
{
    const uint index = get_global_id(0);
    const uint lx = index << 1;
    const uint rx = lx + 1;

    if (lx >= n) {
        return;
    }

    const uint sum0 = input_levels[lx];
    const uint sum1 = (rx < n) ? input_levels[rx] : 0;

    output_levels[index] = sum0 + sum1;
}
