#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_04_scatter(
    __global const uint* array,
    __global const uint* prefix_sum,
    __global uint* result,
    const uint n,
    const uint iter)
{
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint group_index = get_group_id(0);

    const uint num_buckets = (1u << BIT_BATCH);
    const uint bucket_mask = num_buckets - 1u;

    if (index >= n) {
        return;
    }

    uint value = array[index];

    const uint bucket = (value >> iter) & bucket_mask;

    const uint items_per_group = (n + GROUP_SIZE - 1u) / GROUP_SIZE;
    const uint prefix_idx = bucket * items_per_group + group_index;

    const uint prev_sum = (prefix_idx == 0u) ? 0u : prefix_sum[prefix_idx - 1u];

    __local uint local_prefix[GROUP_SIZE][1 << BIT_BATCH];

    for (uint b = 0; b < num_buckets; ++b) {
        local_prefix[local_index][b] = (b == bucket);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index < num_buckets) {
        for (uint i = 1; i < GROUP_SIZE; ++i) {
            local_prefix[i][local_index] += local_prefix[i - 1][local_index];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint local_pos = local_prefix[local_index][bucket];
    const uint out_index = prev_sum + local_pos - 1u;

    result[out_index] = value;
}
