#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_01_local_counting(
    __global const uint* arr,
    __global uint* global_cnt,
    const uint n,
    const uint iter)
{
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint group_index = get_group_id(0);
    const uint bucket_cnt = (1 << BIT_BATCH);
    const uint bucket_mask = bucket_cnt - 1;
    __local uint cnt[1 << BIT_BATCH];

    if (local_index < bucket_cnt) {
        cnt[local_index] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (index < n) {
        uint value = arr[index];
        uint bucket_index = (value >> iter) & bucket_mask;
        atomic_inc(&cnt[bucket_index]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index < bucket_cnt) {
        const uint per_group = (n + GROUP_SIZE - 1) / GROUP_SIZE;
        const uint out_index = local_index * per_group + group_index;
        global_cnt[out_index] = cnt[local_index];
    }
}
