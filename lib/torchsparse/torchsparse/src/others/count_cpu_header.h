#ifndef _SPARSE_COUNT_CPU
#define _SPARSE_COUNT_CPU
#include <torch/torch.h>

at::Tensor cpu_count_forward(
    const at::Tensor idx,
    const int s
);

#endif