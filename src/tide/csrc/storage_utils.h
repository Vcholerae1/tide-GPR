#ifndef STORAGE_UTILS_H
#define STORAGE_UTILS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define STORAGE_DEVICE 0
#define STORAGE_CPU 1
#define STORAGE_DISK 2
#define STORAGE_NONE 3

#ifdef __cplusplus
extern "C" {
#endif

void storage_save_snapshot_cpu(void* store_1, FILE* fp, int64_t storage_mode,
                               int64_t step_idx, size_t step_bytes_uncomp);

void storage_save_snapshot_gpu(void* store_1, void* store_3, FILE* fp,
                               int64_t storage_mode, int64_t step_idx,
                               size_t shot_bytes_uncomp, size_t n_shots);

void storage_load_snapshot_cpu(void* store_1, FILE* fp, int64_t storage_mode,
                               int64_t step_idx, size_t step_bytes_uncomp);

void storage_load_snapshot_gpu(void* store_1, void* store_3, FILE* fp,
                               int64_t storage_mode, int64_t step_idx,
                               size_t shot_bytes_uncomp, size_t n_shots);

int tide_cuda_fp8_enabled(void);

const char* tide_cuda_arches(void);

#ifdef __cplusplus
}
#endif

#endif
