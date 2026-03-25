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

#define STORAGE_FORMAT_FULL 0
#define STORAGE_FORMAT_BF16 1

#ifdef __cplusplus
extern "C" {
#endif

void storage_save_snapshot_cpu(void* store_1, FILE* fp, int64_t storage_mode,
                               int64_t step_idx, size_t step_bytes_uncomp);

void storage_load_snapshot_cpu(void* store_1, FILE* fp, int64_t storage_mode,
                               int64_t step_idx, size_t step_bytes_uncomp);

void storage_copy_snapshot_d2h(void* device_store, void* host_store,
                               size_t shot_bytes_uncomp, size_t n_shots,
                               void* stream);

void storage_copy_snapshot_h2d(void* device_store, void* host_store,
                               size_t shot_bytes_uncomp, size_t n_shots,
                               void* stream);

void* storage_async_disk_open(const char* path, bool write_mode,
                              int64_t slot_count);

void storage_async_disk_close(void* handle);

void storage_async_disk_wait_slot(void* handle, int64_t slot_idx);

void storage_async_disk_enqueue_write(void* handle, int64_t slot_idx,
                                      void* host_ptr, size_t nbytes,
                                      int64_t file_offset, void* ready_event);

void storage_async_disk_enqueue_read(void* handle, int64_t slot_idx,
                                     void* host_ptr, size_t nbytes,
                                     int64_t file_offset, void* ready_event);

const char* tide_cuda_arches(void);

#ifdef __cplusplus
}
#endif

#endif
