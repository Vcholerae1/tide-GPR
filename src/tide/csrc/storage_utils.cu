#include <cuda_runtime.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "storage_utils.h"

#ifndef TIDE_CUDA_FP8_ENABLED
#define TIDE_CUDA_FP8_ENABLED 0
#endif
#ifndef TIDE_CUDA_ARCHES_STR
#define TIDE_CUDA_ARCHES_STR "unknown"
#endif

static bool read_exact(FILE* fp, void* dst, size_t nbytes) {
  size_t total = 0;
  while (total < nbytes) {
    size_t n = fread((char*)dst + total, 1, nbytes - total, fp);
    if (n == 0) {
      return false;
    }
    total += n;
  }
  return true;
}

static bool write_exact(FILE* fp, const void* src, size_t nbytes) {
  size_t total = 0;
  while (total < nbytes) {
    size_t n = fwrite((const char*)src + total, 1, nbytes - total, fp);
    if (n == 0) {
      return false;
    }
    total += n;
  }
  return true;
}

static void report_io_error(const char* op, int64_t step_idx) {
  if (errno != 0) {
    fprintf(stderr, "storage_utils: %s failed at step %lld: %s\n",
            op, (long long)step_idx, strerror(errno));
  } else {
    fprintf(stderr, "storage_utils: %s failed at step %lld\n",
            op, (long long)step_idx);
  }
}

static void report_cuda_error(const char* op, cudaError_t err) {
  fprintf(stderr, "storage_utils: %s failed: %s\n",
          op, cudaGetErrorString(err));
}

extern "C" {

int tide_cuda_fp8_enabled(void) {
  return TIDE_CUDA_FP8_ENABLED ? 1 : 0;
}

const char* tide_cuda_arches(void) {
  return TIDE_CUDA_ARCHES_STR;
}

void storage_save_snapshot_gpu(
    void* store_1, void* store_3, FILE* fp, int64_t storage_mode,
    int64_t step_idx, size_t shot_bytes_uncomp, size_t n_shots) {
  if (storage_mode == STORAGE_NONE) return;
  size_t bytes_to_store = shot_bytes_uncomp * n_shots;

  if (storage_mode == STORAGE_CPU || storage_mode == STORAGE_DISK) {
    if (storage_mode == STORAGE_DISK) {
      // Disk mode needs host-visible data immediately for fwrite.
      cudaError_t err =
          cudaMemcpy(store_3, store_1, bytes_to_store, cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        report_cuda_error("cudaMemcpy(D2H)", err);
        return;
      }
    } else {
      // CPU mode: avoid blocking the host thread on every step.
      // Copies are enqueued on the current (default) stream and will be ordered
      // with subsequent CUDA work in the same stream.
      cudaError_t err = cudaMemcpyAsync(
          store_3, store_1, bytes_to_store, cudaMemcpyDeviceToHost, 0);
      if (err != cudaSuccess) {
        report_cuda_error("cudaMemcpyAsync(D2H)", err);
        return;
      }
    }
  }
  if (storage_mode == STORAGE_DISK) {
    int64_t offset = step_idx * (int64_t)bytes_to_store;
    errno = 0;
    if (fseek(fp, offset, SEEK_SET) != 0) {
      report_io_error("fseek(write)", step_idx);
      return;
    }
    if (!write_exact(fp, store_3, bytes_to_store)) {
      report_io_error("fwrite", step_idx);
    }
  }
}

void storage_load_snapshot_gpu(void* store_1, void* store_3, FILE* fp,
                               int64_t storage_mode, int64_t step_idx,
                               size_t shot_bytes_uncomp, size_t n_shots) {
  if (storage_mode == STORAGE_NONE) return;
  size_t bytes_to_load = shot_bytes_uncomp * n_shots;

  if (storage_mode == STORAGE_DISK) {
    int64_t offset = step_idx * (int64_t)bytes_to_load;
    errno = 0;
    if (fseek(fp, offset, SEEK_SET) != 0) {
      report_io_error("fseek(read)", step_idx);
      memset(store_3, 0, bytes_to_load);
    } else if (!read_exact(fp, store_3, bytes_to_load)) {
      if (feof(fp)) {
        fprintf(stderr, "storage_utils: unexpected EOF at step %lld\n",
                (long long)step_idx);
      } else {
        report_io_error("fread", step_idx);
      }
      clearerr(fp);
      memset(store_3, 0, bytes_to_load);
    }
  }

  if (storage_mode == STORAGE_CPU || storage_mode == STORAGE_DISK) {
    if (storage_mode == STORAGE_DISK) {
      // Disk mode reuses the same pinned host buffer (store_3) for many steps.
      // Use a synchronous copy to avoid the host overwriting store_3 (fread)
      // before the device copy has consumed it.
      cudaError_t err =
          cudaMemcpy(store_1, store_3, bytes_to_load, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        report_cuda_error("cudaMemcpy(H2D)", err);
        return;
      }
    } else {
      cudaError_t err = cudaMemcpyAsync(
          store_1, store_3, bytes_to_load, cudaMemcpyHostToDevice, 0);
      if (err != cudaSuccess) {
        report_cuda_error("cudaMemcpyAsync(H2D)", err);
        return;
      }
    }
  }
}
}
