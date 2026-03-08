#include "storage_utils.h"

#include <errno.h>
#include <string.h>

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

void storage_save_snapshot_cpu(void* store_1, FILE* fp, int64_t storage_mode,
                               int64_t step_idx, size_t step_bytes_uncomp) {
  if (storage_mode == STORAGE_NONE) return;
  if (storage_mode == STORAGE_DISK) {
    int64_t offset = step_idx * (int64_t)step_bytes_uncomp;
    errno = 0;
    if (fseek(fp, offset, SEEK_SET) != 0) {
      report_io_error("fseek(write)", step_idx);
      return;
    }
    if (!write_exact(fp, store_1, step_bytes_uncomp)) {
      report_io_error("fwrite", step_idx);
    }
  }
}

void storage_load_snapshot_cpu(void* store_1, FILE* fp, int64_t storage_mode,
                               int64_t step_idx, size_t step_bytes_uncomp) {
  if (storage_mode == STORAGE_NONE) return;
  if (storage_mode == STORAGE_DISK) {
    int64_t offset = step_idx * (int64_t)step_bytes_uncomp;
    errno = 0;
    if (fseek(fp, offset, SEEK_SET) != 0) {
      report_io_error("fseek(read)", step_idx);
      memset(store_1, 0, step_bytes_uncomp);
      return;
    }
    if (!read_exact(fp, store_1, step_bytes_uncomp)) {
      if (feof(fp)) {
        fprintf(stderr, "storage_utils: unexpected EOF at step %lld\n",
                (long long)step_idx);
      } else {
        report_io_error("fread", step_idx);
      }
      clearerr(fp);
      memset(store_1, 0, step_bytes_uncomp);
    }
  }
}

