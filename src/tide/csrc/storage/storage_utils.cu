#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "storage_utils.h"

#ifndef TIDE_CUDA_ARCHES_STR
#define TIDE_CUDA_ARCHES_STR "unknown"
#endif

static void report_cuda_error(const char* op, cudaError_t err) {
  fprintf(stderr, "storage_utils: %s failed: %s\n",
          op, cudaGetErrorString(err));
}

namespace {

struct AsyncDiskTask {
  int64_t slot_idx;
  void* host_ptr;
  size_t nbytes;
  int64_t file_offset;
  cudaEvent_t ready_event;
  bool is_write;
};

class AsyncDiskFile {
 public:
  AsyncDiskFile(const char* path, bool write_mode, int64_t slot_count)
      : slot_busy_(static_cast<size_t>(slot_count), false), closing_(false) {
    int flags = write_mode ? (O_CREAT | O_TRUNC | O_WRONLY) : O_RDONLY;
    fd_ = ::open(path, flags, 0600);
    if (fd_ < 0) {
      int err = errno;
      fprintf(stderr, "storage_utils: open(%s) failed: %s\n", path,
              strerror(err));
      abort();
    }
    worker_ = std::thread(&AsyncDiskFile::run, this);
  }

  ~AsyncDiskFile() {
    close();
  }

  void wait_slot(int64_t slot_idx) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] {
      return !slot_busy_[static_cast<size_t>(slot_idx)] || !slot_error_.empty();
    });
    maybe_abort_locked();
  }

  void enqueue_write(int64_t slot_idx, void* host_ptr, size_t nbytes,
                     int64_t file_offset, cudaEvent_t ready_event) {
    enqueue_task(
        AsyncDiskTask{slot_idx, host_ptr, nbytes, file_offset, ready_event, true});
  }

  void enqueue_read(int64_t slot_idx, void* host_ptr, size_t nbytes,
                    int64_t file_offset, cudaEvent_t ready_event) {
    enqueue_task(AsyncDiskTask{slot_idx, host_ptr, nbytes, file_offset,
                               ready_event, false});
  }

  void close() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (closing_) {
        return;
      }
      closing_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      maybe_abort_locked();
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

 private:
  void enqueue_task(AsyncDiskTask task) {
    std::lock_guard<std::mutex> lock(mu_);
    size_t const slot = static_cast<size_t>(task.slot_idx);
    if (slot_busy_[slot]) {
      fprintf(stderr, "storage_utils: slot %lld reused before completion\n",
              (long long)task.slot_idx);
      abort();
    }
    slot_busy_[slot] = true;
    tasks_.push_back(task);
    cv_.notify_all();
  }

  static bool io_exact_pwrite(int fd, const void* src, size_t nbytes,
                              int64_t offset) {
    size_t total = 0;
    while (total < nbytes) {
      ssize_t n = ::pwrite(fd, (const char*)src + total, nbytes - total,
                           offset + (int64_t)total);
      if (n <= 0) {
        return false;
      }
      total += (size_t)n;
    }
    return true;
  }

  static bool io_exact_pread(int fd, void* dst, size_t nbytes, int64_t offset) {
    size_t total = 0;
    while (total < nbytes) {
      ssize_t n =
          ::pread(fd, (char*)dst + total, nbytes - total, offset + (int64_t)total);
      if (n <= 0) {
        return false;
      }
      total += (size_t)n;
    }
    return true;
  }

  void run() {
    while (true) {
      AsyncDiskTask task{};
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return closing_ || !tasks_.empty(); });
        if (tasks_.empty()) {
          if (closing_) {
            break;
          }
          continue;
        }
        task = tasks_.front();
        tasks_.pop_front();
      }

      bool ok = true;
      int saved_errno = 0;
      if (task.ready_event != nullptr) {
        cudaError_t err = cudaEventSynchronize(task.ready_event);
        cudaEventDestroy(task.ready_event);
        if (err != cudaSuccess) {
          std::lock_guard<std::mutex> lock(mu_);
          slot_error_ = std::string("cudaEventSynchronize failed: ") +
                        cudaGetErrorString(err);
          slot_busy_[static_cast<size_t>(task.slot_idx)] = false;
          cv_.notify_all();
          continue;
        }
      }

      if (task.is_write) {
        ok = io_exact_pwrite(fd_, task.host_ptr, task.nbytes, task.file_offset);
      } else {
        ok = io_exact_pread(fd_, task.host_ptr, task.nbytes, task.file_offset);
      }
      if (!ok) {
        saved_errno = errno;
      }

      {
        std::lock_guard<std::mutex> lock(mu_);
        slot_busy_[static_cast<size_t>(task.slot_idx)] = false;
        if (!ok) {
          slot_error_ =
              std::string(task.is_write ? "pwrite failed: " : "pread failed: ") +
              strerror(saved_errno);
        }
      }
      cv_.notify_all();
    }
  }

  void maybe_abort_locked() const {
    if (!slot_error_.empty()) {
      fprintf(stderr, "storage_utils: async disk I/O failed: %s\n",
              slot_error_.c_str());
      abort();
    }
  }

  int fd_{-1};
  std::vector<bool> slot_busy_;
  std::string slot_error_;
  std::deque<AsyncDiskTask> tasks_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::thread worker_;
  bool closing_;
};

}  // namespace

extern "C" {

const char* tide_cuda_arches(void) {
  return TIDE_CUDA_ARCHES_STR;
}

void storage_copy_snapshot_d2h(void* device_store, void* host_store,
                               size_t shot_bytes_uncomp, size_t n_shots,
                               void* stream) {
  size_t const bytes_to_store = shot_bytes_uncomp * n_shots;
  cudaError_t const err =
      cudaMemcpyAsync(host_store, device_store, bytes_to_store,
                      cudaMemcpyDeviceToHost, (cudaStream_t)stream);
  if (err != cudaSuccess) {
    report_cuda_error("cudaMemcpyAsync(D2H)", err);
  }
}

void storage_copy_snapshot_h2d(void* device_store, void* host_store,
                               size_t shot_bytes_uncomp, size_t n_shots,
                               void* stream) {
  size_t const bytes_to_load = shot_bytes_uncomp * n_shots;
  cudaError_t const err =
      cudaMemcpyAsync(device_store, host_store, bytes_to_load,
                      cudaMemcpyHostToDevice, (cudaStream_t)stream);
  if (err != cudaSuccess) {
    report_cuda_error("cudaMemcpyAsync(H2D)", err);
  }
}

void* storage_async_disk_open(const char* path, bool write_mode,
                              int64_t slot_count) {
  return new AsyncDiskFile(path, write_mode, slot_count);
}

void storage_async_disk_close(void* handle) {
  if (handle == nullptr) return;
  delete static_cast<AsyncDiskFile*>(handle);
}

void storage_async_disk_wait_slot(void* handle, int64_t slot_idx) {
  if (handle == nullptr) return;
  static_cast<AsyncDiskFile*>(handle)->wait_slot(slot_idx);
}

void storage_async_disk_enqueue_write(void* handle, int64_t slot_idx,
                                      void* host_ptr, size_t nbytes,
                                      int64_t file_offset, void* ready_event) {
  if (handle == nullptr) return;
  static_cast<AsyncDiskFile*>(handle)->enqueue_write(
      slot_idx, host_ptr, nbytes, file_offset,
      static_cast<cudaEvent_t>(ready_event));
}

void storage_async_disk_enqueue_read(void* handle, int64_t slot_idx,
                                     void* host_ptr, size_t nbytes,
                                     int64_t file_offset, void* ready_event) {
  if (handle == nullptr) return;
  static_cast<AsyncDiskFile*>(handle)->enqueue_read(slot_idx, host_ptr, nbytes,
                                                    file_offset,
                                                    static_cast<cudaEvent_t>(
                                                        ready_event));
}
}
