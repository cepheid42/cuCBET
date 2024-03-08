//
// Created by cepheid on 3/7/24.
//

#ifndef CUCBET_SMART_POINTER_CUH
#define CUCBET_SMART_POINTER_CUH

template<typename T>
struct cuda_deleter {
  void operator()(T* ptr) {
    cudaChk(cudaDeviceSynchronize());
    cudaChk(cudaFree(ptr));
  }
};

template<typename T, class Deleter = cuda_deleter<T>>
class dev_ptr {
private:
  T* _ptr;

public:
  dev_ptr() = default;
  explicit dev_ptr(T* ptr) : _ptr(ptr) {}

  dev_ptr(dev_ptr&& u) = delete;
  dev_ptr(const dev_ptr&) = delete;

  ~dev_ptr() { get_deleter()(_ptr); }

//  template<typename U>
//  dev_ptr& operator=(dev_ptr<U>&& r) {}
  dev_ptr& operator=(dev_ptr&& r) = delete;
  dev_ptr& operator=(const dev_ptr&) = delete;

  explicit operator bool() const { return _ptr != nullptr; }
  T* get() { return _ptr; }
  T* operator->() { return _ptr; }
  T& operator*() { return *(_ptr); }

  T* release() noexcept {
    auto* old_ptr = _ptr;
    get_deleter()(_ptr);
    return old_ptr;
  }

  void reset(T* ptr) noexcept {
    auto* old_ptr = _ptr;
    _ptr = ptr;
    get_deleter()(old_ptr);
  }

  Deleter& get_deleter() noexcept { return Deleter(); }
  const Deleter& get_deleter() const noexcept { return Deleter(); }
};

template<typename T, class Deleter>
class dev_ptr<T[], Deleter> {
private:
  T* _ptr;

public:
  dev_ptr() = default;
  explicit dev_ptr(T* ptr) : _ptr(ptr) {}

  dev_ptr(dev_ptr&& u) = default;
  dev_ptr(const dev_ptr&) = delete;

  ~dev_ptr() { get_deleter()(_ptr); }

//  template<typename U>
//  dev_ptr& operator=(dev_ptr<U>&& r) {}
  dev_ptr& operator=(dev_ptr&& r) = delete;
  dev_ptr& operator=(const dev_ptr&) = delete;

  explicit operator bool() const { return _ptr != nullptr; }
  T* get() { return _ptr; }
  T* operator->() { return _ptr; }
  T& operator*() { return *(_ptr); }

  T* release() noexcept {
    auto* old_ptr = _ptr;
    get_deleter()(_ptr);
    return old_ptr;
  }

  void reset(T* ptr) noexcept {
    auto* old_ptr = _ptr;
    _ptr = ptr;
    get_deleter()(old_ptr);
  }

  T& operator[](size_t i) const { return _ptr[i]; }

  Deleter& get_deleter() noexcept { return Deleter(); }
  const Deleter& get_deleter() const noexcept { return Deleter(); }
};

// Non-array type T
template<typename T, class... Args>
dev_ptr<T> make_dev_ptr(Args&&... args) {
  T* ptr;
  cudaChk(cudaMallocManaged(&ptr, sizeof(T)));
  cudaChk(cudaDeviceSynchronize());
  *ptr = T(args...);
  return dev_ptr<T>(ptr);
}

// Array type T
template<typename T>
dev_ptr<T> make_dev_ptr(size_t size) {
  T* ptr;
  cudaChk(cudaMallocManaged(&ptr, size * sizeof(T())));
  cudaChk(cudaDeviceSynchronize());
  for (size_t i = 0; i < size; ++i) { ptr[i] = T{}; }
  return dev_ptr<T>(ptr);
}

#endif //CUCBET_SMART_POINTER_CUH
