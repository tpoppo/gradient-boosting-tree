#pragma once
#include <memory>
#include <numeric>
#include <vector>
using std::size_t;

template<class T> class StaticVector {
  size_t size;
  unique_ptr<T[]> data;

public:
  StaticVector(size_t n) noexcept {
    size = n;
    data = new T[size];
  }

  StaticVector(const std::vector<T> &t_data) noexcept {
    size = t_data.size();
    data = new T[size];
    for (int i = 0; i < size; i++) data[i] = t_data[i];
  }

  StaticVector(const StaticVector<T> &t_data) noexcept {
    size = t_data.size();
    data = new T[size];
    for (int i = 0; i < size; i++) data[i] = t_data[i];
  }

  T &operator[](size_t pos) { return data[pos]; }

  const T &operator[](size_t pos) const { return data[pos]; }


  size_t size() const noexcept { return size; }
};