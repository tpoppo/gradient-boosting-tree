#pragma once
#include "dataset.hpp"
#include <cassert>

namespace ogbt {

constexpr uint64_t rol64(uint64_t x, uint8_t k) { return (x << k) | (x >> (64 - k)); }

class Xoshiro256p {
  uint64_t s[4];

public:
  Xoshiro256p() : Xoshiro256p{ 0 } {}
  Xoshiro256p(uint64_t seed) {
    s[0] = seed;
    s[1] = seed ^ 0xabcedfaaull;
    s[2] = seed ^ 0x342abbbbull;
    s[3] = seed ^ 0xaa1dddddull;
  }

  uint64_t operator()() {
    uint64_t const result = s[0] + s[3];
    uint64_t const t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rol64(s[3], 45);

    return result >> 8;
  }

  static constexpr uint64_t min() { return 0; }
  static constexpr uint64_t max() { return 1ull << 32; }
};


Dataset get_dummy_data(const int n, const int m, std::mt19937 &gen) noexcept {

  std::vector<std::vector<float>> data_dense;
  std::vector<float> target(n);
  std::normal_distribution<> d0{ 4, 2 };
  std::normal_distribution<> d1{ 1, 2 };
  for (int i = 0; i < n; i++) { target[i] = static_cast<float>(gen() % 2); }

  for (int i = 0; i < m; i++) {
    data_dense.emplace_back(n);
    for (int j = 0; j < n; j++) {
      if (target[j] > 0.5) {
        data_dense[i][j] = d1(gen);
      } else {
        data_dense[i][j] = d0(gen);
      }
    }
  }
  return ogbt::Dataset(data_dense, target);
}


std::pair<DatasetTest, std::vector<float>>
  get_subsample(const DatasetTest &x, const std::vector<float> &y, std::mt19937 &gen, const unsigned n) noexcept {
  DatasetTest x_ans(x.size(), std::vector<DataType>(n));
  std::vector<float> y_ans(n);

  for (size_t i = 0; i < n; i++) {
    size_t index = gen() % y.size();
    y_ans[i] = y[index];
    for (size_t j = 0; j < x.size(); j++) x_ans[j][i] = x[j][index];
  }
  return { x_ans, y_ans };
}

std::pair<DatasetTest, std::vector<float>> get_goss(const DatasetTest &x,
  const std::vector<float> &y,
  std::mt19937 &gen,
  const unsigned a_n,
  const unsigned b_n,
  const std::vector<bool> selected_features) noexcept {
  /*
  Based on "Gradient-based One-Side Sampling" from "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
  https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf

  Faster implementation
  */
  assert(a_n <= y.size());
  assert(selected_features.size() == x.size());

  uint32_t n_features = 0;
  for (size_t i = 0; i < selected_features.size(); i++) n_features += selected_features[i];
  assert(n_features > 0);

  DatasetTest x_ans(n_features, std::vector<DataType>(a_n + b_n));
  std::vector<float> y_ans(a_n + b_n);

  std::vector<size_t> y_order(y.size());
  for (size_t i = 0; i < y.size(); i++) y_order[i] = i;
  sort(y_order.begin(), y_order.end(), [&](const int a, const int b) { return abs(y[a]) > abs(y[b]); });

  for (unsigned i = 0; i < a_n; i++) {
    y_ans[i] = y[y_order[i]];
    uint32_t cnt = 0;
    for (size_t j = 0; j < x.size(); j++) {
      if (selected_features[j]) {
        x_ans[cnt][i] = x[j][y_order[i]];
        ++cnt;
      }
    }
  }

  for (unsigned i = a_n; i < a_n + b_n; i++) {
    size_t index = gen() % y.size();
    y_ans[i] = y[index];
    uint32_t cnt = 0;
    for (size_t j = 0; j < x.size(); j++) {
      if (selected_features[j]) {
        x_ans[cnt][i] = x[j][index];
        ++cnt;
      }
    }
  }

  return { x_ans, y_ans };
}


}// namespace ogbt