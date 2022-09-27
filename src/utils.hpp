#pragma once
#include "dataset.hpp"

namespace ogbt {
Dataset get_dummy_data(const int n, const int m, std::mt19937 &gen) {

  std::vector<std::vector<double>> data_dense;
  std::vector<double> target(n);
  std::normal_distribution<> d0{ 4, 2 };
  std::normal_distribution<> d1{ 1, 2 };
  for (int i = 0; i < n; i++) { target[i] = gen() % 2; }

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
  return ogbt::Dataset{ data_dense, target };
}


}// namespace ogbt