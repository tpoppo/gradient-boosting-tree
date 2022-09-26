#pragma once
#include "dataset.hpp"
#include <random>
#include <type_traits>

namespace ogbt {

class Tree {
private:
  std::vector<int> decision_table;
  std::vector<int> features;
  std::vector<double> splitting_value;

public:
  Tree(const Dataset &data, std::mt19937 &generator, const unsigned depth = 5)
    : features(depth), splitting_value(depth) {
    for (unsigned i = 0; i < depth; i++) {
      const auto &x = data.get_x();
      features[i] = generator() % x.size();
      splitting_value[i] = x[features[i]][generator() % x[features[i]].size()] + 1e-5;
    }
    build_decision_table(data);
  }

  Tree(const Dataset &data, const std::vector<int> &t_features, const std::vector<double> &t_splitting_value) {
    features = t_features;
    splitting_value = t_splitting_value;
    build_decision_table(data);
  }

  std::vector<double> predict(const DatasetTest &data) const {
    std::vector<double> ans(data[0].size());
    for (size_t i = 0; i < ans.size(); i++) {
      size_t index = 0;
      for (size_t d = 0; d < features.size(); d++) {
        index <<= 1;
        index |= data[features[d]][i] > splitting_value[d];
      }
      ans[i] = decision_table[index];
    }
    return ans;
  }

  void scale(const double factor) {
    for (size_t i = 0; i < decision_table.size(); i++) { decision_table[i] *= factor; }
  }

  void build_decision_table(const Dataset &data) {
    std::vector<int> decision_table_cnt;
    decision_table.clear();
    decision_table.resize(1 << features.size());
    decision_table_cnt.resize(1 << features.size());
    const auto &x = data.get_x();
    const auto &y = data.get_y();

    for (size_t i = 0; i < data.size(); i++) {
      unsigned index = 0;
      for (size_t d = 0; d < features.size(); d++) {
        index <<= 1;
        index |= x[features[d]][i] > splitting_value[d];
      }
      decision_table[index] += y[i];
      decision_table_cnt[index] |= 1;
    }

    for (size_t i = 0; i < decision_table.size(); i++) {
      if (decision_table_cnt[i]) { decision_table[i] /= decision_table_cnt[i]; }
    }
  }

  auto &get_features() { return features; }

  auto &get_splitting_value() { return splitting_value; }
};

}// namespace ogbt