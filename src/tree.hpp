#pragma once
#include "dataset.hpp"
#include <cassert>
#include <random>
#include <type_traits>

namespace ogbt {

class Tree {
private:
  std::vector<double> decision_table;
  std::vector<int> features;
  std::vector<double> splitting_value;

public:
  Tree(const Dataset &data, std::mt19937 &generator, const unsigned depth = 5) noexcept
    : Tree(data.get_x(), data.get_y(), generator, depth) {}

  Tree(const DatasetTest &x, const std::vector<double> &y, std::mt19937 &generator, const unsigned depth = 5) noexcept
    : features(depth), splitting_value(depth) {

    for (unsigned i = 0; i < depth; i++) {
      features[i] = generator() % x.size();
      splitting_value[i] = x[features[i]][generator() % x[features[i]].size()] + 1e-5;
    }

    build_decision_table(x, y);
  }

  Tree(const Dataset &data, const std::vector<int> &t_features, const std::vector<double> &t_splitting_value) noexcept
    : Tree(data.get_x(), data.get_y(), t_features, t_splitting_value) {
    assert(t_features.size() == t_splitting_value.size());
  }

  Tree(const DatasetTest &x,
    const std::vector<double> &y,
    const std::vector<int> &t_features,
    const std::vector<double> &t_splitting_value) noexcept {
    assert(t_features.size() == t_splitting_value.size());
    features = t_features;
    splitting_value = t_splitting_value;
    build_decision_table(x, y);
  }

  std::vector<double> predict(const DatasetTest &data) const noexcept {
    std::vector<double> ans(data[0].size());
    for (size_t i = 0; i < ans.size(); i++) {

      unsigned index = 0;

      for (size_t d = 0; d < features.size(); d++) { index |= ((data[features[d]][i] > splitting_value[d]) << d); }

      ans[i] = decision_table[index];
    }
    return ans;
  }

  void scale(const double factor) noexcept {
    for (size_t i = 0; i < decision_table.size(); i++) { decision_table[i] *= factor; }
  }

  void build_decision_table(const Dataset &data) noexcept { build_decision_table(data.get_x(), data.get_y()); }

  void build_decision_table(const DatasetTest &x, const std::vector<double> &y) noexcept {
    std::vector<int> decision_table_cnt;
    decision_table.clear();
    decision_table.resize(1 << features.size());
    decision_table_cnt.resize(1 << features.size());

    for (size_t i = 0; i < y.size(); i++) {
      unsigned index = 0;
      for (size_t d = 0; d < features.size(); d++) { index |= ((x[features[d]][i] > splitting_value[d]) << d); }
      decision_table[index] += y[i];
      ++decision_table_cnt[index];
    }

    for (size_t i = 0; i < decision_table.size(); i++) {
      if (decision_table_cnt[i]) { decision_table[i] /= decision_table_cnt[i]; }
    }
  }

  constexpr auto &get_features() noexcept { return features; }

  constexpr auto &get_splitting_value() noexcept { return splitting_value; }
};

}// namespace ogbt