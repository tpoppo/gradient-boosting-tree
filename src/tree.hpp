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

  void build_decision_table(const Dataset &data) {
    std::vector<int> decision_table_cnt;
    decision_table.resize(1 << features.size());
    decision_table_cnt.resize(1 << features.size());
    const auto &x = data.get_data();
    const auto &y = data.get_target();

    for (size_t i = 0; i < x.size(); i++) {
      unsigned index = 0;
      for (size_t d = 0; d < features.size(); d++) {
        index <<= 1;
        index |= x[i][d] > splitting_value[d];
      }
      decision_table[index] += y[i];
      decision_table_cnt[index] += 1;
    }

    for (size_t i = 0; i < decision_table.size(); i++) {
      if (decision_table_cnt[i]) { decision_table[i] /= decision_table_cnt[i]; }
    }
  }


public:
  Tree(const Dataset &data, std::mt19937 &generator, const unsigned depth = 5)
    : features(depth), splitting_value(depth) {
    for (unsigned i = 0; i < depth; i++) {
      const auto &x = data.get_data();
      features[i] = generator() % x.size();
      splitting_value[i] = x[features[i]][generator() % x[features[i]].size()] + 1e-5;
    }
    build_decision_table(data);
  }


  std::vector<double> predict(const DatasetTest &data) const {
    std::vector<double> ans(data[0].size());
    for (size_t i = 0; i < ans.size(); i++) {
      size_t index = 0;
      for (size_t d = 0; d < features.size(); d++) {
        index <<= 1;
        index |= data[i][d] > splitting_value[d];
      }
      ans[i] = decision_table[index];
    }
    return ans;
  }

  void scale(const double factor) {
    for (size_t i = 0; i < decision_table.size(); i++) { decision_table[i] *= factor; }
  }
};

}// namespace ogbt