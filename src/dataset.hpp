#pragma once
#include <numeric>
#include <unordered_map>
#include <vector>

namespace ogbt {
using DatasetTest = std::vector<std::vector<double>>;

class Dataset {
private:
  std::vector<std::vector<double>> data;
  std::vector<double> target;

  std::vector<std::unordered_map<int, std::pair<double, int>>> target_encoding_counter;
  int smooth_target_encoding{ 0 };
  double mean_target{ 0 };

public:
  Dataset(const std::vector<std::vector<int>> &t_categorical,
    const std::vector<std::vector<double>> &t_dense,
    const std::vector<double> &t_target,
    const int t_smooth_target_encoding = 100) {

    data = t_dense;
    target = t_target;
    smooth_target_encoding = t_smooth_target_encoding;

    data.reserve(t_dense.size() + t_categorical.size());

    // target encoding
    mean_target = std::accumulate(target.begin(), target.end(), 0.0) / target.size() * smooth_target_encoding;
    target_encoding_counter.reserve(t_categorical.size());
    for (const auto &column : t_categorical) {
      this->data.emplace_back();

      target_encoding_counter.emplace_back();
      std::unordered_map<int, std::pair<double, int>> &counter = target_encoding_counter.back();

      counter.max_load_factor(0.15);
      for (size_t i = 0; i < column.size(); i++) {
        counter[column[i]].first += target[i];
        counter[column[i]].second += 1;
      }

      for (auto &el : counter) {
        el.second.first = (el.second.first + mean_target) / (el.second.second + smooth_target_encoding);
      }

      for (const int val : column) { this->data.back().emplace_back(counter[val].first); }
    }
  }

  const auto &get_data() const { return this->data; }

  const auto &get_target() const { return this->target; }

  const auto process_test(const std::vector<std::vector<int>> &t_categorical,
    const std::vector<std::vector<double>> &t_dense) {

    std::vector<std::vector<double>> data_result = t_dense;

    data_result.reserve(t_categorical.size() + t_dense.size());
    for (size_t i = 0; i < t_categorical.size(); i++) {
      data_result.emplace_back();
      data_result.back().reserve(t_categorical[i].size());

      for (const int val : t_categorical[i]) {
        const auto &target_encoding = target_encoding_counter[i][val];
        data_result.back().emplace_back(target_encoding.first);
      }
    }

    return data_result;
  }
};
}// namespace ogbt
