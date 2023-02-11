#pragma once
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>


namespace ogbt {
const double DEFAULT_BUCKET_RATIO = 0.1;
using DataType = uint32_t;
using DatasetTest = std::vector<std::vector<DataType>>;

std::vector<double> get_buckets_uniform(const std::vector<double> &v, uint32_t n_buckets) noexcept {
  std::vector<double> buckets(n_buckets);
  const auto [min_val, max_val] = std::minmax_element(begin(v), end(v));
  for (uint32_t i = 0; i < n_buckets; i++) { buckets[i] = *min_val + (*max_val - *min_val) * i / n_buckets; }
  return buckets;
}

std::vector<DataType> quantize(const std::vector<double> &v, const std::vector<double> &bucket) noexcept {
  std::vector<DataType> result(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    result[i] = std::distance(bucket.begin(), std::lower_bound(bucket.begin(), bucket.end(), v[i]));
  }
  return result;
}

auto auto_quantize(const std::vector<double> &v, std::vector<DataType> &data, uint32_t n_buckets) noexcept {
  std::vector<double> bucket = get_buckets_uniform(v, n_buckets);
  data = quantize(v, bucket);
  return bucket;
}


class Dataset {
private:
  std::vector<std::vector<DataType>> data;// dims: features x samples
  std::vector<double> target;

  std::vector<std::vector<double>> buckets;

  std::vector<std::unordered_map<int, std::pair<double, int>>> target_encoding_counter;
  size_t smooth_target_encoding;
  double mean_target;

public:
  Dataset(const std::vector<std::vector<double>> &t_dense,
    const std::vector<double> &t_target,
    double buckets_ratio = DEFAULT_BUCKET_RATIO) noexcept
    : data{ t_dense.size() }, target{ t_target } {

    const size_t n_buckets = buckets_ratio * t_target.size();
    
    buckets.resize(t_dense.size());
    
    for (size_t i = 0; i < t_dense.size(); i++) { buckets[i] = auto_quantize(t_dense[i], data[i], n_buckets); }
  }

  Dataset(const std::vector<std::vector<int>> &t_categorical,
    const std::vector<std::vector<double>> &t_dense,
    const std::vector<double> &t_target,
    const size_t t_smooth_target_encoding = 100,
    const double buckets_ratio = DEFAULT_BUCKET_RATIO) noexcept {

    target = t_target;
    smooth_target_encoding = t_smooth_target_encoding;
    const size_t n_buckets = buckets_ratio + t_target.size();

    data.reserve(t_dense.size() + t_categorical.size());
    buckets.reserve(data.size());

    // target encoding
    mean_target =
      std::accumulate(target.begin(), target.end(), 0.0) / static_cast<double>(target.size() * smooth_target_encoding);
    target_encoding_counter.reserve(t_categorical.size());
    for (const auto &column : t_categorical) {

      target_encoding_counter.emplace_back();
      std::unordered_map<int, std::pair<double, int>> &counter = target_encoding_counter.back();

      counter.max_load_factor(0.15F);
      for (size_t i = 0; i < column.size(); i++) {
        counter[column[i]].first += target[i];
        counter[column[i]].second += 1;
      }

      for (auto &el : counter) {
        el.second.first = (el.second.first + mean_target) / (el.second.second + smooth_target_encoding);
      }

      std::vector<double> dense_tmp;
      dense_tmp.reserve(column.size());
      for (const int val : column) { dense_tmp.emplace_back(counter[val].first); }

      data.emplace_back();
      buckets.push_back(auto_quantize(dense_tmp, data.back(), n_buckets));
    }

    // continuous feature
    for (size_t i = 0; i < t_dense.size(); i++) {
      data.emplace_back();
      buckets.push_back(auto_quantize(t_dense[i], data.back(), n_buckets));
    }
  }

  const auto &get_x() const noexcept { return this->data; }

  const auto &get_y() const noexcept { return this->target; }

  auto size() const noexcept { return this->target.size(); }
  auto num_features() const noexcept { return this->data.size(); }

  DatasetTest process_test(const std::vector<std::vector<int>> &t_categorical,
    const std::vector<std::vector<double>> &t_dense) const noexcept {
    DatasetTest data_result(t_categorical.size() + t_dense.size());

    for(size_t i=0; i<t_categorical.size(); i++){
      std::vector<double> tmp_data(t_categorical[i].size());
      for(size_t j=0; j<tmp_data.size(); j++) tmp_data[j] = target_encoding_counter[i].at(t_categorical[i][j]).first;
      data_result[i] = quantize(tmp_data, buckets[i]);
    }

    for (size_t i = t_categorical.size(); i < t_categorical.size()+t_dense.size(); i++) {
      data_result[i] = quantize(t_dense[i-t_categorical.size()], buckets[i]);
    }

    return data_result;
  }
};

}// namespace ogbt
