#pragma once
#include "dataset.hpp"
#include "tree.hpp"

namespace ogbt {
template<class loss> class Model {

private:
  std::vector<Tree> ensemble_tree;

public:
  Model() {}

  void fit(Dataset data) {}

  auto predict(Dataset t_data) const { return predict(t_data.get_data()); }

  auto predict(DatasetTest t_data) const {
    std::vector<double> data;
    for (const auto &tree : ensemble_tree) {
      std::vector<double> target_prediction = tree.predict(t_data);
      for (size_t i = 0; i < data.size(); i++) data[i] += target_prediction[i];
    }
    return data;
  }
};
}// namespace ogbt