#pragma once
#include "dataset.hpp"
#include "tree.hpp"

namespace ogbt {
template<typename loss> class Model {

private:
  std::vector<Tree> ensemble_tree;
  unsigned num_trees = 100;
  unsigned depth = 5;

public:
  Model(unsigned t_num_trees = 100, unsigned t_depth = 5) {
    num_trees = t_num_trees;
    depth = t_depth;
  }

  void fit(const Dataset &data) {}

  auto predict(const Dataset &t_data) const { return predict(t_data.get_x()); }

  std::vector<double> predict(DatasetTest t_data) const {
    std::vector<double> data;
    for (const auto &tree : ensemble_tree) {
      std::vector<double> target_prediction = tree.predict(t_data);
      for (size_t i = 0; i < data.size(); i++) data[i] += target_prediction[i];
    }
    return data;
  }
};
}// namespace ogbt