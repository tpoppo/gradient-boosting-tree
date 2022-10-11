#pragma once
#include "dataset.hpp"
#include "tree.hpp"
#include <functional>
#include <iostream>

namespace ogbt {
using TreeGenerator = std::function<Tree(const DatasetTest &, const std::vector<double> &)>;
template<typename TLoss> class Model {

private:
  std::vector<Tree> ensemble_trees;

  const TreeGenerator tree_generator;
  const uint16_t num_trees;
  const double learning_rate;

public:
  Model(TreeGenerator t_tree_generator,
    const uint16_t t_num_trees = 50,
    const double t_learning_rate = 0.1) noexcept
    : tree_generator{ t_tree_generator }, num_trees{ t_num_trees }, learning_rate{ t_learning_rate } {
  }

  void fit(const Dataset &data) {
    const auto &x = data.get_x();
    const auto &y = data.get_y();
    std::vector<double> y_pred(data.get_y().size());

    for (unsigned i = 0; i < num_trees; i++) {
      auto residual = TLoss::residual(y_pred, y);

#ifndef NDEBUG
      auto score = TLoss::score(y_pred, y);
      std::cout << "score(" << i << "): " << score << std::endl;
#endif

      ensemble_trees.push_back(tree_generator(x, residual));
      ensemble_trees.back().scale(learning_rate);

      const auto &y_pred_tree = ensemble_trees.back().predict(x);
      for (size_t j = 0; j < y_pred.size(); j++) { y_pred[j] += y_pred_tree[j]; }
    }
  }

  auto predict(const Dataset &t_data) const noexcept { return predict(t_data.get_x()); }

  std::vector<double> predict(DatasetTest t_data) const noexcept {
    std::vector<double> data(t_data[0].size());
    for (const auto &tree : ensemble_trees) {
      std::vector<double> target_prediction = tree.predict(t_data);
      for (size_t i = 0; i < data.size(); i++) data[i] += target_prediction[i];
    }
    return data;
  }
};
}// namespace ogbt