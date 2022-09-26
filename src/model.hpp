#pragma once
#include "dataset.hpp"
#include "tree.hpp"
#include <functional>

namespace ogbt {
using TreeGenerator = std::function<Tree(const DatasetTest &, const std::vector<double> &)>;
template<typename TLoss> class Model {

private:
  std::vector<Tree> ensemble_trees;

  TreeGenerator tree_generator;
  unsigned num_trees;
  unsigned depth;
  double learning_rate;

public:
  Model(TreeGenerator t_tree_generator, unsigned t_num_trees = 100, unsigned t_depth = 5, double t_learning_rate = 0.1)
    : tree_generator{ t_tree_generator }, num_trees{ t_num_trees }, depth{ t_depth }, learning_rate{ t_learning_rate } {}

  void fit(const Dataset &data) {
    const auto &x = data.get_x();
    const auto &y = data.get_y();
    auto residual = data.get_y();
    std::vector<double> y_pred(data.get_y());
    for (unsigned i = 0; i < num_trees; i++) {
      ensemble_trees.push_back(tree_generator(x, residual));
      ensemble_trees.back().scale(learning_rate);
      const auto &y_pred_tree = ensemble_trees.back().predict(x);
      for (size_t j = 0; j < y_pred.size(); j++) y_pred[j] += y_pred_tree[j];
      residual = TLoss::residual(y, y_pred);
    }
  }

  auto predict(const Dataset &t_data) const { return predict(t_data.get_x()); }

  std::vector<double> predict(DatasetTest t_data) const {
    std::vector<double> data;
    for (const auto &tree : ensemble_trees) {
      std::vector<double> target_prediction = tree.predict(t_data);
      for (size_t i = 0; i < data.size(); i++) data[i] += target_prediction[i];
    }
    return data;
  }
};
}// namespace ogbt