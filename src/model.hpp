#include "dataset.hpp"
#include "loss.hpp"
#include "tree.hpp"

template<class loss> class Model {
  static_assert(std::is_base_of<Loss, loss>::value, "The loss must derive from Loss");

private:
  std::vector<Tree<loss>> ensemble_tree;

public:
  Model() {}

  void fit(Dataset data) {}

  auto predict(data) {
    vector<double> data for (const auto &tree : ensemble_tree) {}
  }
};