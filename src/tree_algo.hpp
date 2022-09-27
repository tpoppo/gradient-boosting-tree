#pragma once
#include "dataset.hpp"
#include "tree.hpp"
#include <algorithm>
#include <random>
#include <vector>

namespace ogbt {

struct ScoreTree {
  double score;
  Tree tree;

  ScoreTree(const DatasetTest &x, const std::vector<double> &y, std::mt19937 &t_generator, unsigned tree_depth)
    : score(0.0), tree{ x, y, t_generator, tree_depth } {}
  bool operator<(const ScoreTree &r) const { return score < r.score; }
};

template<typename TLoss>
Tree genetic_algo(const DatasetTest &x,
  const std::vector<double> &y,
  unsigned tree_depth = 5,
  unsigned iterations = 2,
  unsigned population = 100,
  unsigned selected = 5,
  unsigned new_mutations = 50,
  unsigned num_mutations = 2) noexcept {
  std::random_device random_dev;
  std::mt19937 generator(random_dev());

  std::vector<ScoreTree> pop_trees;

  for (size_t t = 0; t < iterations; t++) {
    pop_trees.reserve(population);

    int new_population = population - pop_trees.size();
    for (int j = 0; j < new_population; j++) {
      pop_trees.emplace_back(x, y, generator, tree_depth);
      double score = TLoss::score(pop_trees.back().tree.predict(x), y);
      pop_trees.back().score = score;
    }
    if (t + 1 < iterations) {// skip in the last iteration
      // select best elements
      std::nth_element(pop_trees.begin(), pop_trees.begin() + selected, pop_trees.end());
      pop_trees.erase(pop_trees.begin() + selected + 1, pop_trees.end());
    }

    // generate new samples
    for (size_t i = 0; i < new_mutations; i++) {
      auto parent = pop_trees[generator() % selected];
      auto &splitting_value = parent.tree.get_splitting_value();
      auto &features = parent.tree.get_features();

      for (size_t j = 0; j < num_mutations; j++) {
        int mut_depth = generator() % splitting_value.size();// random depth
        features[mut_depth] = generator() % x.size();// change the splitting feature
        splitting_value[mut_depth] = x[features[mut_depth]][generator() % y.size()];
      }
      parent.score = TLoss::score(pop_trees.back().tree.predict(x), y);
      pop_trees.push_back(parent);
    }
  }
  return std::max_element(pop_trees.begin(), pop_trees.end())->tree;
}

}// namespace ogbt
