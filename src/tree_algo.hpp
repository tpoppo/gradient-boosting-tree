#pragma once
#include "dataset.hpp"
#include "tree.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cassert>
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

struct ScoreFeature {
  double score;
  double splitting_value;
  int feature;

  ScoreFeature(double t_score, double t_splitting_value, int t_feature)
    : score{ t_score }, splitting_value{ t_splitting_value }, feature{ t_feature } {}

  bool operator<(const ScoreFeature &r) const { return score < r.score; }
};

template<typename TLoss>
Tree genetic_algo(const DatasetTest &x,
  const std::vector<double> &y,
  const uint8_t tree_depth = 5,
  const unsigned iterations = 2,
  const unsigned population = 100,
  const unsigned selected = 5,
  const unsigned new_mutations = 50,
  const unsigned num_mutations = 2,
  const double subsample = 0.5) noexcept {
  std::random_device random_dev;
  std::mt19937 generator(random_dev());

  std::vector<ScoreTree> pop_trees;

  unsigned sample_size = std::max(100.0, y.size() * subsample);
  for (size_t t = 0; t < iterations; t++) {
    pop_trees.reserve(population);

    auto [x_curr, y_curr] = get_subsample(x, y, generator, sample_size);
    int new_population = population - pop_trees.size();
    for (int j = 0; j < new_population; j++) {
      pop_trees.emplace_back(x_curr, y_curr, generator, tree_depth);
      double score = TLoss::score(pop_trees.back().tree.predict(x_curr), y_curr);
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
        features[mut_depth] = generator() % x_curr.size();// change the splitting feature
        splitting_value[mut_depth] = x_curr[features[mut_depth]][generator() % y_curr.size()];
      }
      parent.score = TLoss::score(pop_trees.back().tree.predict(x_curr), y_curr);
      pop_trees.push_back(parent);
    }
  }
  return std::max_element(pop_trees.begin(), pop_trees.end())->tree;
}


Tree greedy_mse_splitting(const DatasetTest &x_full,
  const std::vector<double> &y_full,
  const uint8_t tree_depth = 5,
  const double lambda_reg = 10.0,
  const double subsample = 0.5) noexcept {

  assert(x_full.size() >= tree_depth);
  
  std::vector<ScoreFeature> feature_score;
  std::random_device random_dev;
  std::mt19937 generator(random_dev());
  auto sample_size = subsample * y_full.size();
  auto [x, y] = get_subsample(x_full, y_full, generator, sample_size);

  for (size_t feat = 0; feat < x.size(); feat++) {
    std::vector<int> x_order(y.size());
    for (size_t i = 0; i < x_order.size(); i++) x_order[i] = i;

    std::sort(x_order.begin(), x_order.end(), [&x, &feat](int l, int r) { return x[feat][l] < x[feat][r]; });

    double l_sum = 0;
    double r_sum = 0;
    double l_squared_sum = 0;
    double r_squared_sum = 0;
    double best_score;
    double best_value = x_order[0] - 1e-5;

    for (size_t i = 0; i < x_order.size(); i++) {
      r_sum += y[x_order[i]];
      r_squared_sum += y[x_order[i]] * y[x_order[i]];
    }

    best_score = r_squared_sum / x_order.size() - r_sum * r_sum / x_order.size() / x_order.size();

    for (size_t i = 0; i < x_order.size(); i++) {
      l_sum += y[x_order[i]];
      r_sum -= y[x_order[i]];

      l_squared_sum += y[x_order[i]] * y[x_order[i]];
      r_squared_sum -= y[x_order[i]] * y[x_order[i]];


      auto score = l_squared_sum / (1 + i) - l_sum * l_sum / (1 + i) / (1 + i);

      if (y.size() - i - 1 > 0) {
        score += r_squared_sum / (y.size() - i - 1) - r_sum * r_sum / (y.size() - i - 1) / (y.size() - i - 1);
      }
      score += lambda_reg * (y.size()/2-i)*(y.size()/2-i)/y.size()/y.size();

      if (score < best_score) {
        best_score = score;
        if (i + 1 < x_order.size()) {
          best_value = x_order[i] + 0.5 * (x_order[i + 1] - x_order[i]);
        } else {
          best_value = x_order[i] + 1e-5;
        }
      }
    }
    assert(best_score >= 0);
    feature_score.emplace_back(best_score, best_value, feat);
  }

  std::nth_element(feature_score.begin(), feature_score.begin() + tree_depth, feature_score.end());
  std::vector<int> features(tree_depth);
  std::vector<double> splitting_value(tree_depth);
  for (size_t i = 0; i < features.size(); i++) features[i] = feature_score[i].feature;
  for (size_t i = 0; i < splitting_value.size(); i++) splitting_value[i] = feature_score[i].splitting_value;
  return Tree{ x, y, features, splitting_value };
}


}// namespace ogbt
