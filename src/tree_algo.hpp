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
  const double subsample_a = 0.1,
  const double subsample_b = 0.1) noexcept {
  std::random_device random_dev;
  std::mt19937 generator(random_dev());

  std::vector<ScoreTree> pop_trees;

  unsigned sample_size_a = std::min(std::max(100ul, static_cast<size_t>(y.size() * subsample_a)), y.size());
  unsigned sample_size_b = std::max(100ul, static_cast<size_t>(y.size() * subsample_b));

  for (size_t t = 0; t < iterations; t++) {
    pop_trees.reserve(population);

    auto [x_curr, y_curr] = get_goss(x, y, generator, sample_size_a, sample_size_b);
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
  const double subsample_a = 0.1,
  const double subsample_b = 0.1) noexcept {

  assert(x_full.size() >= tree_depth);

  std::vector<ScoreFeature> feature_score;
  std::random_device random_dev;
  std::mt19937 generator(random_dev());
  unsigned sample_size_a = std::min(std::max(100ul, static_cast<size_t>(y_full.size() * subsample_a)), y_full.size());
  unsigned sample_size_b = std::max(100ul, static_cast<size_t>(y_full.size() * subsample_b));

  auto [x, y] = get_goss(x_full, y_full, generator, sample_size_a, sample_size_b);

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

      if (score < best_score) {
        best_score = score;
        if (i + 1 < x_order.size()) {
          best_value = x[feat][x_order[i]] + 0.5 * (x[feat][x_order[i + 1]] - x[feat][x_order[i]]);
        } else {
          best_value = x[feat][x_order[i]] + 1e-5;
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


Tree mse_splitting_bdt(const DatasetTest &x_full,
  const std::vector<double> &y_full,
  const uint8_t tree_depth = 5,
  const unsigned steps = 10,
  const double subsample_a = 0.1,
  const double subsample_b = 0.1) noexcept {
  /*
  Based on BDT: Gradient Boosted Decision Tables for High Accuracy and Scoring Efficiency
  https://yinlou.github.io/papers/lou-kdd17.pdf

  Faster implementation
  */
  assert(steps >= tree_depth);

  std::random_device random_dev;
  std::mt19937 generator(random_dev());
  unsigned sample_size_a = std::min(std::max(100ul, static_cast<size_t>(y_full.size() * subsample_a)), y_full.size());
  unsigned sample_size_b = std::max(100ul, static_cast<size_t>(y_full.size() * subsample_b));

  auto [x, y] = get_goss(x_full, y_full, generator, sample_size_a, sample_size_b);

  std::vector<int> L(y.size());
  std::vector<int> features(tree_depth);
  std::vector<double> splitting_value(tree_depth, -1e100);

  std::vector<int> counter(1 << tree_depth);
  std::vector<double> sum(1 << tree_depth);
  double current_score;

  for (size_t i = 0; i < y.size(); i++) sum[0] += y[i];
  counter[0] = y.size();
  current_score = sum[0] * sum[0] / counter[0];

  for (unsigned t = 0; t < steps; t++) {
    const int k = t % tree_depth;

    double best_score = -1e100;
    double best_cut = -1e100;
    int best_feat = 0;

    std::vector<int> x_order(y.size());
    for (size_t i = 0; i < x_order.size(); i++) x_order[i] = i;

    for (size_t feat = 0; feat < x.size(); feat++) {
      std::sort(x_order.begin(), x_order.end(), [&x, &feat](int l, int r) { return x[feat][l] < x[feat][r]; });

      // remove previous cut
      for (size_t i = 0; i < y.size(); i++) {
        if (L[i] & (1 << k)) {
          current_score -= sum[L[i]] * sum[L[i]] / counter[L[i]];
          counter[L[i]]--;
          assert(counter[L[i]] >= 0);
          sum[L[i]] -= y[i];
          if (counter[L[i]]) current_score += sum[L[i]] * sum[L[i]] / counter[L[i]];

          L[i] -= 1 << k;

          if (counter[L[i]]) current_score -= sum[L[i]] * sum[L[i]] / counter[L[i]];
          counter[L[i]]++;
          sum[L[i]] += y[i];
          current_score += sum[L[i]] * sum[L[i]] / counter[L[i]];
        }
      }

      // iterate and update
      for (size_t i = 0; i < x_order.size(); i++) {
        current_score -= sum[L[x_order[i]]] * sum[L[x_order[i]]] / counter[L[x_order[i]]];
        counter[L[x_order[i]]]--;
        sum[L[x_order[i]]] -= y[x_order[i]];
        if (counter[L[x_order[i]]]) current_score += sum[L[x_order[i]]] * sum[L[x_order[i]]] / counter[L[x_order[i]]];

        L[x_order[i]] |= (1 << k);

        if (counter[L[x_order[i]]]) current_score -= sum[L[x_order[i]]] * sum[L[x_order[i]]] / counter[L[x_order[i]]];
        counter[L[x_order[i]]]++;
        sum[L[x_order[i]]] += y[x_order[i]];
        current_score += sum[L[x_order[i]]] * sum[L[x_order[i]]] / counter[L[x_order[i]]];

        if (current_score > best_score) {
          best_score = current_score;
          best_feat = feat;
          if (i + 1 < x_order.size()) {
            best_cut = x[feat][x_order[i]] + 0.5 * (x[feat][x_order[i + 1]] - x[feat][x_order[i]]);
          } else {
            best_cut = x[feat][x_order[i]] + 1e-5;
          }
        }
      }
    }

    for (size_t i = 0; i < y.size(); i++) {
      if (x[best_feat][i] < best_cut) {
        if (L[i] & (1 << k)) {
          current_score -= sum[L[i]] * sum[L[i]] / counter[L[i]];
          counter[L[i]]--;
          assert(counter[L[i]] >= 0);
          sum[L[i]] -= y[i];
          if (counter[L[i]]) current_score += sum[L[i]] * sum[L[i]] / counter[L[i]];

          L[i] -= 1 << k;

          if (counter[L[i]]) current_score -= sum[L[i]] * sum[L[i]] / counter[L[i]];
          counter[L[i]]++;
          sum[L[i]] += y[i];
          current_score += sum[L[i]] * sum[L[i]] / counter[L[i]];
        }
      } else {
        if (!(L[i] & (1 << k))) {
          current_score -= sum[L[i]] * sum[L[i]] / counter[L[i]];
          counter[L[i]]--;
          assert(counter[L[i]] >= 0);
          sum[L[i]] -= y[i];
          if (counter[L[i]]) current_score += sum[L[i]] * sum[L[i]] / counter[L[i]];

          L[i] |= 1 << k;

          if (counter[L[i]]) current_score -= sum[L[i]] * sum[L[i]] / counter[L[i]];
          counter[L[i]]++;
          sum[L[i]] += y[i];
          current_score += sum[L[i]] * sum[L[i]] / counter[L[i]];
        }
      }
    }

    features[k] = best_feat;
    splitting_value[k] = best_cut;
  }

  return Tree{ x, y, features, splitting_value };
}


}// namespace ogbt
