#pragma once
#include "dataset.hpp"
#include "tree.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

using std::vector;

namespace ogbt {


const size_t MIN_SAMPLE = 100;

struct ScoreTree {
  double score;
  Tree tree;

  ScoreTree(const DatasetTest &x, const vector<double> &y, std::mt19937 &t_generator, unsigned tree_depth)
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
  const vector<double> &y,
  const uint8_t tree_depth = 5,
  const unsigned iterations = 2,
  const unsigned population = 300,
  const unsigned selected = 5,
  const unsigned new_mutations = 50,
  const unsigned num_mutations = 2,
  const double subsample_a = 0.1,
  const double subsample_b = 0.1) noexcept {
  std::random_device random_dev;
  std::mt19937 generator(random_dev());

  vector<ScoreTree> pop_trees;

  unsigned sample_size_a = std::min(std::max(MIN_SAMPLE, static_cast<size_t>(y.size() * subsample_a)), y.size());
  unsigned sample_size_b = std::max(MIN_SAMPLE, static_cast<size_t>(y.size() * subsample_b));

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
  const vector<double> &y_full,
  const uint8_t tree_depth = 5,
  const double subsample_a = 0.1,
  const double subsample_b = 0.1) noexcept {

  assert(x_full.size() >= tree_depth);

  vector<ScoreFeature> feature_score;
  std::random_device random_dev;
  std::mt19937 generator(random_dev());
  unsigned sample_size_a = std::min(
    std::max(MIN_SAMPLE, static_cast<size_t>(static_cast<double>(y_full.size()) * subsample_a)), y_full.size());
  unsigned sample_size_b = std::max(MIN_SAMPLE, static_cast<size_t>(static_cast<double>(y_full.size()) * subsample_b));

  auto [x, y] = get_goss(x_full, y_full, generator, sample_size_a, sample_size_b);

  auto y_size_double = static_cast<double>(y.size());

  vector<vector<size_t>> sorted_x_order(x.size(), vector<size_t>(y.size()));
  for (size_t feat = 0; feat < x.size(); feat++) {
    const auto &x_feat = x[feat];

    for (size_t i = 0; i < sorted_x_order[feat].size(); i++) sorted_x_order[feat][i] = i;
    std::sort(sorted_x_order[feat].begin(), sorted_x_order[feat].end(), [&x_feat](int l, int r) {
      return x_feat[l] < x_feat[r];
    });
  }

  for (size_t feat = 0; feat < x.size(); feat++) {
    const auto &x_feat = x[feat];
    const auto &x_order = sorted_x_order[feat];

    double l_sum = 0;
    double r_sum = 0;
    double l_squared_sum = 0;
    double r_squared_sum = 0;
    double best_score;
    double best_value = y[x_order[0]] - 1e-5;

    for (size_t i = 0; i < x_order.size(); i++) {
      r_sum += y[x_order[i]];
      r_squared_sum += y[x_order[i]] * y[x_order[i]];
    }

    best_score = r_squared_sum / y_size_double - r_sum * r_sum / y_size_double / y_size_double;

    double l_size = 0.0;
    double r_size = y_size_double - 1;

    for (size_t i = 0; i < x_order.size(); i++) {
      l_size++;
      r_size--;
      l_sum += y[x_order[i]];
      r_sum -= y[x_order[i]];

      l_squared_sum += y[x_order[i]] * y[x_order[i]];
      r_squared_sum -= y[x_order[i]] * y[x_order[i]];


      auto score = l_squared_sum / l_size - l_sum * l_sum / l_size / l_size;

      if (y.size() - i - 1 > 0) { score += r_squared_sum / r_size - r_sum * r_sum / r_size / r_size; }

      if (score < best_score) {
        best_score = score;
        if (i + 1 < x_order.size()) {
          best_value = x_feat[x_order[i]] + 0.5 * (x_feat[x_order[i + 1]] - x_feat[x_order[i]]);
        } else {
          best_value = x_feat[x_order[i]] + 1e-5;
        }
      }
    }

    feature_score.emplace_back(best_score, best_value, feat);
  }

  std::nth_element(feature_score.begin(), feature_score.begin() + tree_depth, feature_score.end());
  vector<unsigned> features(tree_depth);
  vector<DataType> splitting_value(tree_depth);
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
  unsigned sample_size_a = std::min(
    std::max(MIN_SAMPLE, static_cast<size_t>(static_cast<double>(y_full.size()) * subsample_a)), y_full.size());
  unsigned sample_size_b = std::max(MIN_SAMPLE, static_cast<size_t>(static_cast<double>(y_full.size()) * subsample_b));

  auto [x, y] = get_goss(x_full, y_full, generator, sample_size_a, sample_size_b);

  std::vector<unsigned> L(y.size());
  std::vector<unsigned> features(tree_depth);
  std::vector<DataType> splitting_value(tree_depth, 0);

  std::vector<double> counter(1u << tree_depth);
  std::vector<double> sum(1u << tree_depth);
  double current_score;

  for (size_t i = 0; i < y.size(); i++) sum[0] += y[i];
  counter[0] = y.size();
  current_score = sum[0] * sum[0] / counter[0];


  // precompute sorting methods
  DataType n_buckets = *std::max_element(x[0].begin(), x[0].end());
  for(size_t i=1; i<x.size(); i++) n_buckets = std::max(n_buckets, *std::max_element(x[i].begin(), x[i].end()));
  n_buckets += 1;

  std::vector<std::vector<size_t>> buckets(x.size() * n_buckets);// feature * n_buckets + id_bucket

  for (size_t feat = 0; feat < x.size(); feat++) {
    for (size_t i = 0; i < y.size(); i++) {
      assert(x[feat][i] < n_buckets);
      buckets[feat * n_buckets + x[feat][i]].push_back(i);
    }
  }

  for (unsigned t = 0; t < steps; t++) {
    const int k = t % tree_depth;

    double best_score = -1e100;
    DataType best_cut = 0;
    size_t best_feat = 0;

    for (size_t feat = 0; feat < x.size(); feat++) {

      // remove previous cut
      for (size_t i = 0; i < y.size(); i++) {
        if (L[i] & (1u << k)) {
          current_score -= sum[L[i]] * sum[L[i]] / counter[L[i]];
          counter[L[i]]--;
          assert(counter[L[i]] >= -1e-5);
          sum[L[i]] -= y[i];
          if (counter[L[i]] >= 1e-5) current_score += sum[L[i]] * sum[L[i]] / counter[L[i]];

          L[i] -= 1u << k;

          if (counter[L[i]] >= 1e-5) current_score -= sum[L[i]] * sum[L[i]] / counter[L[i]];
          counter[L[i]]++;
          sum[L[i]] += y[i];
          current_score += sum[L[i]] * sum[L[i]] / counter[L[i]];
        }
      }

      // iterate and update
      for (DataType id_bucket = 0; id_bucket < n_buckets; id_bucket++) {
        for (auto index : buckets[feat * n_buckets + id_bucket]) {
          current_score -= sum[L[index]] * sum[L[index]] / counter[L[index]];
          counter[L[index]]--;
          sum[L[index]] -= y[index];
          if (counter[L[index]] >= 1e-5) current_score += sum[L[index]] * sum[L[index]] / counter[L[index]];

          L[index] |= (1u << k);

          if (counter[L[index]] >= 1e-5) current_score -= sum[L[index]] * sum[L[index]] / counter[L[index]];
          counter[L[index]]++;
          sum[L[index]] += y[index];
          current_score += sum[L[index]] * sum[L[index]] / counter[L[index]];
        }

        if (current_score > best_score) {
          best_score = current_score;
          best_feat = feat;
          best_cut = id_bucket + 1;
        }
      }
    }

    for (size_t i = 0; i < y.size(); i++) {
      if (x[best_feat][i] < best_cut) {
        if (L[i] & (1u << k)) {
          current_score -= sum[L[i]] * sum[L[i]] / counter[L[i]];
          counter[L[i]]--;
          assert(counter[L[i]] >= -1e-5);
          sum[L[i]] -= y[i];
          if (counter[L[i]] >= 1e-5) current_score += sum[L[i]] * sum[L[i]] / counter[L[i]];

          L[i] -= 1u << k;

          if (counter[L[i]] >= 1e-5) current_score -= sum[L[i]] * sum[L[i]] / counter[L[i]];
          counter[L[i]]++;
          sum[L[i]] += y[i];
          current_score += sum[L[i]] * sum[L[i]] / counter[L[i]];
        }
      } else {
        if (!(L[i] & (1u << k))) {
          current_score -= sum[L[i]] * sum[L[i]] / counter[L[i]];
          counter[L[i]]--;
          assert(counter[L[i]] >= 0);
          sum[L[i]] -= y[i];
          if (counter[L[i]] >= 1e-5) current_score += sum[L[i]] * sum[L[i]] / counter[L[i]];

          L[i] |= 1u << k;

          if (counter[L[i]] >= 1e-5) current_score -= sum[L[i]] * sum[L[i]] / counter[L[i]];
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
