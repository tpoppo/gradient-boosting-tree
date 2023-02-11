// g++ -std=c++17 -DNDEBUG -Ofast -Wall -Wextra -Werror -Wshadow -Wfloat-equal -Wpedantic -Wformat=2 -march=native -g
// examples/benchmark.cpp -o main && ./main
#include "../src/dataset.hpp"
#include "../src/loss.hpp"
#include "../src/model.hpp"
#include "../src/tree_algo.hpp"
#include "../src/utils.hpp"
#include <chrono>
#include <random>

const uint16_t MAX_TREE = 300;
const double LEARNING_RATE = 0.05;


// from https://stackoverflow.com/questions/2808398/easily-measure-elapsed-time
template<class result_t = std::chrono::milliseconds,
  class clock_t = std::chrono::steady_clock,
  class duration_t = std::chrono::milliseconds>
auto since(std::chrono::time_point<clock_t, duration_t> const &start) {
  return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}


std::pair<ogbt::Dataset, ogbt::Dataset> get_dataset() {
  std::mt19937 rng{ 42 };
  return std::make_pair(ogbt::get_dummy_data(1000, 5, rng), ogbt::get_dummy_data(5000, 5, rng));
}

template<typename F>
void evaluate_algo(std::string name,
  uint16_t num_trees,
  double learning_rate,
  const ogbt::Dataset &dataset,
  const ogbt::Dataset &dataset_validation,
  F tree_generator) {
  ogbt::Model<ogbt::MSE> model(tree_generator, num_trees, learning_rate);
  auto start = std::chrono::steady_clock::now();
  model.fit(dataset);

  auto y_pred_val = model.predict(dataset_validation.get_x());
  auto y_pred = model.predict(dataset.get_x());

  auto score_val = ogbt::MSE::score(y_pred_val, dataset_validation.get_y());
  auto score = ogbt::MSE::score(y_pred, dataset.get_y());

  std::cout << "[" << name << "] score: " << score_val << " (" << score << ") time(ms): " << since(start).count()
            << std::endl;
}

int main() {
  auto [dataset, dataset_validation] = get_dataset();

  evaluate_algo("mse_splitting_bdt",
    MAX_TREE,
    LEARNING_RATE,
    dataset,
    dataset_validation,
    [](const ogbt::DatasetTest &x, const std::vector<double> &y) {
      const unsigned tree_depth = 6;
      const int steps = 20;
      const double subsample_a = 0.5;
      const double subsample_b = 0.5;
      return ogbt::mse_splitting_bdt(x, y, tree_depth, steps, subsample_a, subsample_b);
    });

  evaluate_algo("mse_splitting_bdt2",
    MAX_TREE,
    LEARNING_RATE,
    dataset,
    dataset_validation,
    [](const ogbt::DatasetTest &x, const std::vector<double> &y) {
      const unsigned tree_depth = 6;
      const int steps = 20;
      const double subsample_a = 0.5;
      const double subsample_b = 0.5;
      return ogbt::mse_splitting_bdtv2(x, y, tree_depth, steps, subsample_a, subsample_b);
    });


  std::mt19937 rng_random_tree{ 42 };
  evaluate_algo("random_tree",
    MAX_TREE,
    LEARNING_RATE,
    dataset,
    dataset_validation,
    [&rng_random_tree](const ogbt::DatasetTest &x, const std::vector<double> &y) {
      return ogbt::Tree{ x, y, rng_random_tree, 6 };
    });

  evaluate_algo("genetic_algo",
    MAX_TREE,
    LEARNING_RATE,
    dataset,
    dataset_validation,
    [](const ogbt::DatasetTest &x, const std::vector<double> &y) {
      const unsigned iterations = 6;
      const unsigned tree_depth = 6;
      const unsigned population = 200;
      const unsigned selected = 2;
      const unsigned new_mutations = 80;
      const unsigned num_mutations = 2;
      const double subsample_a = 0.5;
      const double subsample_b = 0.5;
      return ogbt::genetic_algo<ogbt::MSE>(
        x, y, tree_depth, iterations, population, selected, new_mutations, num_mutations, subsample_a, subsample_b);
    });

  evaluate_algo("greedy_mse_splitting",
    MAX_TREE,
    LEARNING_RATE,
    dataset,
    dataset_validation,
    [](const ogbt::DatasetTest &x, const std::vector<double> &y) {
      const unsigned tree_depth = 6;
      const double subsample_a = 0.5;
      const double subsample_b = 0.5;
      return ogbt::greedy_mse_splitting(x, y, tree_depth, subsample_a, subsample_b);
    });
}