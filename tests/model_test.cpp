#include "../src/loss.hpp"
#include "../src/model.hpp"
#include "../src/tree_algo.hpp"

#include <gtest/gtest.h>

TEST(Model, Constructor) {}


TEST(Model, genetic_algo) {
  std::vector<std::vector<double>> data_dense = { { 1.2, -10.0, 0.22 }, { 1.0, 2.0, 3.0 } };
  std::vector<double> target = { 0.23, 1.0, -1.0 };

  ogbt::Dataset dataset(data_dense, target);
  unsigned iterations = 2;
  unsigned population = 100;
  unsigned selected = 10;
  unsigned new_mutations = 50;
  unsigned num_mutations = 2;
  float subsampling = 0.5;

  ogbt::Tree tree =
    ogbt::genetic_algo<ogbt::MSE>(dataset, iterations, population, selected, new_mutations, num_mutations, subsampling);
}