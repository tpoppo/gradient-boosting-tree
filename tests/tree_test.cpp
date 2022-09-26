#include "../src/tree.hpp"
#include <gtest/gtest.h>

TEST(Tree, Constructor) {
  std::vector<std::vector<int>> data_categorical = { { 0, 1 }, { 1, 1 } };
  std::vector<std::vector<double>> data_dense = { { 1.2, -10.0 } };
  std::vector<double> target = { 1.0, 0.0 };
  ogbt::Dataset dataset(data_categorical, data_dense, target);
  std::mt19937 gen;
  ogbt::Tree tree(dataset, gen);
}


TEST(Tree, Predict) {
  std::vector<std::vector<int>> data_categorical = { { 0, 1 }, { 1, 1 } };
  std::vector<std::vector<double>> data_dense = { { 1.2, -10.0 } };
  std::vector<double> target = { 1.0, 0.0 };
  ogbt::Dataset dataset(data_categorical, data_dense, target);
  std::mt19937 gen;
  ogbt::Tree tree(dataset, gen);
  tree.predict(dataset.get_x());
}