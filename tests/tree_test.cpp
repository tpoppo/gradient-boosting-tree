#include "../src/tree.hpp"
#include <gtest/gtest.h>

TEST(Tree, Constructor) {
  std::vector<std::vector<int>> data_categorical = { { 0, 1 }, { 1, 1 } };
  std::vector<std::vector<float>> data_dense = { { 1.2, -10.0 } };
  std::vector<float> target = { 1.0, 0.0 };
  ogbt::Dataset dataset(data_categorical, data_dense, target);
  std::mt19937 gen;
  ogbt::Tree tree(dataset, gen, 6);

  EXPECT_EQ(tree.get_splitting_value().size(), 6);
  
  EXPECT_EQ(tree.get_features().size(), 6);
  for(size_t i=0; i<tree.get_features().size(); i++){
    EXPECT_GE(tree.get_features()[i], 0);
    EXPECT_LT(tree.get_features()[i], 3);
  }
}


TEST(Tree, Predict) {
  std::vector<std::vector<int>> data_categorical = { { 0, 1 }, { 1, 1 } };
  std::vector<std::vector<float>> data_dense = { { 1.2, -10.0 } };
  std::vector<float> target = { 1.0, 0.0 };
  ogbt::Dataset dataset(data_categorical, data_dense, target);
  std::mt19937 gen;
  ogbt::Tree tree(dataset, gen);
  tree.predict(dataset.get_x());
}