#include "../src/dataset.hpp"
#include <gtest/gtest.h>

TEST(DatasetTest, Constructor) {
  std::vector<std::vector<int>> data_categorical = { { 0, 1 }, { 1, 1 } };
  std::vector<std::vector<double>> data_dense = { { 1.2, -10.0 } };
  std::vector<double> target = { 0.23, 1.0 };

  ogbt::Dataset dataset(data_categorical, data_dense, target);

  // Test data
  auto data = dataset.get_x();

  EXPECT_EQ(data.size(), 3);
  for (auto &row : data) { EXPECT_EQ(row.size(), 2); }

  // Test target
  auto dataset_target = dataset.get_y();
  EXPECT_EQ(dataset_target, target);
}

TEST(DatasetTest, PreprocessTest) {
  std::vector<std::vector<int>> data_categorical = { { 0, 1 }, { 1, 1 } };
  std::vector<std::vector<double>> data_dense = { { 1.2, -10.0 } };
  std::vector<double> target = { 1.0, 1.0 };
  std::vector<std::vector<int>> data_categorical_test = data_categorical;
  std::vector<std::vector<double>> data_dense_test = data_dense;
  std::vector<double> target_test = target;

  ogbt::Dataset dataset(data_categorical, data_dense, target);
  EXPECT_EQ(data_dense, data_dense_test);
  EXPECT_EQ(data_categorical, data_categorical);

  auto data = dataset.get_x();

  auto dataset_test = dataset.process_test(data_categorical_test, data_dense_test);
  EXPECT_EQ(dataset_test, data);
}