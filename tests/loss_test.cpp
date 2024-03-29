#include "../src/loss.hpp"
#include <gtest/gtest.h>

TEST(MSE, score) {
  std::vector<std::vector<int>> data_categorical = { { 0, 1 }, { 1, 1 } };
  std::vector<std::vector<float>> data_dense = { { 1.2, -10.0 } };
  std::vector<float> target = { 0.23, 1.0 };

  ogbt::Dataset dataset(data_categorical, data_dense, target);

  EXPECT_TRUE(abs(ogbt::MSE::score(target, dataset)) <= 1e-7);

  auto residual = ogbt::MSE::residual(target, dataset);
  for (auto r : residual) { EXPECT_TRUE(abs(r) <= 1e-7); }
}


TEST(LogLoss, score) {
  std::vector<std::vector<int>> data_categorical = { { 0, 1 }, { 1, 1 } };
  std::vector<std::vector<float>> data_dense = { { 1.2, -10.0 } };
  std::vector<float> target = { 0.0, 0.0 };

  ogbt::Dataset dataset(data_categorical, data_dense, target);
  std::cout << ogbt::LogLoss::score(target, dataset) << std::endl;
  EXPECT_TRUE(ogbt::LogLoss::score(target, dataset) <= 0);

  auto residual = ogbt::LogLoss::residual(target, dataset);
  for (auto r : residual) { EXPECT_TRUE(r <= 0); }
}
