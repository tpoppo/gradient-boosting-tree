#include "../src/utils.hpp"
#include <gtest/gtest.h>


TEST(Utils, Subset) {
  const int n = 100;
  const int m = 5;
  const int a = 10;
  std::mt19937 gen{ 42 };

  ogbt::Dataset dataset = ogbt::get_dummy_data(n, m, gen);

  auto [x, y] = ogbt::get_subsample(dataset.get_x(), dataset.get_y(), gen, a);

  EXPECT_EQ(x.size(), m);
  EXPECT_EQ(y.size(), a);
  for (size_t i = 0; i < x.size(); i++) EXPECT_EQ(x[i].size(), a);
}


TEST(Utils, GOSSSubset) {
  const int n = 100;
  const int m = 5;
  const int a = 10;
  const int b = 50;
  std::mt19937 gen{ 42 };

  ogbt::Dataset dataset = ogbt::get_dummy_data(n, m, gen);

  auto [x, y] = ogbt::get_goss(dataset.get_x(), dataset.get_y(), gen, a, b);

  EXPECT_EQ(x.size(), m);
  EXPECT_EQ(y.size(), a + b);
  for (size_t i = 0; i < x.size(); i++) EXPECT_EQ(x[i].size(), a + b);
}
