#include "../src/dataset.hpp"
#include "../src/loss.hpp"
#include "../src/model.hpp"
#include "../src/tree.hpp"
#include "../src/tree_algo.hpp"

#include <vector>

#include <gtest/gtest.h>


TEST(Model, Constructor) {
  constexpr auto gen_random = [](const ogbt::DatasetTest &x, const std::vector<double> &y) {
    std::mt19937 rng{ 42 };
    return ogbt::Tree{ x, y, rng };
  };

  ogbt::Model<ogbt::MSE> model(gen_random);

  constexpr auto genetic_algo = [](const ogbt::DatasetTest &x, const std::vector<double> &y) {
    return ogbt::genetic_algo<ogbt::MSE>(x, y);
  };

  ogbt::Model<ogbt::MSE> model2(genetic_algo);

}

TEST(Model, Fit) {
  constexpr auto genetic_algo = [](const ogbt::DatasetTest &x, const std::vector<double> &y) {
    return ogbt::genetic_algo<ogbt::MSE>(x, y);
  };

  ogbt::Model<ogbt::MSE> model(genetic_algo);

  std::vector<std::vector<double>> data_dense = { { 1.2, -10.0, 0.22 }, { 1.0, 2.0, 3.0 } };
  std::vector<double> target = { 0.23, 1.0, -1.0 };
  ogbt::Dataset dataset(data_dense, target);
  
  model.fit(dataset);
}

TEST(Model, FitPredict) {
  constexpr auto genetic_algo = [](const ogbt::DatasetTest &x, const std::vector<double> &y) {
    return ogbt::genetic_algo<ogbt::MSE>(x, y);
  };

  ogbt::Model<ogbt::MSE> model(genetic_algo);

  std::vector<std::vector<double>> data_dense = { { 1.2, -10.0, 0.22 }, { 1.0, 2.0, 3.0 } };
  std::vector<double> target = { 0.23, 1.0, -1.0 };
  ogbt::Dataset dataset(data_dense, target);
  
  model.fit(dataset);

  model.predict(data_dense);
}



TEST(Model, GeneticAlgo) {
  std::vector<std::vector<double>> data_dense = { { 1.2, -10.0, 0.22 }, { 1.0, 2.0, 3.0 } };
  std::vector<double> target = { 0.23, 1.0, -1.0 };

  ogbt::Dataset dataset(data_dense, target);
  unsigned iterations = 2;
  unsigned tree_depth = 4;
  unsigned population = 70;
  unsigned selected = 5;
  unsigned new_mutations = 20;
  unsigned num_mutations = 2;

  ogbt::Tree tree = ogbt::genetic_algo<ogbt::MSE>(
    dataset.get_x(), dataset.get_y(), tree_depth, iterations, population, selected, new_mutations, num_mutations);
}
