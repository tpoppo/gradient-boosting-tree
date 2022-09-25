#pragma once
#include "dataset.hpp"
#include "model.hpp"
#include <vector>

namespace ogbt {
class Loss {
public:
  Loss();

  static double score(const std::vector<double> &predicted_target, const Dataset &data) { return 0.0; }

  static double score(const Model<Loss> &model, const Dataset &data) {
    return score(model.predict(data.get_data()), data);
  }

  static std::vector<double> residual(const std::vector<double> &predicted_target, const Dataset &data) {
    auto copy = predicted_target;
    return copy;
  }

  static std::vector<double> residual(const Model<Loss> &model, Dataset data) {
    return residual(model.predict(data.get_data()), data);
  }
};


class MSE : public Loss {
public:
  static double score(const std::vector<double> &predicted_target, const Dataset &data) {
    const auto &target = data.get_target();
    double result = 0;
    for (size_t i = 0; i < predicted_target.size(); i++)
      result += (target[i] - predicted_target[i]) * (target[i] - predicted_target[i]);
    return result / target.size();
  }

  static std::vector<double> residual(const std::vector<double> &predicted_target, const Dataset &data) {
    const auto &target = data.get_target();
    std::vector<double> ans;
    ans.resize(target.size());
    for (size_t i = 0; i < target.size(); i++) { ans[i] = predicted_target[i] - target[i]; }
    return move(ans);
  }
};

}// namespace ogbt