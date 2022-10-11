#pragma once
#include "dataset.hpp"
#include "model.hpp"
#include <cassert>
#include <vector>

namespace ogbt {

template<typename Score, typename Residual> class Loss {
public:
  static double score(const std::vector<double> &y_pred, const std::vector<double> &y_true) noexcept {
    return Score::evaluate(y_pred, y_true);
  }

  static double score(const std::vector<double> &y_pred, const Dataset &data) noexcept {
    return score(y_pred, data.get_y());
  }

  template<typename TLoss, typename TreeGenAlgo>
  static double score(const Model<TLoss> &model, const Dataset &data) noexcept {
    return score(model.predict(data.get_x()), data.get_y());
  }

  static double score(const Tree &tree, const Dataset &data) noexcept {
    return score(tree.predict(data.get_x()), data);
  }

  static std::vector<double> residual(const std::vector<double> &y_pred, const std::vector<double> &y_true) noexcept {
    return Residual::evaluate(y_pred, y_true);
  }

  static std::vector<double> residual(const std::vector<double> &y_pred, const Dataset &data) noexcept {
    return residual(y_pred, data.get_y());
  }

  template<typename TLoss, typename TreeGenAlgo>
  static std::vector<double> residual(const Model<TLoss> &model, Dataset data) noexcept {
    return residual(model.predict(data.get_x()), data);
  }

  static std::vector<double> residual(const Tree &tree, Dataset data) noexcept {
    return residual(tree.predict(data.get_x()), data);
  }
};


struct ScoreMSE {
  static double evaluate(const std::vector<double> &y_pred, const std::vector<double> &y_true) noexcept {
    assert(y_pred.size() == y_true.size());
    double result = 0;
    for (size_t i = 0; i < y_pred.size(); i++) { result += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]); }
    return result / y_true.size();
  }
};

struct ResidualMSE {
  static std::vector<double> evaluate(const std::vector<double> &y_pred, const std::vector<double> &y_true) noexcept {
    std::vector<double> ans;
    ans.resize(y_true.size());
    for (size_t i = 0; i < y_true.size(); i++) { ans[i] = y_true[i] - y_pred[i]; }
    return ans;
  }
};

struct ScoreLogLoss {
  static double evaluate(const std::vector<double> &y_pred, const std::vector<double> &y_true) noexcept {
    assert(y_pred.size() == y_true.size());
    double result = 0;
    for (size_t i = 0; i < y_pred.size(); i++) {
      auto sigmoid_val = 1.0 / (1.0 + exp(-y_pred[i]));
      result += y_true[i] * log(sigmoid_val) + (1 - y_true[i]) * log(1 - sigmoid_val);
    }
    return result / y_true.size();
  }
};

struct ResidualLogLoss {
  static std::vector<double> evaluate(const std::vector<double> &y_pred, const std::vector<double> &y_true) noexcept {
    std::vector<double> ans;
    ans.resize(y_true.size());
    for (size_t i = 0; i < y_true.size(); i++) {
      auto sigmoid_val = 1.0 / (1.0 + exp(-y_pred[i]));
      ans[i] = sigmoid_val;

      if (y_true[i] < 0.5) { ans[i] = -ans[i]; }
    }
    return ans;
  }
};

using MSE = Loss<ScoreMSE, ResidualMSE>;
using LogLoss = Loss<ScoreLogLoss, ResidualLogLoss>;
}// namespace ogbt