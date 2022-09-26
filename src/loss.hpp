#pragma once
#include "dataset.hpp"
#include "model.hpp"
#include <vector>

namespace ogbt {

template<typename Score, typename Residual> class Loss {
public:
  static double score(const std::vector<double> &y_pred, const std::vector<double> &y_true) {
    return Score::evaluate(y_pred, y_true);
  }

  static double score(const std::vector<double> &y_pred, const Dataset &data) { return score(y_pred, data.get_y()); }

  template<class loss> static double score(const Model<loss> &model, const Dataset &data) {
    return score(model.predict(data.get_x()), data.get_y());
  }

  static double score(const Tree &tree, const Dataset &data) { return score(tree.predict(data.get_x()), data); }

  static std::vector<double> residual(const std::vector<double> &y_pred, const std::vector<double> &y_true) {
    return Residual::evaluate(y_pred, y_true);
  }

  static std::vector<double> residual(const std::vector<double> &y_pred, const Dataset &data) {
    return residual(y_pred, data.get_y());
  }

  template<class loss> static std::vector<double> residual(const Model<loss> &model, Dataset data) {
    return residual(model.predict(data.get_x()), data);
  }

  static std::vector<double> residual(const Tree &tree, Dataset data) {
    return residual(tree.predict(data.get_x()), data);
  }
};


struct ScoreMSE {
  static double evaluate(const std::vector<double> &y_pred, const std::vector<double> &y_true) {
    double result = 0;
    for (size_t i = 0; i < y_pred.size(); i++) result += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    return result / y_true.size();
  }
};

struct ResidualMSE {
  static std::vector<double> evaluate(const std::vector<double> &y_pred, const std::vector<double> &y_true) {
    std::vector<double> ans;
    ans.resize(y_true.size());
    for (size_t i = 0; i < y_true.size(); i++) { ans[i] = y_pred[i] - y_true[i]; }
    return ans;
  }
};


using MSE = Loss<ScoreMSE, ResidualMSE>;

}// namespace ogbt