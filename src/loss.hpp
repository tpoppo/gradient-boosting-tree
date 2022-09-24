#include<vector>
#include "data.hpp"
#include "model.hpp"


class Loss {
  public virtual double score(Dataset data);
  public virtual std::vector<double> residual(Model model, Dataset data);

};


class MSE : public Loss {
  
  double score(Dataset)

}