#include<type_traits>
#include "dataset.hpp"

template <class loss>
class Tree {
  static_assert(std::is_base_of<Loss, loss>::value, "The loss must derive from Loss");

  Tree(Dataset data){
    
  }

  auto predict(Dataset data){
    
  }


};