#include<vector>
#include<unordered_map>
#include <numeric>

using DatasetTest = std::vector<std::vector<double>>;

class Dataset {
  private:
    std::vector<std::vector<double>> data;
    std::vector<double> target;

    std::vector<std::unordered_map<int, std::pair<double, int>>> target_encoding_counter;
    int smooth_target_encoding = 0;
    double mean_target = 0;
  
  public:
    Dataset(const std::vector<std::vector<int>>& data_categorical, 
      const std::vector<std::vector<double>>& data_dense, 
      const std::vector<double>& data_target,
      const int smooth_target_encoding_param=100) {

        this->data = data_dense;
        this->target = data_target;
        this->data.reserve(data_dense.size()+data_categorical.size());
        this->smooth_target_encoding = smooth_target_encoding_param;
        
        // target encoding
        mean_target = std::accumulate(target.begin(), target.end(), 0.0) / target.size() * smooth_target_encoding;
        target_encoding_counter.reserve(data_categorical.size());
        for(const auto& column : data_categorical){
          this->data.emplace_back();

          target_encoding_counter.emplace_back();
          std::unordered_map<int, std::pair<double, int>>& counter = target_encoding_counter.back();

          counter.max_load_factor(0.15);
          for(size_t i=0; i<column.size(); i++){
            counter[column[i]].first += target[i];
            counter[column[i]].second += 1;
          }

          for(const int val : column){
            auto target_encoding = counter[val];
            this->data.back().emplace_back((target_encoding.first + mean_target) / (target_encoding.second + smooth_target_encoding));
          }
        }
    }

    const auto& get_data(){
      return this->data;
    }

    const auto& get_target(){
      return this->target;
    }

    const auto process_test(const std::vector<std::vector<int>>& data_categorical, 
      const std::vector<std::vector<double>>& data_dense){
        
        std::vector<std::vector<double>> data_result = data_dense;
        
        data_result.reserve(data_categorical.size()+data_dense.size());
        for(size_t i=0; i<data_categorical.size(); i++){
          data_result.emplace_back();
          data_result.back().reserve(data_categorical[i].size());
        
          for(const int val : data_categorical[i]) {
            auto target_encoding = target_encoding_counter[i][val];
            data_result.back().emplace_back((target_encoding.first + mean_target) / (target_encoding.second + smooth_target_encoding));
          }
        }
        
        return data_result;
        
    }

};