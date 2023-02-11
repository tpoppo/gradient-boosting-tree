n = 500000
m = 80
[ RUN      ] Model.Constructor
[       OK ] Model.Constructor (0 ms)
[ RUN      ] Model.Fit
[       OK ] Model.Fit (79 ms)
[ RUN      ] Model.FitPredict
[       OK ] Model.FitPredict (57 ms)
[ RUN      ] Model.GeneticAlgo
[       OK ] Model.GeneticAlgo (0 ms)
[ RUN      ] Model.AlgoGenGaussianTest
[       OK ] Model.AlgoGenGaussianTest (135012 ms)
[ RUN      ] Model.MSEGreedyGaussianTest
[  FAILED  ] Model.MSEGreedyGaussianTest (85489 ms)
[ RUN      ] Model.MSEBDTGaussianTest
[  FAILED  ] Model.MSEBDTGaussianTest (286893 ms)
[----------] 7 tests from Model (507533 ms total)