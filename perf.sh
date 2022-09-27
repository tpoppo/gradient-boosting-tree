echo "\n\n--------------------------------- STEP 1 ---------------------------------" &&
cmake -S . -B build -DRELEASE=ON &&
echo "\n\n--------------------------------- STEP 2 ---------------------------------" &&
cmake --build build &&
echo "\n\n--------------------------------- STEP 3 ---------------------------------" &&
cd build &&
sudo perf record -F 99 -a -g ./tests &&
sudo perf report > profiling-results.txt && 
echo "Created build/profiling-results.txt"


