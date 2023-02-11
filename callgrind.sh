echo "\n\n--------------------------------- STEP 1 ---------------------------------" &&
cmake -S . -B build -DRELEASE=ON &&
echo "\n\n--------------------------------- STEP 2 ---------------------------------" &&
cmake --build build &&
echo "\n\n--------------------------------- STEP 3 ---------------------------------" &&
cd build &&
valgrind --tool=callgrind ./tests
# kcachegrind calgrind.out.*





