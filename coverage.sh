sh test.sh &&
lcov --directory ./build/ --capture --output-file ./code_coverage.info -rc lcov_branch_coverage=1 &&
genhtml code_coverage.info --output-directory out