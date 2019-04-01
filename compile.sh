#clang++ -O3 -Wall -shared -std=c++14 -fPIC `python3 -m pybind11 --includes` mylib.cpp -I ~/.pyenv/versions/3.7.2/include/python3.7m -o mylibs`python3-config --extension-suffix`

#clang++ -O3 -Wall -shared -std=c++14 -fPIC `python3 -m pybind11 --includes` mylib.cpp -I ~/.pyenv/versions/3.7.2/include/python3.7m -o mylibs`python3-config --extension-suffix`

clang++ -mavx2 -mfma -O3 -Wall -shared -std=c++14 -fPIC `python -m pybind11 --includes` -I ~/.pyenv/versions/3.7.2/lib/python3.7/site-packages/numpy/core/include/ -undefined dynamic_lookup mylib.cpp -o mylib`python3-config --extension-suffix`



#clang -Xpreprocessor -fopenmp -lomp