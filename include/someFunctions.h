#include <iostream>
#include <vector>
#include <string>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

// some bonus functions to use
void printArr(np::ndarray const &arr, std::string const &array_name) {
  std::cout << array_name << ": " << p::extract<char const *>(p::str(arr)) << std::endl;
}

void checkIfDouble(np::ndarray const &arr){
  // checks if double, otherwise throws error
  if (arr.get_dtype() != np::dtype::get_builtin<double>()) {
    throw std::invalid_argument("Array must have dtype double. Is the python array a double?");
  }
}
