#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "include/someFunctions.h"

namespace p = boost::python;
namespace np = boost::python::numpy;


void helloWorldFunction(np::ndarray const &xarr) {
  checkIfDouble(xarr); // checks if python array is doubles.

  printArr(xarr, "xarr_original"); // function for printing the array info:
  double* x_arr_data = reinterpret_cast<double*>(xarr.get_data()); //
  x_arr_data[0] = 10.0;
  printArr(xarr, "xarr_new");

}


BOOST_PYTHON_MODULE(subchannel) {  // parenthesis match filename --> helloWorld is the module name / filename
    Py_Initialize();        // initialize py
    np::initialize();  // initialize np
    p::def("hello_world", helloWorldFunction); // make function available in python - module name is subchannel and we can call function subchannel.hello_world to run helloWorldFunction
}
