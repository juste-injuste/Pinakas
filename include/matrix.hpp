#ifndef matrix_hpp
#define matrix_hpp
#include <stdlib.h>
#define CLEARSCREEN "\033[2J\033[1;1H"


class matrix {
    public:
        matrix  operator ,  (double val);               // append value

        matrix  operator |  (matrix  arr);              // solve linear system
        
        matrix  operator += (matrix  arr);              // add&assign matrix
        matrix  operator += (double val);               // add&assign constant
        matrix  operator -= (matrix  arr);              // sub&assign matrix
        matrix  operator -= (double val);               // sub&assign constant
        matrix  operator *= (matrix  arr);              // mul&assign matrix
        matrix  operator *= (double val);               // mul&assign constant
        matrix  operator /= (matrix  arr);              // div&assign matrix
        matrix  operator /= (double val);               // div&assign constan
};











#include "matrix.cpp"
#endif