#ifndef matrix_hpp
#define matrix_hpp
#include <stdlib.h>
#define CLEARSCREEN "\033[2J\033[1;1H"

typedef struct {unsigned int x, y, numel;} _size;
typedef struct {unsigned int idx; double val;} _idxval;

class matrix {
    public:
        matrix(void);
        matrix(unsigned int xsz, unsigned int ysz);
        matrix(_size sz);
        matrix(unsigned int sz);

        matrix  show();

        matrix  operator ^  (double val);               // exponentiation
        matrix  operator ,  (matrix arr);               // join matrix column-wise
        matrix  operator ,  (double val);               // append value
        matrix  operator !  (void);                     // transpose
        matrix  operator |  (matrix  arr);              // solve linear system
        matrix  operator &  (matrix  arr);              // matrix multiplication
        matrix  operator =  (matrix  arr);              // matrix copy
        matrix  operator +  (matrix  arr);              // add matrix
        matrix  operator +  (double val);               // add constant
        matrix  operator += (matrix  arr);              // add&assign matrix
        matrix  operator += (double val);               // add&assign constant
        matrix  operator -  (void);                     // negate matrix
        matrix  operator -  (matrix  arr);              // sub matrix
        matrix  operator -  (double val);               // sub constant
        matrix  operator -= (matrix  arr);              // sub&assign matrix
        matrix  operator -= (double val);               // sub&assign constant
        matrix  operator *  (matrix  arr);              // mul matrix
        matrix  operator *  (double val);               // mul constant
        matrix  operator *= (matrix  arr);              // mul&assign matrix
        matrix  operator *= (double val);               // mul&assign constant
        matrix  operator /  (matrix  arr);              // div matrix
        matrix  operator /  (double val);               // div constant
        matrix  operator /= (matrix  arr);              // div&assign matrix
        matrix  operator /= (double val);               // div&assign constant

        double& operator () (unsigned int x_idx, unsigned int y_idx);
        double& operator () (unsigned int idx);
        double& operator [] (unsigned int idx);         // index into memory directly

        unsigned int    numel(void) {return size.numel;};
        unsigned int    sizex(void) {return size.x;};
        unsigned int    sizey(void) {return size.y;};

        _idxval min(void);

    private:
        _size   size;
        double* data;
        int     resize(unsigned int xsz, unsigned int ysz = 1);
        int     resize(_size size);
};

class matrix_array {
    public:
        matrix_array(int n) {
            n_matrix = n;
            data = (matrix*) calloc(n_matrix, sizeof(matrix));
        };
        matrix& operator [] (unsigned int idx) {
            return data[idx];
        };
    private:
        matrix* data;
        int     n_matrix;
};












#include "matrix.cpp"
#endif