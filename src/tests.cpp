#include <complex>
#include <iostream>

struct integer {
    integer(const int val) : data_(val) {}
    integer(const integer_add_expression_template& to_be_evaluated);
    int data_;
};

class integer_add_expression_template {
public:
    const integer& A_;
    const integer& B_;
    integer_add_expression_template(const integer& A, const integer& B) : A_(A), B_(B) {}
    integer_add_expression_template(const integer_add_expression_template&) = delete;
    integer_add_expression_template(integer_add_expression_template&&) = delete;

    operator integer() const {
        return integer(A_.data_ + B_.data_);
    }
};



integer::integer(const integer_add_expression_template& to_be_evaluated)
{
    data_ = to_be_evaluated.A_.data_ + to_be_evaluated.B_.data_;
}



integer_add_expression_template operator+(const integer& A, const integer& B) {
    return integer_add_expression_template(A, B);
}

int main() {
    integer a = 1;
    integer b = 2;

    auto c = a + b;

    return 0;
}
