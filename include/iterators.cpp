
    public:
      class Iterator {
        public:
          Iterator(Matrix& matrix, const size_t index) : matrix(matrix), index(index) {}
          bool operator==(const Iterator& other) const { return index == other.index; }
          bool operator!=(const Iterator& other) const { return !(*this == other); }
          Iterator& operator++() { ++index; return *this; }
          Value operator*() const { return matrix[0][index]; }
        private:
          Matrix& matrix;
          size_t index;
      };
      class const_Iterator {
        public:
          const_Iterator(const Matrix& matrix, const size_t index) : matrix(matrix), index(index) {}
          bool operator==(const const_Iterator& other) const { return index == other.index; }
          bool operator!=(const const_Iterator& other) const { return !(*this == other); }
          const_Iterator& operator++() { ++index; return *this; }
          const Value operator*() const { return matrix[0][index]; }
        private:
          const Matrix& matrix;
          size_t index;
      };
      Iterator begin() { return Iterator(*this, 0); }
      Iterator end()   { return Iterator(*this, size.numel); }
      const_Iterator begin() const { return const_Iterator(*this, 0); }
      const_Iterator end() const { return const_Iterator(*this, size.numel); }