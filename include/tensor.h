#pragma once

#include <vector>

using std::vector;

/* [Tensor Structure] */
struct Tensor {
  size_t ndim = 0;
  size_t shape[4];
  float *buf = nullptr;
  int device = -1; // -1 for CPU, 0~3 for GPU

  Tensor(const vector<size_t> &shape, int device=-1);
  Tensor(const vector<size_t> &shape_, float *buf_, int device=-1); 
  ~Tensor();

  size_t num_elem();
  Tensor* cpu();
};

typedef Tensor Parameter;
typedef Tensor Activation;
