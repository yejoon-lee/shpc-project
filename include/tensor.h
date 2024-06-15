#pragma once

#include <vector>

using std::vector;

/* [Tensor Structure] */
struct Tensor {
  size_t ndim = 0;
  size_t shape[5];
  float *buf = nullptr;
  int device; // -1 for CPU, 0~3 for GPU

  Tensor(const vector<size_t> &shape, int device=0);
  Tensor(const vector<size_t> &shape_, float *buf_, int device=0); 
  ~Tensor();

  size_t num_elem();
  
  Tensor* cpu();
  Tensor* set_device(int device);
};

typedef Tensor Parameter;
typedef Tensor Activation;
