#include "model.h"

/* [Tensor Structure] */
/* Tensor
 * @brief - A multi-dimensional matrix containing elements of a single data
 type.
 * @member - buf  : Data buffer containing elements
 * @member - shape: Shape of tensor from outermost dimension to innermost
 dimension e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
 */
Tensor::Tensor(const vector<size_t> &shape_, int device) : device(device) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();

  if (device >= 0) {
    cudaSetDevice(device);
    cudaMalloc(&buf, N_ * sizeof(float));
    cudaMemset(buf, 0, N_ * sizeof(float));
  } else {
    buf = (float *) calloc(N_, sizeof(float));
  }
}

Tensor::Tensor(const vector<size_t> &shape_, float *buf_, int device) : device(device) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();

  if (device >= 0) {
    cudaSetDevice(device);
    cudaMalloc(&buf, N_ * sizeof(float));
    cudaMemcpy(buf, buf_, N_ * sizeof(float), cudaMemcpyHostToDevice);
  } else {  
    buf = (float *) malloc(N_ * sizeof(float));
    memcpy(buf, buf_, N_ * sizeof(float));
  }
}

Tensor::~Tensor() {
  if (buf != nullptr) {
    if (device >= 0) { cudaFree(buf); }
    else { free(buf); }
  }
}

size_t Tensor::num_elem() {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
  return size;
}

Tensor* Tensor::cpu() {
  // Check if the tensor is already on the CPU
  if (device == -1) {
    printf("Tensor is already on the CPU\n");
    return this;
  }

  // Create a CPU tensor with the same shape
  Tensor* cpu_tensor = new Tensor(vector<size_t>(shape, shape + ndim), -1);

  // Copy data from GPU to CPU
  cudaMemcpy(cpu_tensor->buf, buf, num_elem() * sizeof(float), cudaMemcpyDeviceToHost);

  return cpu_tensor;
}

Tensor* Tensor::set_device(int device) {
  // Check if the tensor is already on the device
  if (this->device >= 0) {
    printf("Tensor is already on one of the devices\n");
    return this;
  }

  // Create a tensor on the device
  Tensor* device_tensor = new Tensor(vector<size_t>(shape, shape + ndim), device);

  // Copy data from CPU to device
  cudaMemcpy(device_tensor->buf, buf, num_elem() * sizeof(float), cudaMemcpyHostToDevice);

  return device_tensor;
}