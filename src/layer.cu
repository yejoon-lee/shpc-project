// Invalid but no-error CUDA code

#include "layer.h"

#include <cuda_runtime.h>
#include <mpi.h>

#define DIV_CEIL(a, b) (((a) + (b)-1) / (b))

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)


// CUDA Kernel for token_pos_embedding
__global__ void token_pos_embedding_kernel(int *in, float *wte, float *wpe, float *out, size_t B, size_t s, size_t H) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < B && i < s && j < H) {
        // out[b,i,j] = wte[in[b,i],j] + wpe[i,j]
        out[(b * s * H) + i * H + j] = wte[in[b * s + i] * H + j] + wpe[i * H + j];
    }
}

/* Token + Positional Embedding
 * @param [in1]  in: [B, s]
 * @param [in2] wte: [NUM_VOCAB, H]
 * @param [in3] wpe: [MAX_SEQ_LEN, H]
 * @param [out] out: [B, s, H]
 * 'B' is the batch size.
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
void token_pos_embedding(vector<int> in, Tensor *wte, Tensor *wpe,
                              Tensor *out, int prompt_size) {
  size_t s = prompt_size;
  size_t B = in.size() / s;
  size_t H = wte->shape[1];

  // `in` is on the host, so we need to copy it to the device
  int *d_in;
  CHECK_CUDA(cudaMalloc(&d_in, B*s * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_in, in.data(), B*s * sizeof(int), cudaMemcpyHostToDevice));

  dim3 blockDim(16, 2, 16);
  dim3 gridDim(DIV_CEIL(B, blockDim.x), DIV_CEIL(s, blockDim.y), DIV_CEIL(H, blockDim.z));

  token_pos_embedding_kernel<<<gridDim, blockDim>>>(d_in, wte->buf, wpe->buf, out->buf, B, s, H);

  CHECK_CUDA(cudaFree(d_in));
}

/* GELU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
// CUDA Kernel for GELU
__global__ void gelu_kernel(float *inout, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = inout[idx];
        inout[idx] = 0.5 * x * (1.f + tanh(sqrt(2.f / MATH_PI) * (x + 0.044715f * x * x * x)));
    }
}

// GELU using CUDA
void gelu(Tensor *inout) {
  size_t N = inout->num_elem();

  gelu_kernel<<<DIV_CEIL(N, 256), 256>>>(inout->buf, N);
  CHECK_CUDA(cudaGetLastError());
}


/* Softmax (w/ Max Trick)
 * @param [in & out] inout: [s, H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
__global__ void softmax_kernel(float *inout, size_t s, size_t H) {
    // Calculate the thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < s){
      // Find the maximum value in the row
      float max_val = inout[i * H];
      for (size_t j = 1; j < H; j++) {
          if (inout[i * H + j] > max_val) {
              max_val = inout[i * H + j];
          }
      }

      // Compute the denominator
      float sum = 0.0;
      for (size_t j = 0; j < H; j++) {
        inout[i * H + j] = exp(inout[i * H + j] - max_val);
        sum += inout[i * H + j];
      }

      // Normalize the row
      for (size_t j = 0; j < H; j++) {
        inout[i * H + j] /= sum;
      }
    }
}

void softmax(Tensor *inout) {
    size_t s = inout->shape[0];
    size_t H = inout->shape[1];

    // Define grid and block dimensions
    dim3 blockDim(32); // warp = 32 threads
    dim3 gridDim(DIV_CEIL(s, blockDim.x));

    // Launch the kernel
    softmax_kernel<<<gridDim, blockDim>>>(inout->buf, s, H);
    CHECK_CUDA(cudaGetLastError());
}



// CUDA Kernel for layer normalization
__global__ void layer_norm_kernel(float *inout, float *gamma, float *beta, size_t B, size_t s, size_t H) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < B && i < s){
        float eps = 1e-5;
        float mean = 0;
        float var = 0;

        // Compute the mean and variance
        for (size_t j = 0; j < H; j++) {
            mean += inout[b * s * H + i * H + j];
            var += inout[b * s * H + i * H + j] * inout[b * s * H + i * H + j];
        }
        mean /= H;
        var = var / H - mean * mean;

        // Normalize the row
        for (size_t j = 0; j < H; j++) {
            inout[b * s * H + i * H + j] = (inout[b * s * H + i * H + j] - mean) *
            (1.0 / sqrt(var + eps)) * gamma[j] + beta[j];
        }
    }
}

/* Layer Normalization
 * @param [in1 & out] inout: [B, s, H]
 * @param [in2]       gamma: [H]
 * @param [in3]        beta: [H]
 * 'B' is the batch size.
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
void layer_norm(Tensor *inout, Tensor *gamma, Tensor *beta) {
  size_t B = inout->shape[0];
  size_t s = inout->shape[1];
  size_t H = inout->shape[2];

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim(DIV_CEIL(B, blockDim.x), DIV_CEIL(s, blockDim.y));

    // Launch the kernel
    layer_norm_kernel<<<gridDim, blockDim>>>(inout->buf, gamma->buf, beta->buf, B, s, H);
    CHECK_CUDA(cudaGetLastError());
}

// CUDA Kernel for linear
__global__ void linear_kernel(float *in, float *W, float *Bias, float *out, size_t B, size_t M, size_t K, size_t N) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < B && i < M && j < N) {
        float sum = 0.0;
        for (size_t k = 0; k < K; k++) {
            sum += in[b * M * K + i * K + k] * W[k * N + j];
        }
        out[b * M * N + i * N + j] = sum + Bias[j];
    }
}

/* Linear
 * @param [in1]  in: [B, M, K]
 * @param [in2]   w: [K, N]
 * @param [in3]   b: [N]
 * @param [out] out: [B, M, N]
 */
void linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t B = in->shape[0];
  size_t M = in->shape[1];
  size_t K = in->shape[2];
  size_t N = w->shape[1];

  // Define grid and block dimensions
  dim3 blockDim(8, 8, 8);
  dim3 gridDim(DIV_CEIL(B, blockDim.x), DIV_CEIL(M, blockDim.y), DIV_CEIL(N, blockDim.z));

  // Launch the kernel
  linear_kernel<<<gridDim, blockDim>>>(in->buf, w->buf, b->buf, out->buf, B, M, K, N);
  CHECK_CUDA(cudaGetLastError());
}

/* Matmul
 * @param [in1]  in1: [M, K]
 * @param [in2]  in2: [K, N]
 * @param [out]  out: [M, N]
 */
// CUDA Kernel for matmul
__global__ void matmul_kernel(float *in1, float *in2, float *out, size_t M, size_t K, size_t N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        float sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += in1[i * K + k] * in2[k * N + j];
        }
        out[i * N + j] = sum;
    }
}

// Matmul using CUDA
void matmul(Tensor *in1, Tensor *in2, Tensor *out) {
  size_t M = in1->shape[0];
  size_t K = in1->shape[1];
  size_t N = in2->shape[1];

  // Define grid and block dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim(DIV_CEIL(N, blockDim.x), DIV_CEIL(M, blockDim.y));

  // Launch the kernel
  matmul_kernel<<<gridDim, blockDim>>>(in1->buf, in2->buf, out->buf, M, K, N);
  CHECK_CUDA(cudaGetLastError());
}

/* Transpose
 * @param [in1]  in: [M, N]
 * @param [out] out: [N, M]
 */
// CUDA Kernel for transpose
__global__ void transpose_kernel(float *in, float *out, size_t M, size_t N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        out[j * M + i] = in[i * N + j];
    }
}

// Transpose using CUDA
void transpose(Tensor *in, Tensor *out) {
  size_t M = in->shape[0];
  size_t N = in->shape[1];

  // Define grid and block dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim(DIV_CEIL(N, blockDim.x), DIV_CEIL(M, blockDim.y));

  // Launch the kernel
  transpose_kernel<<<gridDim, blockDim>>>(in->buf, out->buf, M, N);
  CHECK_CUDA(cudaGetLastError());
}

/* Scaling
 * @param [in1 & out] inout: [N]
 * @param [in2]       scale: [1]
 * 'N' is the number of elements in the tensor.
 */
// CUDA Kernel for scaling
__global__ void scaling_kernel(float *inout, float scale, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { inout[idx] *= scale; }
}

// Scaling using CUDA
void scaling(Tensor *inout, float scale) {
  size_t N = inout->num_elem();

  scaling_kernel<<<DIV_CEIL(N, 256), 256>>>(inout->buf, scale, N);
  CHECK_CUDA(cudaGetLastError());
}

/* Generate mask
 * @param [in & out] inout: [s, s]
 * 's' is the number of tokens in the prompt.
 */
// CUDA Kernel for generate mask
__global__ void generate_mask_kernel(float *inout, size_t s) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < s && j < s) {
        inout[i * s + j] = (i >= j) ? 0 : -1e10;
    }
}

// Generate mask using CUDA
void generate_mask(Tensor *inout) {
  size_t s = inout->shape[0];

  // Define grid and block dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim(DIV_CEIL(s, blockDim.x), DIV_CEIL(s, blockDim.y));

  // Launch the kernel
  generate_mask_kernel<<<gridDim, blockDim>>>(inout->buf, s);
  CHECK_CUDA(cudaGetLastError());
}

// CUDA Kernel for copy
__global__ void copy_kernel(float *in, float *out, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { out[idx] = in[idx]; }
}

/* Copy
 * @param [in1]  in: [N]
 * @param [out] out: [N]
 * 'N' is the number of elements in the tensor.
 */
void copy(Tensor *in, Tensor *out) {
  size_t N = in->num_elem();

  copy_kernel<<<DIV_CEIL(N, 256), 256>>>(in->buf, out->buf, N);
  CHECK_CUDA(cudaGetLastError());
}


/* Add GPU kernel
 * @param [in1 & out] inout: [N]
 * @param [in2]           x: [N]
 * 'N' is the number of elements in the tensor.
 */
__global__ void add_kernel(float *inout, float *x, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) { inout[idx] += x[idx]; }
}

/* Add using CUDA GPU
 * @param [in1 & out] inout: [N]
 * @param [in2]           x: [N]
 * 'N' is the number of elements in the tensor.
 */
void add(Tensor *inout, Tensor *x) {
  size_t N = inout->num_elem();

  add_kernel<<<(N + 255) / 256, 256>>>(inout->buf, x->buf, N);
  CHECK_CUDA(cudaGetLastError());
}

__global__ void split_qkv_kernel(float *in, float *out, size_t B, size_t s, size_t H) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < B && j < s && k < H / 3) {
      for (size_t i = 0; i < 3; i++) {
        // out[b, i, j, k] = in[b, j, i * (H / 3) + k]
        out[(b * s * H) + i * s * (H / 3) + j * (H / 3) + k] = in[(b * s * H) + j * H + i * (H / 3) + k];
      }
    }
}

/* Split into QKV
 * @param [in1]  in: [B, s, H]
 * @param [out] out: [B, 3, s, H/3]
 */
void split_qkv(Tensor *in, Tensor *out) {
  size_t B = in->shape[0];
  size_t s = in->shape[1];
  size_t H = in->shape[2];

  // Define grid and block dimensions
  dim3 blockDim(16, 4, 8);
  dim3 gridDim(DIV_CEIL(B, blockDim.x), DIV_CEIL(s, blockDim.y), DIV_CEIL(H / 3, blockDim.z));

  // Launch the kernel
  split_qkv_kernel<<<gridDim, blockDim>>>(in->buf, out->buf, B, s, H);
  CHECK_CUDA(cudaGetLastError());
}

// CUDA Kernel for split_head
__global__ void split_head_kernel(float *in, float *out, size_t B, size_t s, size_t H, size_t n_head) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < B && j < n_head && k < s) {
      for (size_t i = 0; i < 3; i++) {
        for (size_t l = 0; l < H / n_head; l++) {
            // out[b, i, j, k, l] = in[b, i, k, j * (H / n_head) + l]
            out[(b * 3 * s * H) + i * s * H + j * s * H / n_head + k * H / n_head + l] =
                in[(b * 3 * s * H) + i * s * H + k * H + j * H / n_head + l];
        }
      }
    }
}

/* Split into heads
 * @param [in1]  in: [B, 3, s, H]
 * @param [out] out: [B, 3, n_head, s, H/n_head]
 * 'B' is the batch size.
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 * 'n_head' is the number of heads.
 */
void split_head(Tensor *in, size_t n_head, Tensor *out) {
  size_t B = in->shape[0];
  size_t s = in->shape[2];
  size_t H = in->shape[3];

  // Define grid and block dimensions
  dim3 blockDim(16, 4, 8);
  dim3 gridDim(DIV_CEIL(B, blockDim.x), DIV_CEIL(s, blockDim.y), n_head);

  // Launch the kernel
  split_head_kernel<<<gridDim, blockDim>>>(in->buf, out->buf, B, s, H, n_head);
  CHECK_CUDA(cudaGetLastError());
}

/* Extract Q, K, V from QKV head
 * @param [in1]       in: [3, n_head, s, H_]
 * @param [in2] head_idx: [1]
 * @param [in3]   n_head: [1]
 * @param [out]        q: [s, H_]
 * @param [out]        k: [s, H_]
 * @param [out]        v: [s, H_]
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 * 'n_head' is the number of heads.
 */
// CUDA Kernel for extract_qkv
__global__ void extract_qkv_kernel(float *in, size_t head_idx, size_t n_head, float *q, float *k, float *v, size_t s, size_t H_) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < s && j < H_) {
        q[i * H_ + j] = in[0 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
        k[i * H_ + j] = in[1 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
        v[i * H_ + j] = in[2 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
    }   
}

// Extract Q, K, V from QKV head using CUDA
void extract_qkv(Tensor *in, size_t head_idx, size_t n_head, Tensor *q, Tensor *k, Tensor *v) {
  size_t s = in->shape[2];
  size_t H_ = in->shape[3];  // = HIDDEN_DIM/NUM_HEAD

  // Define grid and block dimensions
  dim3 blockDim(8, 32);
  dim3 gridDim(DIV_CEIL(s, blockDim.x), DIV_CEIL(H_, blockDim.y));

  // Launch the kernel
  extract_qkv_kernel<<<gridDim, blockDim>>>(in->buf, head_idx, n_head, q->buf, k->buf, v->buf, s, H_);
  CHECK_CUDA(cudaGetLastError());
}

/* Merge each heads
 * @param [in1]       in: [s, H_]
 * @param [in2] head_idx: [1]
 * @param [in3]   n_head: [1]
 * @param [out]      out: [n_head, s, H_]
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 * 'n_head' is the number of heads.
 */
// CUDA Kernel for merge_head
__global__ void merge_head_kernel(float *in, size_t head_idx, size_t n_head, float *out, size_t s, size_t H_) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < s && j < H_) {
        out[head_idx * s * H_ + i * H_ + j] = in[i * H_ + j];
    }
}

// Merge each heads using CUDA
void merge_head(Tensor *in, size_t head_idx, size_t n_head, Tensor *out) {
  size_t s = in->shape[0];
  size_t H_ = in->shape[1];  // = HIDDEN_DIM/NUM_HEAD

  // Define grid and block dimensions
  dim3 blockDim(8, 32);
  dim3 gridDim(DIV_CEIL(s, blockDim.x), DIV_CEIL(H_, blockDim.y));

  // Launch the kernel
  merge_head_kernel<<<gridDim, blockDim>>>(in->buf, head_idx, n_head, out->buf, s, H_);
  CHECK_CUDA(cudaGetLastError());
}

/* Concatenate each heads
 * @param [in1]     in: [n_head, s, H_]
 * @param [out]    out: [s, H_*n_head]
 * 'n_head' is the number of heads.
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 */
// CUDA Kernel for concat_head
__global__ void concat_head_kernel(float *in, float *out, size_t n_head, size_t s, size_t H_) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < s && j < n_head) {
        for (size_t k = 0; k < H_; k++) {
            out[i * n_head * H_ + j * H_ + k] = in[j * s * H_ + i * H_ + k];
        }
    }
}

// Concatenate each heads using CUDA
void concat_head(Tensor *in, Tensor *out) {
  size_t n_head = in->shape[0];
  size_t s = in->shape[1];
  size_t H_ = in->shape[2];  // = HIDDEN_DIM/NUM_HEAD

  // Define grid and block dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim(DIV_CEIL(s, blockDim.x), DIV_CEIL(n_head, blockDim.y));

  // Launch the kernel
  concat_head_kernel<<<gridDim, blockDim>>>(in->buf, out->buf, n_head, s, H_);
  CHECK_CUDA(cudaGetLastError());
}

/* Greedy Max Sampling
 * @param  [in1]  in: [s, V]
 * @return [ret] out: [1]
 * 's' is the number of tokens in the prompt.
 * 'V' is the number of vocabulary.
 */
int top1_sampling(Tensor *in) {
  size_t s = in->shape[0];
  size_t V = in->shape[1];

  int out = 0;
  float max = -INFINITY;
  for (size_t i = 0; i < V; i++) {
    if (in->buf[(s - 1) * V + i] > max) {
      max = in->buf[(s - 1) * V + i];
      out = i;
    }
  }

  return out;
}
