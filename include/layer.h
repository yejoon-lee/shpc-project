#pragma once

#include <vector>

#include "tensor.h"

using std::vector;

#define MATH_PI 3.1415926535f

/* Elementwise operations */
void gelu(Tensor *inout);
void add(Tensor *inout, Tensor *x);
void add_batch(Tensor *inout, Tensor *x);
void scaling(Tensor *inout, float scale);

/* Matmul operations */
void linear_up(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void linear_down(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void linear_qkv(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void linear_out(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void matmul_attnscore(Tensor *in1, Tensor *in2, Tensor *out);
void matmul_attnout(Tensor *in1, Tensor *in2, Tensor *out);
void matmul_ffn(Tensor *in1, Tensor *in2, Tensor *out);

/* Data movement operations */
void copy(Tensor *in, Tensor *out);
void transpose(Tensor *in, Tensor *out);
void transpose_batch(Tensor *in, Tensor *out);
void split_qkv(Tensor *in, Tensor *out);
void split_head(Tensor *in, size_t n_head, Tensor *out);
void concat_head(Tensor *in, Tensor *out);
void extract_qkv(Tensor *in, size_t head_idx, size_t n_head, Tensor *q,
                 Tensor *k, Tensor *v);
void merge_head(Tensor *in, size_t head_idx, size_t n_head, Tensor *out);
void token_pos_embedding(vector<int> *in, Parameter *wte, Parameter *wpe,
                         Tensor *out, int prompt_size, int batch_size);

/* Other operations */
void softmax(Tensor *inout);
void layer_norm(Tensor *inout, Tensor *gamma, Tensor *beta);
void generate_mask(Tensor *inout);
void top1_sampling(Tensor *in, int *next_token_ids);
