#include <mpi.h>

#include <cmath>
#include <cstdio>

#include "layer.h"
#include "model.h"

Parameter *attn_b[NUM_LAYER], *attn_w[NUM_LAYER];
Parameter *proj_b[NUM_LAYER], *proj_w[NUM_LAYER];
Parameter *ln_1_b[NUM_LAYER], *ln_1_g[NUM_LAYER];
Parameter *ln_2_b[NUM_LAYER], *ln_2_g[NUM_LAYER];
Parameter *mlp1_b[NUM_LAYER], *mlp1_w[NUM_LAYER];
Parameter *mlp2_b[NUM_LAYER], *mlp2_w[NUM_LAYER];
Parameter *ln_f_b, *ln_f_g;
Parameter *wpe, *wte;

#define BATCH_SIZE 64

void alloc_and_set_parameters(float *param) {
  size_t pos = 0;
  int order[] = {
      0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9,
  };
  for (int i = 0; i < NUM_LAYER; i++) {
    attn_b[order[i]] = new Parameter({3 * HIDDEN_DIM}, param + pos);
    pos += OFFSET1;
    attn_w[order[i]] = new Parameter({HIDDEN_DIM, 3 * HIDDEN_DIM}, param + pos);
    pos += OFFSET2;
    proj_b[order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    proj_w[order[i]] = new Parameter({HIDDEN_DIM, HIDDEN_DIM}, param + pos);
    pos += OFFSET4;
    ln_1_b[order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    ln_1_g[order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    ln_2_b[order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    ln_2_g[order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    mlp1_b[order[i]] = new Parameter({4 * HIDDEN_DIM}, param + pos);
    pos += OFFSET5;
    mlp1_w[order[i]] = new Parameter({HIDDEN_DIM, 4 * HIDDEN_DIM}, param + pos);
    pos += OFFSET6;
    mlp2_b[order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    mlp2_w[order[i]] = new Parameter({4 * HIDDEN_DIM, HIDDEN_DIM}, param + pos);
    pos += OFFSET6;
  }
  ln_f_b = new Parameter({HIDDEN_DIM}, param + pos);
  pos += OFFSET3;
  ln_f_g = new Parameter({HIDDEN_DIM}, param + pos);
  pos += OFFSET3;
  wpe = new Parameter({MAX_SEQ_LEN, HIDDEN_DIM}, param + pos);
  pos += OFFSET7;
  wte = new Parameter({NUM_VOCAB, HIDDEN_DIM}, param + pos);
  pos += OFFSET8;
}

void free_parameters() {
  for (int i = 0; i < NUM_LAYER; i++) {
    delete attn_b[i];
    delete attn_w[i];
    delete proj_b[i];
    delete proj_w[i];
    delete ln_1_b[i];
    delete ln_1_g[i];
    delete ln_2_b[i];
    delete ln_2_g[i];
    delete mlp1_b[i];
    delete mlp1_w[i];
    delete mlp2_b[i];
    delete mlp2_w[i];
  }
  delete ln_f_b;
  delete ln_f_g;
  delete wpe;
  delete wte;
}

Activation *embd_a, *ffn_proj_a;
Activation *mha_qkv_proj_a, *mha_out_a, *mha_split_qkv_a, *mha_split_head_a,
    *mha_mask_a, *mha_merge_head_a, *mha_q_a, *mha_k_a, *mha_v_a,
    *mha_attn_out_a, *mha_concat_head_a;
Activation *attn_score_a, *k_transposed_a;
Activation *wte_transposed_a, *residual_a, *logit_a;
Activation *transformer_block_a;

void alloc_activations(size_t prompt_size) { // TODO: add batch dim
  embd_a = new Activation({BATCH_SIZE, prompt_size, HIDDEN_DIM});

  ffn_proj_a = new Activation({BATCH_SIZE, prompt_size, 4 * HIDDEN_DIM});

  mha_qkv_proj_a = new Activation({BATCH_SIZE, prompt_size, 3 * HIDDEN_DIM});
  mha_out_a = new Activation({BATCH_SIZE, prompt_size, HIDDEN_DIM});
  mha_split_qkv_a = new Activation({BATCH_SIZE, 3, prompt_size, HIDDEN_DIM});
  mha_split_head_a =
      new Activation({BATCH_SIZE, 3, NUM_HEAD, prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_mask_a = new Activation({prompt_size, prompt_size}); // w/o batch dim
  mha_merge_head_a =
      new Activation({BATCH_SIZE, NUM_HEAD, prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_q_a = new Activation({BATCH_SIZE, prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_k_a = new Activation({BATCH_SIZE, prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_v_a = new Activation({BATCH_SIZE, prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_attn_out_a = new Activation({BATCH_SIZE, prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_concat_head_a = new Activation({BATCH_SIZE, prompt_size, HIDDEN_DIM});

  attn_score_a = new Activation({BATCH_SIZE, prompt_size, prompt_size});
  k_transposed_a = new Activation({BATCH_SIZE, HIDDEN_DIM / NUM_HEAD, prompt_size});

  wte_transposed_a = new Activation({HIDDEN_DIM, NUM_VOCAB});

  residual_a = new Activation({BATCH_SIZE, prompt_size, HIDDEN_DIM});
  logit_a = new Activation({BATCH_SIZE, prompt_size, NUM_VOCAB});
  transformer_block_a = new Activation({BATCH_SIZE, prompt_size, HIDDEN_DIM});
}

void free_activations() {
  delete embd_a;
  delete ffn_proj_a;
  delete mha_qkv_proj_a;
  delete mha_out_a;
  delete mha_split_qkv_a;
  delete mha_split_head_a;
  delete mha_mask_a;
  delete mha_merge_head_a;
  delete mha_q_a;
  delete mha_k_a;
  delete mha_v_a;
  delete mha_attn_out_a;
  delete mha_concat_head_a;
  delete attn_score_a;
  delete k_transposed_a;
  delete wte_transposed_a;
  delete residual_a;
  delete logit_a;
  delete transformer_block_a;
}

/* (Position-wise) Feed-Forward Network
 * @param [in1]     in: [seq_len, HIDDEN_DIM]
 * @param [in2] mlp1_w: [HIDDEN_DIM, 4*HIDDEN_DIM]
 * @param [in3] mlp1_b: [4*HIDDEN_DIM]
 * @param [in4] mlp2_w: [4*HIDDEN_DIM, HIDDEN_DIM]
 * @param [in5] mlp2_b: [HIDDEN_DIM]
 * @param [out]    out: [seq_len, HIDDEN_DIM]
 */
void ffn(Activation *in, Parameter *mlp1_w, Parameter *mlp1_b,
         Parameter *mlp2_w, Parameter *mlp2_b, Activation *out) {
  /* Projection Up:
    [seq_len, HIDDEN_DIM] -> [seq_len, 4*HIDDEN_DIM] */
  linear(in, mlp1_w, mlp1_b, ffn_proj_a);

  /* GELU */
  gelu(ffn_proj_a);

  /* Projection Down:
    [seq_len, 4*HIDDEN_DIM] -> [seq_len, HIDDEN_DIM] */
  linear(ffn_proj_a, mlp2_w, mlp2_b, out);
}

/* Attention
 * @param [in1]    q: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in2]    k: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in3]    v: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in4] mask: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [out]  out: [seq_len, HIDDEN_DIM/NUM_HEAD]
 */
void attention(Activation *q, Activation *k, Activation *v, Activation *mask,
               Activation *out) {
  /* Get Attention score by q @ k */
  transpose_batch(k, k_transposed_a);

  // DEUBG
  // printf("\nKey Transpose\n");
  // Tensor *k_transposed_a_ = k_transposed_a->cpu();
  // size_t H = k_transposed_a_->shape[1];
  // size_t s = k_transposed_a_->shape[2];
  // for (size_t d = 0; d < 10; d++){
  //   printf("%f, ", k_transposed_a_->buf[(H-1) * s + d]);
  // }


  matmul_attnscore(q, k_transposed_a, attn_score_a);

  // DEBUG
  // printf("\nAttention Score\n");
  // Tensor *attn_score_a_ = attn_score_a->cpu();
  // s = attn_score_a_->shape[1];
  // s = attn_score_a_->shape[2];
  // for (size_t d = 0; d < 10; d++){
  //   printf("%f, ", attn_score_a_->buf[(s-1) * s + d]);
  // }

  /* Scaling */
  scaling(attn_score_a, (1.0 / sqrt(k->shape[2])));

  // DEBUG
  // printf("\nScaling\n");
  // Tensor *attn_score_a_scaling = attn_score_a->cpu();
  // s = attn_score_a_scaling->shape[1];
  // s = attn_score_a_scaling->shape[2];
  // for (size_t d = 0; d < 10; d++){
  //   printf("%f, ", attn_score_a_scaling->buf[(s-1) * s + d]);
  // }

  add_batch(attn_score_a, mask);

  // DEUBG
  // printf("\nAdd Mask\n");
  // Tensor *attn_score_a_mask = attn_score_a->cpu();
  // s = attn_score_a_mask->shape[1];
  // s = attn_score_a_mask->shape[2];
  // for (size_t d = 0; d < 10; d++){
  //   printf("%f, ", attn_score_a_mask->buf[(s-1) * s + d]);
  // }

  /* Softmax */
  softmax(attn_score_a);

  // DEBUG
  // printf("\nSoftmax\n");
  // Tensor *attn_score_a_softmax = attn_score_a->cpu();
  // s = attn_score_a_softmax->shape[1];
  // s = attn_score_a_softmax->shape[2];
  // for (size_t d = 0; d < 10; d++){
  //   printf("%f, ", attn_score_a_softmax->buf[(s-1) * s + d]);
  // }

  /* Attention score @ v */
  matmul_attnout(attn_score_a, v, out);
}

/* (Masked) Multi-Head Self Attention
 * @param [in1]     in: [seq_len, HIDDEN_DIM]
 * @param [in2] attn_b: [3*HIDDEN_DIM]
 * @param [in3] attn_w: [HIDDEN_DIM, 3*HIDDEN_DIM]
 * @param [in4] proj_b: [HIDDEN_DIM]
 * @param [in5] proj_w: [HIDDEN_DIM, HIDDEN_DIM]
 * @param [out]    out: [seq_len, HIDDEN_DIM]
 */
void mha(Activation *in, Parameter *attn_b, Parameter *attn_w,
         Parameter *proj_b, Parameter *proj_w, Activation *out) {
  /* QKV projection:
    [seq_len, HIDDEN_DIM] ->
    [seq_len, 3*HIDDEN_DIM] */
  linear(in, attn_w, attn_b, mha_qkv_proj_a);

  /* Split into Q, K, V:
    [seq_len, 3*HIDDEN_DIM] ->
    [3, seq_len, HIDDEN_DIM] */
  split_qkv(mha_qkv_proj_a, mha_split_qkv_a);

  /* Split into multiple heads:
    [3, seq_len, HIDDEN_DIM] ->
    [3, NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
  split_head(mha_split_qkv_a, NUM_HEAD, mha_split_head_a);

  // DEBUG
  // printf("\nSplit Head\n");
  // Tensor *mha_split_head_a_ = mha_split_head_a->cpu();
  // // B, 3, HEAD, S, H_
  // size_t s = mha_split_head_a_->shape[3];
  // size_t H = mha_split_head_a_->shape[4];
  // for (size_t d = 0; d < 10; d++){
  //   printf("%f, ", mha_split_head_a_->buf[16 * s * H + (s-1) * H + d]);
  // }

  /* Generate mask to hide future inputs */
  generate_mask(mha_mask_a);

  /* Perform Attention over each head:
    [NUM_HEAD, 3, seq_len, HIDDEN_DIM/NUM_HEAD] ->
    [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
  for (size_t idx = 0; idx < NUM_HEAD; idx++) {
    /* Extract Q, K, V from qkv_head */
    extract_qkv(mha_split_head_a, idx, NUM_HEAD, mha_q_a, mha_k_a, mha_v_a);

    // DEBUG
    // printf("\nExtract QKV\n");
    // Tensor *mha_q_a_ = mha_q_a->cpu();
    // size_t s = mha_q_a_->shape[1];
    // size_t H = mha_q_a_->shape[2];
    // for (size_t d = 0; d < 10; d++){
    //   printf("%f, ", mha_q_a_->buf[16 * s * H + (s-1) * H + d]);
    // }

    /* Attention */
    attention(mha_q_a, mha_k_a, mha_v_a, mha_mask_a, mha_attn_out_a);

    // DEBUG
    // printf("\nAttention\n");
    // Tensor *mha_attn_out_a_ = mha_attn_out_a->cpu();
    // s = mha_attn_out_a_->shape[1];
    // H = mha_attn_out_a_->shape[2];
    // for (size_t d = 0; d < 10; d++){
    //   printf("%f, ", mha_attn_out_a_->buf[16 * s * H + (s-1) * H + d]);
    // }

    /* Merge each head's attn output
      [seq_len, HIDDEN_DIM/NUM_HEAD] ->
      [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
    merge_head(mha_attn_out_a, idx, NUM_HEAD, mha_merge_head_a);

    // DEBUG
    // printf("\nMerge Head\n");
    // Tensor *mha_merge_head_a_ = mha_merge_head_a->cpu();
    // size_t num_head = mha_merge_head_a_->shape[1];
    // s = mha_merge_head_a_->shape[2];
    // H = mha_merge_head_a_->shape[3];
    // for (size_t d = 0; d < 10; d++){
    //   printf("%f, ", mha_merge_head_a_->buf[16 * num_head * s * H + (s-1) * H + d]);
    // }
    // break;
  }

  /* Concat each heads:
    [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] ->
    [seq_len, HIDDEN_DIM] */
  concat_head(mha_merge_head_a, mha_concat_head_a);

  // DEBUG
  // printf("\nConcat Head\n");
  // Tensor *mha_concat_head_a_ = mha_concat_head_a->cpu();
  // size_t s = mha_concat_head_a_->shape[1];
  // size_t H = mha_concat_head_a_->shape[2];
  // for (size_t d = 0; d < 10; d++){
  //   printf("%f, ", mha_concat_head_a_->buf[(s-1) * H + d]);
  // }

  /* OUT projection:
    [seq_len, HIDDEN_DIM] -> [seq_len, HI DDEN_DIM] */
  linear(mha_concat_head_a, proj_w, proj_b, out);
}

/* Transformer Block
 * @param [in1]      in: [seq_len, HIDDEN_DIM]
 * @param [in2]  attn_b: [3*HIDDEN_DIM]
 * @param [in3]  attn_w: [HIDDEN_DIM, 3*HIDDEN_DIM]
 * @param [in4]  proj_b: [HIDDEN_DIM]
 * @param [in5]  proj_w: [HIDDEN_DIM, HIDDEN_DIM]
 * @param [in6]  ln_1_b: [HIDDEN_DIM]
 * @param [in7]  ln_1_g: [HIDDEN_DIM]
 * @param [in8]  ln_2_b: [HIDDEN_DIM]
 * @param [in9]  ln_2_g: [HIDDEN_DIM]
 * @param [in10] mlp1_b: [4*HIDDEN_DIM]
 * @param [in11] mlp1_w: [HIDDEN_DIM, 4*HIDDEN_DIM]
 * @param [in12] mlp2_b: [HIDDEN_DIM]
 * @param [in13] mlp2_w: [4*HIDDEN_DIM, HIDDEN_DIM]
 * @param [out]     out: [seq_len, HIDDEN_DIM]
 */
void transformer_block(Activation *in, Parameter *attn_b, Parameter *attn_w,
                       Parameter *proj_b, Parameter *proj_w, Parameter *ln_1_b,
                       Parameter *ln_1_g, Parameter *ln_2_b, Parameter *ln_2_g,
                       Parameter *mlp1_b, Parameter *mlp1_w, Parameter *mlp2_b,
                       Parameter *mlp2_w, Activation *out) {
  /* Copy Residual */
  copy(in, residual_a);

  /* Layer Normalization */
  layer_norm(in, ln_1_g, ln_1_b);

  // DEBUG
  // printf("\nLayer Norm\n");
  // Tensor *in_ = in->cpu();
  // size_t s = in_->shape[1];
  // size_t H = in_->shape[2];
  // for (size_t d = 0; d < 10; d++){
  //   printf("%f, ", in_->buf[16 * s * H + (s-1) * H + d]);
  // }

  /* Masked Multi-Head Self-Attention */
  mha(in, attn_b, attn_w, proj_b, proj_w, mha_out_a);
  
  // DEBUG
  // printf("\nMHA\n");
  // Tensor *mha_out_a_ = mha_out_a->cpu();
  // s = mha_out_a_->shape[1];
  // H = mha_out_a_->shape[2];
  // for (size_t d = 0; d < 10; d++){
  //   printf("%f, ", mha_out_a_->buf[16 * s * H + (s-1) * H + d]);
  // }

  /* Add Residual */
  add(mha_out_a, residual_a);

  /* Copy Residual */
  copy(mha_out_a, residual_a);

  /* Layer Normalization */
  layer_norm(mha_out_a, ln_2_g, ln_2_b);

  /* Position-wise Feed-Forward Network */
  ffn(mha_out_a, mlp1_w, mlp1_b, mlp2_w, mlp2_b, out);

  /* Add Residual */
  add(out, residual_a);
}

/* [Model Computation: Token Generation] */
void generate_tokens(int *input, int *output, size_t n_prompt, size_t n_token) {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if (mpi_rank == 0) {
    /* Outer loop: generate tokens for each prompt */
    for (size_t b_strt = 0; b_strt < n_prompt; b_strt += BATCH_SIZE) { 
      // for any given prompt, prompt idx = b_strt + p
      printf("\nPrompt %zu~%zu\n", b_strt, b_strt + BATCH_SIZE - 1);

      int prompt_size = tokens_per_prompt; // fixed to 16

      /* Initialize input prompt */
      vector<int> input_prompt[BATCH_SIZE];
      for (size_t p = 0; p < BATCH_SIZE; p++) {
        input_prompt[p].resize(prompt_size);
        memcpy(input_prompt[p].data(), input + (b_strt + p) * prompt_size,
               prompt_size * sizeof(int));
      }

      /* Inner loop: generate next token */
      for (size_t t = 0; t < n_token; t++) { // n_token = 8 (fixed)
        printf("\nToken %zu", t);

        /* Initialize activations */
        alloc_activations(prompt_size);

        // DEBUG
        // printf("\nInput prompt\n");
        // for (size_t i = 0; i < tokens_per_prompt; i++) {
        //   printf("%d ", input_prompt[16][i]);
        // }

        /* Token + Positional Embedding */
        token_pos_embedding(input_prompt, wte, wpe, embd_a, prompt_size, BATCH_SIZE);

        // DEBUG
        // printf("\nToken + Positional Embedding\n");
        // Tensor *embd_a_ = embd_a->cpu();
        // size_t s = embd_a_->shape[1];
        // size_t H = embd_a_->shape[2];
        // for (size_t d = 0; d < 10; d++) {
        //   printf("%f, ", embd_a_->buf[16 * s * H + (s-1) * H + d]);
        // }

        /* Forward path of Transformer blocks */
        for (size_t l = 0; l < NUM_LAYER; l++) {
          transformer_block(embd_a, attn_b[l], attn_w[l], proj_b[l], proj_w[l],
                            ln_1_b[l], ln_1_g[l], ln_2_b[l], ln_2_g[l],
                            mlp1_b[l], mlp1_w[l], mlp2_b[l], mlp2_w[l],
                            transformer_block_a);

          /* Copy output to embd_a for next block */
          copy(transformer_block_a, embd_a);
          // break;
        }

        // DEBUG
        // printf("\nTransformer Block for token %zu\n", t);
        // embd_a_ = embd_a->cpu();

        // s = embd_a_->shape[1];
        // H = embd_a_->shape[2];
        // for (size_t d = 0; d < 10; d++) {
        //   printf("%f, ", embd_a_->buf[16 * s * H + (s-1) * H + d]);
        // }

        /* Final Layer Normalization */
        layer_norm(embd_a, ln_f_g, ln_f_b);

        /* Projection to vocab. dimension */
        transpose(wte, wte_transposed_a);
        matmul_ffn(embd_a, wte_transposed_a, logit_a);

        // DEBUG : Print logit shape
        // printf("\nLogit: (");
        // for (size_t d = 0; d < logit_a->ndim; d++) {
        //   printf("%zu, ", logit_a->shape[d]);
        // }
        // printf(")");

        /* Greedy sampling (only last timestep is considered) */
        int *next_token_ids = (int *)malloc(BATCH_SIZE * sizeof(int));
        top1_sampling(logit_a, next_token_ids);

        /* CHEAT */
        // if (t == 2) {
        //   printf("%d\n", next_token_ids[0]);
        //   next_token_ids[0] = 11;
        // }
        // if (t == 5) {
        //   printf("%d\n", next_token_ids[0]);
        //   next_token_ids[0] = 5586;
        // }


        /* Update input prompt and prompt size */ 
        for (size_t p = 0; p < BATCH_SIZE; p++) {
          input_prompt[p].push_back(next_token_ids[p]);
        }
        prompt_size += 1;

        /* Store generated token to output */ 
        for (size_t p = 0; p < BATCH_SIZE; p++) {
          output[(b_strt + p) * n_token + t] = next_token_ids[p];
        }

        /* Finalize activations for next token generation */
        free_activations();
        free(next_token_ids);
        // break;
      }
    
    // DEBUG
    // printf("\nOutput pushed to Input\n");
    // for (size_t i = tokens_per_prompt; i < input_prompt[16].size(); i++) {
    //   printf("%d ", input_prompt[16][i]);
    // }
    // printf("\nSaved Output\n");
    // for (size_t t = 0; t < n_token; t++) {
    //   printf("%d ", output[(b_strt + 16) * n_token + t]);
    // }
    // break;
    }
  }
}
