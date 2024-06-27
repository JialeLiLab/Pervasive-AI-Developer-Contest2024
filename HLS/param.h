#ifndef _PARAM_H
#define _PARAM_H

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>

typedef ap_fixed<12, 5, AP_RND, AP_SAT> D32; // data type
typedef ap_fixed<22, 10, AP_RND, AP_SAT> D16; // data type
typedef half f_32;
using namespace std;

void transfor(f_32 in[3][128][128], D32 out[3][128][128]);
void conv1(D32 in[3][128][128], D32 Kw[32][3][5][5], D32 Kb[32], D32 out[32][64][64]);
void pool1(D32 in[32][64][64], D32 out[32][32][32]);

void copy_input_to_identity(D32 input[32][32][32], D32 identity[32][32][32]);
void l_conv1(D32 in[32][32][32], D32 Kw[64][32][3][3], D32 Kb[64], D32 out[64][16][16]);
void l_conv2(D32 in[64][16][16], D32 Kw[64][64][3][3], D32 out[64][16][16]);
void l_bn1(D32 in[64][16][16], D16 Kw[64], D16 Kb[64], D16 Km[64], D16 Kv[64], D32 out[64][16][16]);

void l_shortcut_conv(D32 identity[32][32][32], D32 Kw[64][32][1][1], D32 out[64][16][16]);
void l_shortcut_bn(D32 in[64][16][16], D16 Kw[64], D16 Kb[64], D16 Km[64], D16 Kv[64], D32 identity1[64][16][16]);
void l_shortcut_add(D32 in[64][16][16], D32 identity1[64][16][16], D32 out[64][16][16]);
void relu(D32 input[64][16][16], D32 output[64][16][16]);

void adaptive_avg_pool(D32 input[64][16][16], D32 output[64]);

void fc1(D32 in[64], D32 fc1_w[48][64], D32 fc1_b[48], D32 out[48]);
void fc2(D32 in[48], D32 fc1_w[24][48], D32 fc1_b[24], D32 out[24]);
void fc3(D32 in[24], D32 fc1_w[2][24], D32 fc1_b[2], D32 out[2]);
void softmax(D32 in[2], D32 out[2], f_32 f_out[2]);

void resnet(f_32 img_in[3][128][128], f_32 Cout[2]);

#endif
