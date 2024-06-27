#include "cstring"
#include "param.h"
#include <iostream>
#include <hls_math.h>


void transfor(f_32 in[3][128][128], D32 out[3][128][128]){
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 128; j++) {
			for (int k = 0; k < 128; k++){
                out[i][j][k] = in[i][j][k];
			}
		}
	}
}


// Initial Convolution Function
void conv1(D32 in[3][128][128], D32 Kw[32][3][5][5], D32 Kb[32], D32 out[32][64][64]){
#pragma HLS ARRAY_PARTITION variable=Kw complete dim=1
#pragma HLS ARRAY_PARTITION variable=Kb complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1


    for(int i = 0; i < 64; i++) {
        for(int j = 0; j < 64; j++) {
            for(int k = 0; k < 32; k++) {
                D32 sum = 0;
                for(int c = 0; c < 3; c++) {  // kernel size
                    for(int y = 0; y < 5; y++) {
                        for(int x = 0; x < 5; x++) {
                        	int row = i*2 + y - 2;
                        	int col = j*2 + x - 2;
                        	if (row >= 0 && row < 128 && col >= 0 && col < 128) {
                        		sum += in[c][row][col] * Kw[k][c][y][x];
                        	}
                        }
                    }
                }
                out[k][i][j] = sum + Kb[k];  // bias
                if (out[k][i][j] < 0){
                	out[k][i][j] = 0;
                }
                }
            }
        }
    }

// Pooling Function
void pool1(D32 in[32][64][64], D32 out[32][32][32]){
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 32; j++) {
            for(int k = 0; k < 32; k++) {
                D32 max = 0;
                for(int y = 0; y < 2; y++) {
                    for(int x = 0; x < 2; x++) {
                        if(in[k][2*i + y][2*j + x] > max) {
                            max = in[k][2*i + y][2*j + x];
                        }
                    }
                }
                out[k][i][j] = max;
            }
        }
    }
}

// Copy Input to Identity
void copy_input_to_identity(D32 input[32][32][32], D32 identity[32][32][32]) {
#pragma HLS ARRAY_PARTITION variable=input complete dim=1
#pragma HLS ARRAY_PARTITION variable=identity complete dim=1

    for (int k = 0; k < 32; k++) {
        for (int x = 0; x < 32; x++) {
            for (int y = 0; y < 32; y++) {
                identity[y][k][x] = input[y][k][x];
            }
        }
    }
}

// First Convolution in Residual Block
void l_conv1(D32 in[32][32][32], D32 Kw[64][32][3][3], D32 Kb[64], D32 out[64][16][16]){
#pragma HLS ARRAY_PARTITION variable=Kw complete dim=1
#pragma HLS ARRAY_PARTITION variable=Kb complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 16; j++) {
            for(int k = 0; k < 64; k++) {
                D32 sum = 0;
                for(int c = 0; c < 32; c++) {
                    for(int y = 0; y < 3; y++) {
                        for(int x = 0; x < 3; x++) {
                            int in_i = i * 2 + y - 1;
                            int in_j = j * 2 + x - 1;
                            if (in_i >= 0 && in_i < 32 && in_j >= 0 && in_j < 32) {
                                sum += in[c][in_i][in_j] * Kw[k][c][y][x];
                            }
                        }
                    }
                }
                out[k][i][j] = sum + Kb[k];
                if(out[k][i][j] < 0) {
                	out[k][i][j] = 0;
                }
            }
        }
    }
}

// Second Convolution in Residual Block
void l_conv2(D32 in[64][16][16], D32 Kw[64][64][3][3], D32 out[64][16][16]){
#pragma HLS ARRAY_PARTITION variable=Kw complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1

    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 16; j++) {
            for(int k = 0; k < 64; k++) {
                D32 sum = 0;
                for(int c = 0; c < 64; c++) {
                    for(int y = 0; y < 3; y++) {
                        for(int x = 0; x < 3; x++) {
                            int row = i + y - 1;
                            int col = j + x - 1;
                            if (row >= 0 && row < 16 && col >= 0 && col < 16) {
                                sum += in[c][row][col] * Kw[k][c][y][x];
                            }
                        }
                    }
                }
                out[k][i][j] = sum;
            }
        }
    }
}

void l_bn1(D32 in[64][16][16], D16 Kw[64], D16 Kb[64], D16 Km[64], D16 Kv[64], D32 out[64][16][16]) {
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
#pragma HLS ARRAY_PARTITION variable=Kw complete dim=1
#pragma HLS ARRAY_PARTITION variable=Kb complete dim=1
#pragma HLS ARRAY_PARTITION variable=Km complete dim=1
#pragma HLS ARRAY_PARTITION variable=Kv complete dim=1

    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 16; j++) {
            for(int k = 0; k < 64; k++) {
                D16 normalized = (in[k][i][j] - Km[k]) * Kv[k];
                D32 result = Kw[k] * normalized + Kb[k];
                out[k][i][j] = result;
            }
        }
    }
}

// Shortcut Convolution
void l_shortcut_conv(D32 identity[32][32][32], D32 Kw[64][32][1][1], D32 out[64][16][16]) {
#pragma HLS ARRAY_PARTITION variable=Kw complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 16; j++) {
            for(int k = 0; k < 64; k++) {
                D32 sum = 0;
                for(int c = 0; c < 32; c++) {
                    sum += identity[c][i*2][j*2] * Kw[k][c][0][0];
                }
                out[k][i][j] = sum;
            }
        }
    }
}

void l_shortcut_bn(D32 in[64][16][16], D16 Kw[64], D16 Kb[64], D16 Km[64], D16 Kv[64], D32 identity1[64][16][16]) {
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=identity1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=Kw complete dim=1
#pragma HLS ARRAY_PARTITION variable=Kb complete dim=1
#pragma HLS ARRAY_PARTITION variable=Km complete dim=1
#pragma HLS ARRAY_PARTITION variable=Kv complete dim=1

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            for (int k = 0; k < 64; k++) {
                D16 normalized = (in[k][i][j] - Km[k]) * Kv[k];
                D32 result = Kw[k] * normalized + Kb[k];
                identity1[k][i][j] = result;
            }
        }
    }
}

void l_shortcut_add(D32 in[64][16][16], D32 identity1[64][16][16], D32 out[64][16][16]) {
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=identity1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1

    for (int k = 0; k < 16; k++) {
        for (int x = 0; x < 16; x++) {
            for (int y = 0; y < 64; y++) {
                out[y][k][x] = in[y][k][x] + identity1[y][k][x];
            }
        }
    }
}

void relu(D32 input[64][16][16], D32 output[64][16][16]){
#pragma HLS ARRAY_PARTITION variable=input complete dim=1
#pragma HLS ARRAY_PARTITION variable=output complete dim=1

    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 16; j++) {
            for(int k = 0; k < 64; k++) {
            	output[k][i][j] = (input[k][i][j] < 0) ? D32(0) : input[k][i][j];
            	 }
            }
        }
    }

void adaptive_avg_pool(D32 input[64][16][16], D32 output[64]){
#pragma HLS ARRAY_PARTITION variable=input complete dim=1
#pragma HLS ARRAY_PARTITION variable=output complete dim=1

    for(int k = 0; k < 64; k++) {
        D16 sum = 0;
        for(int i = 0; i < 16; i++) {
            for(int j = 0; j < 16; j++) {
                sum += D16(input[k][i][j]);
            }
        }
        output[k] = sum >> 8; // 相当于除以256
    }
}


// Fully Connected Layer 1
void fc1(D32 in[64], D32 fc1_w[48][64], D32 fc1_b[48], D32 out[48]) {
#pragma HLS ARRAY_PARTITION variable=fc1_w complete dim=1
#pragma HLS ARRAY_PARTITION variable=fc1_b complete dim=1
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
    for(int i = 0; i < 48; i++) {
        D32 sum = fc1_b[i];
        for(int j = 0; j < 64; j++) {
            sum += fc1_w[i][j] * in[j];
        }
        out[i] = sum;
    }
}

// Fully Connected Layer 2
void fc2(D32 in[48], D32 fc2_w[24][48], D32 fc2_b[24], D32 out[24]) {
#pragma HLS ARRAY_PARTITION variable=fc2_w complete dim=1
#pragma HLS ARRAY_PARTITION variable=fc2_b complete dim=1
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
    for(int i = 0; i < 24; i++) {
        D32 sum = fc2_b[i];
        for(int j = 0; j < 48; j++) {
            sum += fc2_w[i][j] * in[j];
        }

        out[i] = sum;
    }
}

// Fully Connected Layer 3
void fc3(D32 in[24], D32 fc3_w[2][24], D32 fc3_b[2], D32 out[2]) {
#pragma HLS ARRAY_PARTITION variable=fc3_w complete dim=1
#pragma HLS ARRAY_PARTITION variable=fc3_b complete dim=1
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
    for(int i = 0; i < 2; i++) {
        D32 sum = fc3_b[i];
        for(int j = 0; j < 24; j++) {
            sum += fc3_w[i][j] * in[j];
        }

        out[i] = sum;
    }
}

// Softmax Function
void softmax(D32 in[2], D32 out[2], f_32 f_out[2]){
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1

    D32 max_val = in[0];
    D32 sum = 0.0;

    for (int i = 0; i < 2; i++){
    	if (in[i] > max_val)
    		max_val = in[i];
    }

    for (int i = 0; i < 2; i++){
    	out[i] = hls::exp(in[i] - max_val);
    	sum += out[i];
    }

    for (int i = 0; i< 2; i++){
#pragma HLS UNROLL
    	out[i] /= sum;
    }

    for(int i = 0; i < 2; i++) {
		f_out[i] = out[i].to_half();

	}
}

