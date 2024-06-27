#include "param.h"
#include <iostream>
#include <typeinfo>

void resnet(f_32 img_in[3][128][128], f_32 Cout[2]){
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE m_axi depth=49152 port=img_in offset=slave bundle=IMG
#pragma HLS INTERFACE m_axi depth=2 port=Cout offset=slave bundle=OUT


D32 d_in[3][128][128];
D32 c1_out[32][64][64];
D32 p1_out[32][32][32];
D32 identity[32][32][32];
D32 l_c1_out[64][16][16];
D32 l_c2_out[64][16][16];
D32 l_bn1_out[64][16][16];
D32 l_short_c_out[64][16][16];
D32 identity1[64][16][16];
D32 shortcut_out[64][16][16];
D32 relu_out[64][16][16];
D32 avgPool_out[64];
D32 fc1_out[48];
D32 fc2_out[24];
D32 fc3_out[2];
D32 soft_out[2];

D32 c1_w[32][3][5][5]={
#include "c1_w.h"
};

D32 c1_b[32]={
#include "c1_b.h"
};

D32 l_c1_w[64][32][3][3]={
#include "l_c1_w.h"
};

D32 l_c1_b[64]={
#include "l_c1_b.h"
};

D32 l_c2_w[64][64][3][3]={
#include "l_c2_w.h"
};

D16 l_bn1_w[64]={
#include "l_bn_w.h"
};
D16 l_bn1_b[64]={
#include "l_bn_b.h"
};
D16 l_bn1_m[64]={
#include "l_bn_m.h"
};
D16 l_bn1_v[64]={
#include "l_bn_qv.h"
};

D32 l_short_c_w[64][32][1][1]={
#include "l_short_c_w.h"
};

D16 l_short_bn_w[64]={
#include "l_short_bn_w.h"
};
D16 l_short_bn_b[64]={
#include "l_short_bn_b.h"
};
D16 l_short_bn_m[64]={
#include "l_short_bn_m.h"
};
D16 l_short_bn_v[64]={
#include "l_short_bn_qv.h"
};

D32 fc1_w[48][64]={
#include "fc1_w.h"
};
D32 fc1_b[48]={
#include "fc1_b.h"
};
D32 fc2_w[24][48]={
#include "fc2_w.h"
};
D32 fc2_b[24]={
#include "fc2_b.h"
};
D32 fc3_w[2][24]={
#include "fc3_w.h"
};
D32 fc3_b[2]={
#include "fc3_b.h"
};


transfor(img_in, d_in);
conv1(d_in, c1_w, c1_b, c1_out);
pool1(c1_out, p1_out);
copy_input_to_identity(p1_out, identity);
l_conv1(p1_out, l_c1_w, l_c1_b, l_c1_out);
l_conv2(l_c1_out, l_c2_w, l_c2_out);
l_bn1(l_c2_out, l_bn1_w, l_bn1_b, l_bn1_m, l_bn1_v, l_bn1_out);

l_shortcut_conv(identity, l_short_c_w, l_short_c_out);
l_shortcut_bn(l_short_c_out, l_short_bn_w, l_short_bn_b, l_short_bn_m, l_short_bn_v, identity1);
l_shortcut_add(l_bn1_out, identity1, shortcut_out);
relu(shortcut_out, relu_out);

adaptive_avg_pool(relu_out, avgPool_out);

fc1(avgPool_out, fc1_w, fc1_b, fc1_out);
fc2(fc1_out, fc2_w, fc2_b, fc2_out);
fc3(fc2_out, fc3_w, fc3_b, fc3_out);
softmax(fc3_out, soft_out, Cout);
}




























