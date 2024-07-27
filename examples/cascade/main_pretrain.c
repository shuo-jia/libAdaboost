#include "lib/pretrain.h"

// 最大正例样本数量
#define MAX_POSI 1000
// 最大负例样本同正例样本之比
#define NEG_PER_POSI 20
// 图片所在目录
#define DIRPATH "../BioID-FaceDatabase-V1.2/"
// 正例标注文件
#define P_MARKPATH "./p_mark"
// 正例标注文件
#define N_MARKPATH "./n_mark"
// 训练图像尺寸（像素）
#define FACE_SIZE 20
// 模型保存路径
#define MODEL_PATH "./cascade_data"
// 最大错误率
#define MAX_ERR 1E-3

int main(void)
{
	struct cascade cascade;
	struct haar_ada_handles handles;
	ada_set_haar (&handles, ADA_ASYM, ADA_GA);
	pre_train (&cascade, MAX_POSI, FACE_SIZE, MAX_ERR, P_MARKPATH,
			N_MARKPATH, DIRPATH, &handles);

	FILE * file = fopen (MODEL_PATH, "wb");
	cas_write (&cascade, file, &handles);
	cas_free (&cascade, &handles);
	fclose (file);

	return 0;
}
