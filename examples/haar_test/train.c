#include <stdlib.h>
#include "cascade.h"
#include "lib/sample.h"

// 图片所在目录
#define DIRPATH "../BioID-FaceDatabase-V1.2/"
// 标注文件
#define MARKPATH "./p_mark"
// 每个 AdaBoost 分类器的检测率、假阳率
#define DET_RATE 0.995
#define FP_RATE 0.8
// 最大假阳率
#define MAX_FP_RATE 1E-5
// 训练集占比（其余设置为验证集）
#define TRAIN_SET_PERCENT 0.7
// 训练图像尺寸（像素）
#define FACE_SIZE 20
// 模型保存路径
#define MODEL_PATH "./cascade_data"
int main(void)
{
	struct sample sample;
	struct cascade cascade;
	struct haar_ada_handles handles;
	FILE * mark = fopen (MARKPATH, "r");
	FILE * n_mark = fopen ("n_mark", "r");
	if (mark == NULL)
		exit(1);

	//if (!get_rand_sample (&sample, 1000, 500, FACE_SIZE, mark, DIRPATH))
	if (!get_mark_sample (&sample, 1000, FACE_SIZE, mark, n_mark,DIRPATH))
		goto get_sp_err;
	ada_set_haar (&handles, ADA_ASYM_IMP, ADA_GA);
	int m = sample.m * TRAIN_SET_PERCENT;
	if (!cas_train (&cascade, DET_RATE, FP_RATE, MAX_FP_RATE, sample.m - m,
			m, FACE_SIZE, sample.X, sample.X2, sample.Y, &handles)) {
		fprintf(stderr, "Training Error.\n");
		goto train_err;
	}

	FILE * file = fopen (MODEL_PATH, "wb");
	cas_write (&cascade, file, &handles);
	fclose (file);
	cas_free (&cascade, &handles);
	free_sample (&sample);
	fclose (mark);
	return 0;

train_err:
	free_sample (&sample);
get_sp_err:
	fclose (mark);
	exit(1);
}
