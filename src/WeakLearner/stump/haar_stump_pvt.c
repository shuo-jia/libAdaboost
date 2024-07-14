#include <math.h>
#include "haar_stump_pvt.h"

/**
 * \file haar_stump_pvt.c
 * \brief haar_stump 的私有部分（函数实现）
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */
/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
void init_feature(void *feature, const void *samples)
{
	struct haar_feature *ft = feature;
	ft->type = FEAT_START + 1;
	ft->start_x = 0;
	ft->start_y = 0;
	ft->width = 1;
	ft->height = 1;
}

void *next_feature(void *feature, const void *samples)
{
	struct haar_feature *ft = feature;
	const struct sp_wrap *sp = samples;
	if (++ft->start_x < sp->w - rect_x_ct[ft->type] * ft->width)
		return feature;
	ft->start_x = 0;
	if (++ft->start_y < sp->h - rect_y_ct[ft->type] * ft->height)
		return feature;
	ft->start_y = 0;
	if (++ft->width <= (sp->w - 1) / rect_x_ct[ft->type])
		return feature;
	ft->width = 1;
	if (++ft->height <= (sp->h - 1) / rect_y_ct[ft->type])
		return feature;
	ft->height = 1;
	if (++ft->type < FEAT_END)
		return feature;
	return NULL;
}

void update_opt(void *opt, const void *feature)
{
	struct haar_feature *ptr_opt = opt;
	const struct haar_feature *ptr_ft = feature;
	*ptr_opt = *ptr_ft;
}

const sample_t *get_vals_raw(num_t m, const void *samples, const void *feature)
{
	const struct sp_wrap *sp = samples;

	for (num_t i = 0; i < m; ++i)
		sp->vector[i] = get_value(feature, sp->h, sp->w, sp->w,
					  (void *)sp->X[i], (void *)sp->X2[i],
					  1);

	return sp->vector;
}

sample_t get_value(const struct haar_feature *feat, imgsz_t h, imgsz_t w,
		   imgsz_t wid, const sample_t x[h][wid],
		   const sample_t x2[h][wid], flt_t scale)
{
	h -= 1;			// 第一行弃置不用
	w -= 1;			// 第一列弃置不用
	flt_t std_dev;		// 标准差
	std_dev = (flt_t) (x[h][w] - x[h][0] - x[0][w] + x[0][0]) / (h * w);
	std_dev *= -std_dev;	// 计算均值的平方
	std_dev +=
	    (flt_t) (x2[h][w] - x2[h][0] - x2[0][w] + x2[0][0]) / (h * w);
	// 方差为 0，从现实意义的角度来说，哈尔特征为 0（标准差用于消除光照差异）
	if (std_dev == 0)
		return 0;
	std_dev = sqrt(std_dev);	// 计算标准差

	flt_t start_x = feat->start_x * scale;	// 左上角横坐标
	flt_t start_y = feat->start_y * scale;	// 左上角纵坐标
	w = feat->width * scale;	// 单个矩形的宽度
	h = feat->height * scale;	// 单个矩形的高度
	imgsz_t i[3] = { start_y, start_y + h, start_y + 2 * h };
	imgsz_t j[4] =
	    { start_x, start_x + w, start_x + 2 * w, start_x + 3 * w };
	switch (feat->type) {
	case LEFT_RIGHT:
		return (x[i[1]][j[2]] - x[i[0]][j[2]] - 2 * x[i[1]][j[1]]
			+ 2 * x[i[0]][j[1]] + x[i[1]][j[0]] - x[i[0]][j[0]]) /
		    std_dev / scale / scale;
	case UP_DOWN:
		return (2 * x[i[1]][j[1]] - x[i[0]][j[1]] - 2 * x[i[1]][j[0]]
			+ x[i[0]][j[0]] - x[i[2]][j[1]] + x[i[2]][j[0]]) /
		    std_dev / scale / scale;
	case TRIPLE:
		return (2 * x[i[1]][j[2]] - 2 * x[i[0]][j[2]] -
			2 * x[i[1]][j[1]]
			+ 2 * x[i[0]][j[1]] + x[i[1]][j[0]] - x[i[0]][j[0]]
			- x[i[1]][j[3]] +
			x[i[0]][j[3]]) / std_dev / scale / scale;
	case QUAD:
		return (2 * x[i[1]][j[2]] - x[i[0]][j[2]] - 4 * x[i[1]][j[1]]
			+ 2 * x[i[0]][j[1]] + 2 * x[i[1]][j[0]] - x[i[0]][j[0]]
			- x[i[2]][j[2]] - x[i[2]][j[0]] + 2 * x[i[2]][j[1]]) /
		    std_dev / scale / scale;
	default:
		return NAN;
	}
}
