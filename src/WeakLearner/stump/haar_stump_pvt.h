#ifndef HAAR_STUMP_PVT_H
#define HAAR_STUMP_PVT_H
#include "haar_stump.h"

/**
 * \file haar_stump_pvt.h
 * \brief haar_stump 的私有部分（函数声明）
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */
/*******************************************************************************
* 				  全局的常量
*******************************************************************************/
/// 各类型的哈尔特征横向矩形个数
const static char rect_x_ct[] = {
	[LEFT_RIGHT] = 2,
	[UP_DOWN] = 1,
	[TRIPLE] = 3,
	[QUAD] = 2,
};

/// 各类型的哈尔特征纵向矩形个数
const static char rect_y_ct[] = {
	[LEFT_RIGHT] = 1,
	[UP_DOWN] = 2,
	[TRIPLE] = 1,
	[QUAD] = 2,
};

/*******************************************************************************
* 				   类型定义
*******************************************************************************/
/// 样本集结构体
struct sp_wrap {
	const sample_t **X;	///< 积分图指针数组
	const sample_t **X2;	///< 像素值平方的积分图指针数组
	imgsz_t h;		///< 训练图像高度
	imgsz_t w;		///< 训练图像宽度
	sample_t *vector;	///< 保存样本集在某一特征上的取值
};

/*******************************************************************************
 * 				函数声明（私有）
 ******************************************************************************/
/// 初始化特征变量
void init_feature(void *feature, const void *samples);
/// 更新特征变量
void *next_feature(void *feature, const void *samples);
/// 更新最优特征
void update_opt(void *opt, const void *feature);
/// 获取特征数组
const sample_t *get_vals_raw(num_t m, const void *samples, const void *feature);

 /**
 * \brief 计算样本在指定特征上的取值
 * \param[in] feat  指定特征，函数将返回样本在该特征上的取值
 * \param[in] h     截取的图像高度
 * \param[in] w     截取的图像宽度
 * \param[in] wid   原图像宽度
 * \param[in] x     积分图（二维数组，h * w 大小）
 * \param[in] x2    灰度值平方的积分图（二维数组，h * w 大小）
 * \param[in] scale 缩放比例，即检测图像尺寸：训练图像尺寸
 * \return 返回样本在特征 feat 上的取值
 */
sample_t get_value(const struct haar_feature *feat, imgsz_t h, imgsz_t w,
		   imgsz_t wid, const sample_t x[h][wid],
		   const sample_t x2[h][wid], flt_t scale);

#endif
