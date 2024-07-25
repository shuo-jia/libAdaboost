// 决策树桩子类
// 构建 Haar 特征的决策树桩
#ifndef HAAR_STUMP_H
#define HAAR_STUMP_H
#include <stdio.h>
#include <stdbool.h>
#include "stump_base.h"

/**
 * \file haar_stump.h
 * \brief 使用 Haar 特征的决策树桩（函数声明）
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */
/*******************************************************************************
* 				   类型定义
 * 名称约定：后缀 cf 表示 confident
 * stump: 连续型变量、不带置信度的决策树桩。
 * 	  此类决策树桩接受连续型变量，并选择其中一个特征确定
 * 	  一个阈值来将样本划分为正例或负例；输出 -1 或 +1。
 * stumpcf: 连续型变量、带有置信度的决策树桩。
 * 	    此类决策树桩接受连续型变量，并选择其中一个特征确定
 * 	    一个阈值来将样本划分为正例或负例；输出一实数。
 *******************************************************************************/
/// Haar 特征类型
enum haar_feattype {
	FEAT_START,		///< 标签的起始标志，用于循环的起始条件
	LEFT_RIGHT,		///< 由左、右两矩形构成的哈尔特征
	UP_DOWN,		///< 由上、下两矩形构成的哈尔特征
	TRIPLE,			///< 由并排的三个矩形构成的哈尔特征
	QUAD,			///< 由 4 个矩形构成的哈尔特征
	FEAT_END,		///< 标签的起始标志，用于循环的结束条件
};

/// Haar 特征的描述
struct haar_feature {
	enum haar_feattype type;	///< 4 种哈尔特征之一
	imgsz_t start_x;		///< 第一个矩形左上角的横坐标
					/**<（该像素不包含在矩形中）*/
	imgsz_t start_y;		///< 第一个矩形左上角的纵坐标
					/**<（该像素不包含在矩形中）*/
	imgsz_t width;			///< 第一个矩形的宽度
	imgsz_t height;			///< 第一个矩形的高度
};

/// Haar 决策树桩结构体（连续型变量、不带置信度）
struct haar_stump {
	struct haar_feature feature;	///< feature: 表示用于分类的特征
	struct cstump_base base;	///< base: 基类结构体
};

/// Haar 决策树桩结构体（连续型变量、带置信度）
struct haar_stump_cf {
	struct haar_feature feature;	///< feature: 表示用于分类的特征
	struct cstump_cf_base base;	///< base: 基类结构体
};

/*******************************************************************************
* 				   函数声明
*******************************************************************************/
/**
 * \brief haar_stump 获取分类结果（用于图片），分类结果为 -1 或 +1
 * \details \copydetails wl_h_haar_fn
 */
label_t haar_stump_h(const void *stump, imgsz_t h, imgsz_t w, imgsz_t wid,
		     const sample_t x[h][wid], const sample_t x2[h][wid],
		     flt_t scale);

/**
 * \brief haar_stump_cf 获取分类结果（用于图片），分类结果为置信度
 * \details \copydetails wl_h_haar_fn
 */
flt_t haar_stump_cf_h(const void *stump, imgsz_t h, imgsz_t w, imgsz_t wid,
		      const sample_t x[h][wid], const sample_t x2[h][wid],
		      flt_t scale);

/**
 * \brief haar_stump 类型的训练
 * \details \copydetails wl_train_haar_fn
 */
bool haar_stump_train(void *stump, num_t m, imgsz_t h, imgsz_t w,
		      const sample_t * const X[], const sample_t * const X2[],
		      const label_t Y[], const flt_t D[]);

/**
 * \brief haar_stump_cf 类型的训练
 * \details \copydetails wl_train_haar_fn
 */
bool haar_stump_cf_train(void *stump, num_t m, imgsz_t h, imgsz_t w,
			 const sample_t * const X[],const sample_t * const X2[],
			 const label_t Y[], const flt_t D[]);

#endif
