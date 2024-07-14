#ifndef HAAR_ASYM_ADA_H
#define HAAR_ASYM_ADA_H
#include "haar_base.h"
/**
 * \file haar_asym_ada.h
 * \brief 基于哈尔特征的 Adaboost 分类器函数声明--子类（增加训练方法）。
 * 	主要实现 Paul Viola, Michael Jones 于 2002 年给出的方法
 * 	（Fast and Robust Classification using Asymmetric AdaBoost and a
 * 	Detector Cascade.）
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/**
 * \brief haar_adaboost 训练方法，将 alpha 合并到弱学习器 h 中，即 alpha 恒为 1；
 * 	且采用非对称的损失函数
 * \details \copydetails haar_ada_train_fn
 */
bool haar_ada_asym_train(struct haar_adaboost *adaboost, flt_t * d, flt_t * f,
			 num_t l, num_t m, imgsz_t h, imgsz_t w,
			 const sample_t * X[], const sample_t * X2[],
			 const label_t Y[], const struct wl_handles *handles);

/**
 * \brief haar_adaboost 训练方法，将 alpha 合并到弱学习器 h 中，即 alpha 恒为 1；
 * 	且采用非对称的损失函数；此外，改进了概率分布的更新方式
 * \details \copydetails haar_ada_train_fn
 */
bool haar_ada_asym_imp_train(struct haar_adaboost *adaboost, flt_t * d,
			     flt_t * f, num_t l, num_t m, imgsz_t h, imgsz_t w,
			     const sample_t * X[], const sample_t * X2[],
			     const label_t Y[],
			     const struct wl_handles *handles);

/**
 * \brief 获取分类结果，弱学习器系数并入弱学习器
 * \details \copydetails haar_ada_h_fn
 */
flt_t haar_ada_fold_h(const struct haar_adaboost *adaboost, imgsz_t h,
		      imgsz_t w, imgsz_t wid, const sample_t x[h][wid],
		      const sample_t x2[h][wid], flt_t scale,
		      const struct wl_handles *handles);

#endif
