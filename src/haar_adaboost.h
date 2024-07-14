#ifndef HAAR_ADABOOST_H
#define HAAR_ADABOOST_H
#include "haar_base.h"
/**
 * \file haar_adaboost.h
 * \brief 基于哈尔特征的 Adaboost 分类器函数声明--子类（增加训练方法）。
 *	主要实现 Paul Viola, Michael Jones 于 2001 年给出的方法
 *	（Robust Real-time Object Detection）
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/**
 * \brief haar_adaboost 训练方法，使用近似方法，利用不等式缩放获取系数 alpha
 * \details \copydetails haar_ada_train_fn
 */
bool haar_ada_approx_train(struct haar_adaboost *adaboost, flt_t * d,
			   flt_t * f, num_t l, num_t m, imgsz_t h, imgsz_t w,
			   const sample_t * X[], const sample_t * X2[],
			   const label_t Y[], const struct wl_handles *handles);

/**
 * \brief haar_adaboost 训练方法，数值方法，应用牛顿二分法求系数 alpha
 * \details \copydetails haar_ada_train_fn
 */
bool haar_ada_newton_train(struct haar_adaboost *adaboost, flt_t * d,
			   flt_t * f, num_t l, num_t m, imgsz_t h, imgsz_t w,
			   const sample_t * X[], const sample_t * X2[],
			   const label_t Y[], const struct wl_handles *handles);

/**
 * \brief 获取分类结果，弱学习器系数不并入弱学习器
 * \details \copydetails haar_ada_h_fn
 */
flt_t haar_ada_h(const struct haar_adaboost *adaboost, imgsz_t h, imgsz_t w,
		 imgsz_t wid, const sample_t x[h][wid],
		 const sample_t x2[h][wid], flt_t scale,
		 const struct wl_handles *handles);

#endif
