#ifndef MVEC_HLOSS_H
#define MVEC_HLOSS_H
#include "mvec_adaboost.h"
/**
 * \file mvec_hloss.h
 * \brief mvec_adaboost 训练方法声明：使用汉明损失（Hamming loss）进行训练
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/**
 * \brief mvec_adaboost 训练方法，使用汉明损失
 * 	（近似方法，利用不等式缩放获取系数 alpha）
 * \details \copydetails mvec_ada_train_fn
 */
bool mvec_ada_approx_train(struct mvec_adaboost *adaboost, turn_t T, num_t m,
			   dim_t n, const sample_t X[m][n], const mlabel_t Y[],
			   bool cache_on, const struct wl_handles *handles);

/**
 * \brief mvec_adaboost 训练方法，使用汉明损失
 * 	（将 alpha 合并到弱学习器 h 中，即 alpha 恒为 1）
 * \details \copydetails mvec_ada_train_fn
 */
bool mvec_ada_fold_train(struct mvec_adaboost *adaboost, turn_t T, num_t m,
			 dim_t n, const sample_t X[m][n], const mlabel_t Y[],
			 bool cache_on, const struct wl_handles *handles);

/**
 * \brief mvec_adaboost 训练方法，使用汉明损失
 * 	（数值方法，应用牛顿二分法求系数 alpha）
 * \details \copydetails mvec_ada_train_fn
 */
bool mvec_ada_newton_train(struct mvec_adaboost *adaboost, turn_t T, num_t m,
			   dim_t n, const sample_t X[m][n], const mlabel_t Y[],
			   bool cache_on, const struct wl_handles *handles);

/**
 * \brief 获取 mvec_adaboost 分类结果
 * 	（弱学习器系数不并入弱学习器）
 * \details \copydetails mvec_ada_h_fn
 */
mlabel_t mvec_ada_h(const struct mvec_adaboost *adaboost, const sample_t x[],
		    dim_t n, const struct wl_handles *handles);

/**
 * \brief 获取 mvec_adaboost 分类结果
 * 	（弱学习器系数并入弱学习器）
 * \details \copydetails mvec_ada_h_fn
 */
mlabel_t mvec_ada_fold_h(const struct mvec_adaboost *adaboost,
			 const sample_t x[], dim_t n,
			 const struct wl_handles *handles);

#endif
