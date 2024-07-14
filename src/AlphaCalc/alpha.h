#ifndef ALPHA_H
#define ALPHA_H
#include "boost_cfg.h"
/**
 * \file alpha.h
 * \brief 函数声明，给出一组弱学习器系数的计算方法
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-15
 */

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/**
 * \brief 计算弱学习器系数 alpha 的值，使用近似方法
 * \param[in] vals: Y[i] * h(x[i]) 构成的数组
 * \param[in] vals_len: vals 数组长度
 * \param[in] m: 样本数量，这里默认为等同于 vals_len
 * \param[in] label: 标签集，这里默认弃置不用
 * \param[in] D: 样本概率分布数组
 * \return 返回弱学习器系数
 */
flt_t alpha_approx(const flt_t vals[], num_t vals_len, num_t m,
		   const void *label, const flt_t D[]);

/**
 * \brief 计算弱学习器系数 alpha 的值，alpha 恒为 1
 * \details \copydetails alpha_approx()
 */
flt_t alpha_eq_1(const flt_t vals[], num_t vals_len, num_t m,
		 const void *label, const flt_t D[]);

/**
 * \brief 计算弱学习器系数 alpha 的值，使用简单的牛顿二分法（数值方法）
 * \details \copydetails alpha_approx()
 */
flt_t alpha_newton(const flt_t vals[], num_t vals_len, num_t m,
		   const void *label, const flt_t D[]);

#endif
