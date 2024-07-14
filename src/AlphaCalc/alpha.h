#ifndef ALPHA_H
#define ALPHA_H
#include "boost_cfg.h"

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/*
 * 弱学习器系数 alpha 的计算方法
 * vals: Y[i] * h(x[i]) 构成的数组
 * vals_len: vals 数组长度
 * m: 样本数量，这里默认为等同于 vals_len
 * label: 标签集，这里默认弃置不用
 * D: 样本概率分布数组
 */
// 计算 alpha 的值，使用近似方法
flt_t alpha_approx(const flt_t vals[], num_t vals_len, num_t m,
		   const void *label, const flt_t D[]);

// 计算 alpha 的值，令 alpha 恒为 1
flt_t alpha_eq_1(const flt_t vals[], num_t vals_len, num_t m,
		 const void *label, const flt_t D[]);

// 计算 alpha 的值，使用简单的牛顿二分法（数值方法）
flt_t alpha_newton(const flt_t vals[], num_t vals_len, num_t m,
		   const void *label, const flt_t D[]);

#endif
