#include <math.h>
#include <float.h>
#include <stdbool.h>
#include "alpha.h"

/*******************************************************************************
 * 				   宏常量定义
 ******************************************************************************/
// 落于闭区间 [-ZERO_REGION, ZERO_REGION] 的实数被认为是 0
#define ZERO_REGION 1E-6

/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
/*
 * 用于牛顿二分法的导函数计算
 * r_arr: r[i] = D[i] * vals[i], i = 1, 2, ..., m
 * vals: 数组，保存有 label[i] * h(sample[i]) 的值，i = 1, 2, ..., m
 * m: 样本数量
 * alpha: 当前的弱学习器系数取值
 */
static flt_t derived_fun(const flt_t r_arr[], const flt_t vals[],
			 num_t m, flt_t alpha);

/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
flt_t alpha_approx(const flt_t vals[], num_t vals_len, num_t m,
		   const void *label, const flt_t D[])
{
	flt_t r = 0;
	for (num_t i = 0; i < m; ++i)
		r += D[i] * vals[i];
	return log((1 + r) / (1 - r)) / 2.0;
}

flt_t alpha_eq_1(const flt_t vals[], num_t vals_len, num_t m,
		 const void *label, const flt_t D[])
{
	return 1;
}

flt_t alpha_newton(const flt_t vals[], num_t vals_len, num_t m,
		   const void *label, const flt_t D[])
{
	// 区间左端, 区间右端，区间中点
	flt_t lb, ub, mid;
	flt_t lb_val, mid_val;
	// v_m[0] 依次表示 vals < 0 的最大值、最小值
	// v_m[1] 依次表示 vals > 0 的最大值、最小值
	flt_t v_m[2][2] = { { -DBL_MAX, 0 }, { 0, DBL_MAX } };

	flt_t r_arr[m];				// r[i] = D[i] * vals[i]
	// r_sum[0] 表示全体 r < 0 绝对值之和
	// r_sum[1] 表示全体 r > 0 之和
	flt_t r_sum[2] = { 0, 0 };
	bool p_or_n;

	for (num_t i = 0; i < m; ++i) {		// 数组初始化
		r_arr[i] = D[i] * vals[i];
		p_or_n = (bool)(vals[i] > 0);
		if (vals[i] > v_m[p_or_n][0])
			v_m[p_or_n][0] = vals[i];
		if (vals[i] < v_m[p_or_n][1])
			v_m[p_or_n][1] = vals[i];
		r_sum[p_or_n] += (r_arr[i] > 0) ? r_arr[i] : -r_arr[i];
	}
	// 范围 [lb, ub] 估计
	lb = log(r_sum[1] / r_sum[0]) / (v_m[1][0] - v_m[0][1]);
	ub = log(r_sum[1] / r_sum[0]) / (v_m[1][1] - v_m[0][0]);
	mid = (lb + ub) / 2;
	lb_val = derived_fun(r_arr, vals, m, lb);
	mid_val = derived_fun(r_arr, vals, m, mid);
	while (fabs(mid_val) > ZERO_REGION) {	// 二分法
		if (lb_val * mid_val <= 0) {
			ub = mid;
			mid = (mid + lb) / 2;
		} else {
			lb = mid;
			lb_val = mid_val;
			mid = (mid + ub) / 2;
		}
		mid_val = derived_fun(r_arr, vals, m, mid);
	}
	return mid;
}

/*******************************************************************************
 * 				  静态函数定义
 ******************************************************************************/
flt_t derived_fun(const flt_t r_arr[], const flt_t vals[], num_t m, flt_t alpha)
{
	flt_t result = 0;
	for (num_t i = 0; i < m; ++i)
		result -= r_arr[i] * exp(-alpha * vals[i]);
	return result;
}
