#include "constant.h"
/**
 * \file constant.c
 * \brief 常数弱学习器的具体实现，弱学习器为一固定常数，仅用作测试
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-13
 */

/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
label_t constant_h(const void *model, const sample_t x[], dim_t n)
{
	const constant *m = (const constant *)model;
	return (*m >= 0) ? 1 : -1;
}

bool constant_train(void *model, num_t m, dim_t n, const sample_t X[m][n],
		    const label_t Y[], const flt_t D[], const void *cache)
{
	constant *c = (constant *) model;
	*c = 0;
	for (num_t i = 0; i < m; ++i)
		*c += D[i] * Y[i];

	return true;
}
