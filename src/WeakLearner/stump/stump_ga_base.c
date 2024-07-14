#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "stump_ga_base.h"
#include "stump_base_pvt.h"
/**
 * \file stump_ga_base.c
 * \brief stump_base 训练方法重载，使用进化算法寻优（函数实现）
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-13
 */

/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
/// 种群初始化
static void init_pop(num_t m, size_t size, unsigned char population[][size],
		     const void *samples, const struct stump_ga_handles *hl);
/// 选择，采用二元锦标赛（父代个体、子代个体一同进入筛选，筛选出的个体保存到父代中
static void select_pop(num_t m, size_t size, unsigned char population[][size],
		       flt_t vals_p[], const unsigned char children[][size],
		       const flt_t vals_c[]);
/// 交叉
static void crossover(num_t m, size_t size, unsigned char children[][size],
		      unsigned char population[][size], const void *samples,
		      const struct stump_ga_handles *hl);
/// 变异
static void mutation(num_t m, size_t size, unsigned char children[][size],
		     const void *samples, const struct stump_ga_handles *hl);
/// 计算种群适应值
static void fit_val(flt_t vals[], num_t sp_m, const void *samples,
		    const label_t * label, const flt_t * D, num_t m,
		    size_t size, const unsigned char population[][size],
		    const struct stump_ga_handles *hl);

/// 获取数组最小值的索引
static num_t argmin(flt_t vals[], num_t m);

/// 进化算法框架。成功则返回真，失败返回假
static bool ga(void *opt, size_t ft_size, num_t m, const void *samples,
	       const label_t * label, const flt_t * D,
	       const struct stump_ga_handles *handles);

/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
bool cstump_ga(struct cstump_base *stump, void *opt, size_t ft_size,
	       num_t m, const void *samples, const label_t * label,
	       const flt_t * D, const struct stump_ga_handles *handles)
{
	if (!ga(opt, ft_size, m, samples, label, D, handles))
		return false;

	struct cstump_segment seg;
	struct stump_opt_handles opt_hl = {
		.get_vals.raw = handles->get_vals,
	};
	cstump_raw_get_z(&seg, opt, m, samples, label, D, &opt_hl);
	cstump_update(stump, &seg);
	return true;
}

bool cstump_cf_ga(struct cstump_cf_base *stump, void *opt, size_t ft_size,
		  num_t m, const void *samples, const label_t * label,
		  const flt_t * D, const struct stump_ga_handles *handles)
{
	if (!ga(opt, ft_size, m, samples, label, D, handles))
		return false;

	struct cstump_segment seg;
	struct stump_opt_handles opt_hl = {
		.get_vals.raw = handles->get_vals,
	};
	cstump_raw_get_z(&seg, opt, m, samples, label, D, &opt_hl);
	cstump_cf_update(stump, &seg);
	return true;
}

/*******************************************************************************
 * 				  静态函数实现
 ******************************************************************************/
// 种群初始化
void init_pop(num_t m, size_t size, unsigned char population[][size],
	      const void *samples, const struct stump_ga_handles *hl)
{
	for (num_t i = 0; i < m; ++i)
		hl->init(population[i], samples);
}

// 选择，采用二元锦标赛
void select_pop(num_t m, size_t size, unsigned char population[][size],
		flt_t vals_p[], const unsigned char children[][size],
		const flt_t vals_c[])
{
	// 二元锦标赛
	for (num_t i = 0; i < m; ++i)
		if (vals_p[i] > vals_c[i]) {
			memcpy(population[i], children[i], size);
			vals_p[i] = vals_c[i];
		}
}

// 交叉
void crossover(num_t m, size_t size, unsigned char children[][size],
	       unsigned char population[][size], const void *samples,
	       const struct stump_ga_handles *hl)
{
	num_t i, j;
	num_t ids[m];
	for (i = 0; i < m; ++i)
		ids[i] = i;

	for (i = 0; i < m; ++i) {
		if (rand() > hl->p_c * RAND_MAX) {
			memcpy(children[i], population[i], size);
			continue;
		}
		j = rand() % (m - i);
		hl->crossover(children[i], population[i], population[ids[j]],
			      samples);
		ids[j] = ids[m - i - 1];
	}
}

// 变异
void mutation(num_t m, size_t size, unsigned char children[][size],
	      const void *samples, const struct stump_ga_handles *hl)
{
	const flt_t step = 5;

	for (num_t i = 0; i < m; ++i) {
		if (rand() > hl->p_m * RAND_MAX)
			continue;
		hl->mutate(children[i], samples);
	}
}

// 计算种群适应值
void fit_val(flt_t vals[], num_t sp_m, const void *samples,
	     const label_t * label, const flt_t * D, num_t m, size_t size,
	     const unsigned char population[][size],
	     const struct stump_ga_handles *hl)
{
	struct stump_opt_handles opt_hl = {
		.get_vals.raw = hl->get_vals,
	};
	struct cstump_segment seg;
	for (num_t i = 0; i < m; ++i) {
		cstump_raw_get_z(&seg, population[i], sp_m, samples, label, D,
				 &opt_hl);
		vals[i] = seg.z;
	}
}

// 获取数组最小值的索引
num_t argmin(flt_t vals[], num_t m)
{
	num_t min_id = 0;
	for (num_t i = 1; i < m; ++i)
		if (vals[i] < vals[min_id])
			min_id = i;

	return min_id;
}

bool ga(void *opt, size_t ft_size, num_t m, const void *samples,
	const label_t * label, const flt_t * D,
	const struct stump_ga_handles *handles)
{
	unsigned char (*population)[ft_size] = malloc(ft_size * handles->m);
	unsigned char (*children)[ft_size] = malloc(ft_size * handles->m);
	if (population == NULL || children == NULL) {
		free(population);
		free(children);
		return false;
	}

	flt_t vals_p[handles->m];	// 父代适应值
	flt_t vals_c[handles->m];	// 子代适应值
	init_pop(handles->m, ft_size, population, samples, handles);
	fit_val(vals_p, m, samples, label, D, handles->m, ft_size, population,
		handles);

	num_t id = argmin(vals_p, handles->m);
	flt_t min_val = vals_p[id];	// 历史最优值
	handles->update_opt(opt, population[id]);
	for (num_t t = 0; t < handles->gen; ++t) {
		crossover(handles->m, ft_size, children, population, samples,
			  handles);
		mutation(handles->m, ft_size, children, samples, handles);
		fit_val(vals_c, m, samples, label, D, handles->m, ft_size,
			children, handles);
		select_pop(handles->m, ft_size, population, vals_p, children,
			   vals_c);
		id = argmin(vals_p, handles->m);
		// 更新历史最优值
		if (min_val > vals_p[id]) {
			min_val = vals_p[id];
			handles->update_opt(opt, population[id]);
		}
	}

	free(population);
	free(children);
	return true;
}
