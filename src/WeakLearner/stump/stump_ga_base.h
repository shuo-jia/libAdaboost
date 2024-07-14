#ifndef STUMP_GA_BASE_H
#define STUMP_GA_BASE_H
#include "stump_base.h"
/**
 * \file stump_ga_base.h
 * \brief stump_base 训练方法重载，使用进化算法寻优（函数声明）
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-13
 */

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/// 回调函数类型：对单个个体进行初始化
typedef void (*ga_init_fn)(void *individual, const void *samples);
/// 回调函数类型：对两个父代个体交叉，产生子代个体
typedef void (*ga_crossover_fn)(void *child, const void *parent1,
				const void *parent2, const void *samples);
/// 回调函数类型：对单个个体进行变异操作
typedef void (*ga_mutate_fn)(void *individual, const void *samples);

/// 使用进化算法寻找决策树桩划分属性的回调函数集
struct stump_ga_handles {
	num_t gen;			///< 迭代次数
	num_t m;			///< 种群数量
	flt_t p_c;			///< 交叉概率
	flt_t p_m;			///< 变异概率
	ga_init_fn init;		///< 初始化函数
	ga_crossover_fn crossover;	///< 交叉函数
	ga_mutate_fn mutate;		///< 变异函数
	st_get_vals_fn get_vals;	///< 回调函数，返回样本集在某特征上的取值
	st_update_opt_fn update_opt;	///< 回调函数，更新划分属性
};

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/**
 * \brief 使用进化算法获取决策树桩的划分属性，并保存到决策树桩基类（不带置信度）
 *      （不保证得到最优划分属性）
 * \param[out] stump  未初始化的决策树桩
 * \param[out] opt    用于保存最优划分属性的变量地址
 * \param[in] ft_size 单个属性变量的长度（字节），即特征类型的长度
 * \param[in] m       样本数量
 * \param[in] samples 指向样本集的指针
 * \param[in] D       样本集的概率分布
 * \param[in] handles 已初始化的回调函数集合
 * \return 成功则返回真，失败则返回假
 */
bool cstump_ga(struct cstump_base *stump, void *opt, size_t ft_size,
	       num_t m, const void *samples, const label_t * label,
	       const flt_t * D, const struct stump_ga_handles *handles);

/**
 * \brief 使用进化算法获取决策树桩的划分属性，并保存到决策树桩基类（带置信度）
 *      （不保证得到最优划分属性）
 * \details \copydetails cstump_ga()
 */
bool cstump_cf_ga(struct cstump_cf_base *stump, void *opt, size_t ft_size,
		  num_t m, const void *samples, const label_t * label,
		  const flt_t * D, const struct stump_ga_handles *handles);

#endif
