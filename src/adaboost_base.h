// 定义训练所需的回调函数集及训练方法
#ifndef ADABOOST_BASE_H
#define ADABOOST_BASE_H
#include <stdbool.h>
#include "boost_cfg.h"
/**
 * \file adaboost_base.h
 * \brief Adaboost 分类器基类定义及函数声明
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/// 单个弱学习器及其系数
struct ada_item {
	void *weaklearner;	///< 弱学习器的内存地址
	flt_t *alpha;		///< 加法模型下，弱学习器前的系数的内存地址
	bool status;		///< 执行操作的状态。当成功获取下一轮弱学
				/**< 习器时为真，否则为假 */
};

/// 返回值定义
enum ada_result {
        ADA_SUCCESS,		///< 函数成功返回
        ADA_FAILURE,		///< 函数错误
        ADA_ALL_PASS, 		///< 全部样本分类成功
};

/// 回调函数类型：训练单个弱学习器的回调函数。成功则返回真，否则返回假
typedef bool (*ada_train_fn)(void *weaklearner, num_t m, const void *sample,
			     const void *label, const flt_t D[]);
/**
 * \brief 回调函数类型: 计算中间结果并保存到 vals 数组中
 * \param[out] vals       中间值数组。可用于后续其他回调函数使用
 * \param[in] vals_len    vals 数组长度
 * \param[in] weaklearner 已训练完成的弱学习器地址
 * \param[in] m           样本数量
 * \param[in] sample      样本集
 * \param[in] label       样本标签集
 * \param[in] D[]         样本概率分布数组
 * \return 遇到错误（如训练误差过大）返回 ADA_FAILURE，训练错误终止；
 * 	成功则返回 ADA_SUCCESS，训练继续；
 * 	返回 ADA_ALL_PASS 表示样本全部分类成功，训练过程将快速退出。
 */
typedef enum ada_result (*ada_vals_fn) (flt_t vals[], num_t vals_len,
					const void *weaklearner, num_t m,
					const void *sample, const void *label,
					const flt_t D[]);
/// 回调函数类型：返回弱学习器系数
typedef flt_t(*ada_alpha_fn) (const flt_t vals[], num_t vals_len, num_t m,
			      const void *label, const flt_t D[]);
/**
 * \brief 回调函数类型: 更新参数 item，获取下一轮训练的弱学习器及其系数指针。
 * \param[in, out] item     弱学习器及其系数构成的结构体（地址）
 * \param[in, out] adaboost Adaboost 学习器。必要时保存当前位置到 adaboost
 * \param[in] vals          中间值数组
 * \param[in] vals_len      中间值数组的长度
 * \return 当到达结束条件时，返回假；否则返回真。如果未达到结束条件但遇到错误，
 *      则设置 item->status 为假；否则置为真。
 */
typedef bool (*ada_next_fn)(struct ada_item * item, void *adaboost,
			    const flt_t vals[], num_t vals_len);

/// 回调函数类型：初始化概率分布数组，保存到数组 D 中。
typedef void (*ada_init_D_fn)(flt_t D[], num_t m, const void *label);

/// 更新概率分布数组，保存到数组 D 中
typedef void (*ada_update_D_fn)(flt_t D[], flt_t vals[], num_t vals_len,
				num_t m, const void *label, flt_t alpha);

/// 训练所用的回调函数集
struct ada_handles {
	num_t D_len;			///< 概率分布数组的元素数量
	num_t vals_len;			///< vals 数组长度
	ada_train_fn train;		///< 训练函数
	ada_vals_fn get_vals;		///< 中间值计算函数
	ada_alpha_fn get_alpha;		///< alpha 系数计算函数
	ada_next_fn next;		///< 下一弱学习器获取函数
	ada_init_D_fn init_D;		///< 分布概率初始化函数
	ada_update_D_fn update_D;	///< 分布概率函数更新函数
};

/*******************************************************************************
 * 				    函数原型
 ******************************************************************************/
/**
 * \brief 训练框架
 * \param[out] adaboost 表示 Adaboost 分类器的变量地址
 * \param[in] m         样本数量
 * \param[in] sample    样本集地址
 * \param[in] label     样本标签数组（长度为 m）
 * \param[in] handles   用于处理弱学习器的回调函数集
 * 返回值：成功执行返回 ADA_SUCCESS，出错返回 ADA_FAILURE，错误率为 0 导致的提
 * 	前退出返回 ADA_ALL_PASS
 */
enum ada_result ada_framework(void *adaboost, num_t m, const void *sample,
			      const void *label,
			      const struct ada_handles *handles);

#endif
