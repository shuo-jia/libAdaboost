#ifndef CONSTANT_H
#define CONSTANT_H
/**
 * \file constant.h
 * \brief 常数弱学习器的函数声明，弱学习器为一固定常数，仅用于测试
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-13
 */

#include <stdbool.h>
#include "boost_cfg.h"

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/**
 * \brief 接收样本向量，输出常数弱学习器的输出值，不带置信度
 *
 * \param[in] model 常数弱学习器的地址
 * \param[in] x     数组地址，表示一个样本向量
 * \param[in] n     数组长度
 *
 * \return 返回常数弱学习器的输出值，不带置信度
 */
label_t constant_h(const void *model, const sample_t x[], dim_t n);

/**
 * \brief 训练常数弱学习器
 *
 * \param[out] model     常数弱学习器的地址
 * \param[in] m          样本数量
 * \param[in] n          每个样本向量的维度
 * \param[in] X          样本向量集合，二维数组的每一行表示一个样本
 * \param[in] Y         样本标签数组，每个元素对应一个样本的正确分类
 * \param[in] D         样本概率分布数组
 * \param[in] cache      缓存指针，此处被忽略
 *
 * \return 
 */
bool constant_train(void *model, num_t m, dim_t n, const sample_t X[m][n],
		    const label_t Y[], const flt_t D[], const void *cache);

#endif
