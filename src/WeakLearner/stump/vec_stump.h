#ifndef VEC_STUMP_H
#define VEC_STUMP_H
#include "stump_base.h"
/**
 * \file vec_stump.h
 * \brief 决策树桩子类，从样本特征构成的向量中构造弱学习器（函数声明）
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/// 决策树桩类型，用于样本向量训练，支持连续变量，不带置信度
struct vec_cstump {
	struct cstump_base base;	///< 决策树桩基类
	dim_t feature;			///< 决策树桩所使用的特征（样本向量的下标）
};

/// 决策树桩类型，用于样本向量训练，支持连续变量，带置信度
struct vec_cstump_cf {
	struct cstump_cf_base base;	///< 决策树桩基类
	dim_t feature;			///< 决策树桩所使用的特征（样本向量的下标）
};

/// 决策树桩类型，用于样本向量训练，支持离散变量，不带置信度
struct vec_dstump {
	struct dstump_base base;	///< 决策树桩基类
	dim_t feature;			///< 决策树桩所使用的特征（样本向量的下标）
};

/// 决策树桩类型，用于样本向量训练，支持离散变量，带置信度
struct vec_dstump_cf {
	struct dstump_cf_base base;	///< 决策树桩基类
	dim_t feature;			///< 决策树桩所使用的特征（样本向量的下标）
};

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/**
 * \brief vec_cstump 决策树桩训练
 * \details \copydetails wl_train_vec_fn
 */
bool vec_cstump_train(void *stump, num_t m, dim_t n, const sample_t X[m][n],
		      const label_t Y[], const flt_t D[], const void *cache);

/**
 * \brief vec_cstump_cf 决策树桩训练
 * \details \copydetails wl_train_vec_fn
 */
bool vec_cstump_cf_train(void *stump, num_t m, dim_t n, const sample_t X[m][n],
			 const label_t Y[], const flt_t D[], const void *cache);

/**
 * \brief vec_dstump 决策树桩训练
 * \details \copydetails wl_train_vec_fn
 */
bool vec_dstump_train(void *stump, num_t m, dim_t n, const sample_t X[m][n],
		      const label_t Y[], const flt_t D[], const void *cache);

/**
 * \brief vec_dstump_cf 决策树桩训练
 * \details \copydetails wl_train_vec_fn
 */
bool vec_dstump_cf_train(void *stump, num_t m, dim_t n, const sample_t X[m][n],
			 const label_t Y[], const flt_t D[], const void *cache);

/**
 * \brief 获取 vec_cstump 决策树桩弱学习器的分类结果
 * \details \copydetails wl_h_vec_fn
 */
label_t vec_cstump_h(const void *stump, const sample_t x[], dim_t n);

/**
 * \brief 获取 vec_cstump_cf 决策树桩弱学习器的分类结果
 * \details \copydetails wl_h_vec_cf_fn
 */
flt_t vec_cstump_cf_h(const void *stump, const sample_t x[], dim_t n);

/**
 * \brief 获取 vec_dstump 决策树桩弱学习器的分类结果
 * \details \copydetails wl_h_vec_fn
 */
label_t vec_dstump_h(const void *stump, const sample_t x[], dim_t n);

/**
 * \brief 获取 vec_dstump_cf 决策树桩弱学习器的分类结果
 * \details \copydetails wl_h_vec_cf_fn
 */
flt_t vec_dstump_cf_h(const void *stump, const sample_t x[], dim_t n);

/**
 * \brief 从文件中读取 vec_cstump 决策树桩
 * \details \copydetails wl_read_fn
 */
bool vec_cstump_read(void *stump, FILE * file);

/**
 * \brief 从文件中读取 vec_cstump_cf 决策树桩
 * \details \copydetails wl_read_fn
 */
bool vec_cstump_cf_read(void *stump, FILE * file);
/**
 * \brief 从文件中读取 vec_dstump 决策树桩
 * \details \copydetails wl_read_fn
 */
bool vec_dstump_read(void *stump, FILE * file);

/**
 * \brief 从文件中读取 vec_dstump_cf 决策树桩
 * \details \copydetails wl_read_fn
 */
bool vec_dstump_cf_read(void *stump, FILE * file);

/**
 * \brief 将 vec_cstump 决策树桩写入到文件
 * \details \copydetails wl_write_fn
 */
bool vec_cstump_write(const void *stump, FILE * file);

/**
 * \brief 将 vec_cstump_cf 决策树桩写入到文件
 * \details \copydetails wl_write_fn
 */
bool vec_cstump_cf_write(const void *stump, FILE * file);

/**
 * \brief 将 vec_dstump 决策树桩写入到文件
 * \details \copydetails wl_write_fn
 */
bool vec_dstump_write(const void *stump, FILE * file);

/**
 * \brief 将 vec_dstump_cf 决策树桩写入到文件
 * \details \copydetails wl_write_fn
 */
bool vec_dstump_cf_write(const void *stump, FILE * file);

/**
 * \brief 对 vec_dstump 决策树桩进行深度复制，但不包括缓存字段
 * \details \copydetails wl_copy_fn
 */
void *vec_dstump_copy(void *dst, const void *src);

/**
 * \brief 对 vec_dstump_cf 决策树桩进行深度复制，但不包括缓存字段
 * \details \copydetails wl_copy_fn
 */
void *vec_dstump_cf_copy(void *dst, const void *src);

/**
 * \brief 内存释放方法，用于 vec_dstump 类型
 * \details \copydetails wl_free_fn
 */
void vec_dstump_free(void *stump);

/**
 * \brief 内存释放方法，用于 vec_dstump_cf 类型
 * \details \copydetails wl_free_fn
 */
void vec_dstump_cf_free(void *stump);

/**
 * \brief 创建新缓存（使用完毕后用 free() 释放）
 * \param[in] m 样本数量
 * \param[in] n 样本特征数量
 * \param[in] X 样本集
 * \return 成功则返回缓存指针，失败返回 NULL
 */
void *vec_new_cache(num_t m, dim_t n, const sample_t X[m][n]);

#endif
