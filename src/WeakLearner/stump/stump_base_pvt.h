#ifndef STUMP_BASE_PVT_H
#define STUMP_BASE_PVT_H
#include <stdlib.h>
#include "boost_cfg.h"
#include "stump_base.h"
/**
 * \file stump_base_pvt.h
 * \brief stump_base 的私有部分定义
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-13
 */
/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/// 用于指示 cstump 系列单个特征的最优划分值
struct cstump_segment {
	flt_t z;		///< 即 Z 值的二分之一
	flt_t value;		///< 划分值
	flt_t W[2][2];		///< 划分值对应的权重
	/**< W[0][*] 表示负例权重，W[1][*] 表示正例权重；
	 * W[*][0] 表示划分位置左侧的权重，W[*][1] 表示划分位置右侧的权重 */
};

/// 用于指示 dstump 系列单个特征的最优划分值
struct dstump_segment {
	flt_t z;		///< 即 Z 值的二分之一
	num_t len;		///< 划分值数量
	sample_t *value;	///< 划分值数组
	flt_t *W[2];		///< 划分值对应的权重数组
	/**< （W[0]表示负例权重，W[1]表示正例权重）*/
	flt_t g_W[2];		///< 全局正例权重
	/**<（位置0表示负例权重，位置1表示正例权重）*/
	char array[];		///< 柔性数组，value、W 实际存储位置
};

/*******************************************************************************
 * 				函数声明（私有）
 ******************************************************************************/
/**
 * \brief 为 cstump 系列类型获取最优划分值
 * \param[out] seg     用于保存最优划分值
 * \param[in]  feature 当前特征
 * \param[in]  m       样本数量
 * \param[in]  sample  样本集
 * \param[in] label    样本集标签
 * \param[in]  D       样本集的概率分布
 * \param[in]  handles 操作当前决策树桩的回调函数
 */
void cstump_raw_get_z(struct cstump_segment *seg, const void *feature,
		      num_t m, const void *samples, const label_t * label,
		      const flt_t D[], const struct stump_opt_handles *handles);

/**
 * \brief 为 cstump 系列类型已排序特征数组计算最优划分值
 * \details \copydetails cstump_raw_get_z()
 */
void cstump_sort_get_z(struct cstump_segment *seg, const void *feature,
		       num_t m, const void *samples, const label_t * label,
		       const flt_t D[],
		       const struct stump_opt_handles *handles);

/**
 * \brief 为 dstump 系列类型获取最优划分值
 *      （seg 各字段需初始化，seg->len 初始化为样本数量）
 * \details \copydetails cstump_raw_get_z()
 */
void dstump_raw_get_z(struct dstump_segment *seg, const void *feature,
		      num_t m, const void *samples, const label_t * label,
		      const flt_t D[], const struct stump_opt_handles *handles);

/**
 * \brief 为 dstump 系列类型已排序特征数组计算最优划分值
 *      （seg 各字段需初始化，seg->len 初始化为样本数量）
 * \details \copydetails cstump_raw_get_z()
 */
void dstump_sort_get_z(struct dstump_segment *seg, const void *feature,
		       num_t m, const void *samples, const label_t * label,
		       const flt_t D[],
		       const struct stump_opt_handles *handles);

/**
 * \brief 更新决策树桩基类 cstump_base 类型
 * \param[out] stump 指向决策树桩基类
 * \param[in]  seg   保存有划分值及权重信息的结构体
 */
void cstump_update(struct cstump_base *stump, const struct cstump_segment *seg);

/**
 * \brief 更新决策树桩基类 cstump_cf_base 类型
 * \details \copydetails cstump_update()
 */
void cstump_cf_update(struct cstump_cf_base *stump,
		      const struct cstump_segment *seg);

/**
 * \brief 更新决策树桩基类 dstump_base 类型
 * \details \copydetails cstump_update()
 */
void dstump_update(struct dstump_base *stump, const struct dstump_segment *seg);

/**
 * \brief 更新决策树桩基类 dstump_cf_base 类型
 * \details \copydetails cstump_update()
 */
void dstump_cf_update(struct dstump_cf_base *stump,
		      const struct dstump_segment *seg);

/*******************************************************************************
 * 				  内联函数定义
 ******************************************************************************/
/**
 * \brief 初始化 struct dstump_segment 结构体
 * \param[in] m 样本数量
 * \return 返回一个已初始化的结构体指针，可直接使用 free() 释放
 */
static inline struct dstump_segment *init_dseg(num_t m)
{
	struct dstump_segment *ptr = malloc(sizeof(struct dstump_segment) +
					    sizeof(flt_t) * m * 2 +
					    sizeof(sample_t) * m);
	if (ptr == NULL)
		return NULL;

	ptr->len = m;
	ptr->value = (sample_t *) ptr->array;
	ptr->W[0] = (flt_t *) (ptr->value + m);
	ptr->W[1] = ptr->W[0] + m;
	return ptr;
}

#endif
