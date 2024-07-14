#ifndef STUMP_BASE_H
#define STUMP_BASE_H
#include <stdio.h>
#include <stdbool.h>
#include "boost_cfg.h"

/**
 * \file stump_base.h
 * \brief 决策树桩的基类定义
 * 	按特征变量分为连续型、离散型；按输出值分为带置信度、不带置信度；
 * 	共计有 4 种类型（基类）
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-13
 */

/*******************************************************************************
 * 				    类型定义
 * 名称约定：前缀 c 表示 continuous，后缀 cf 表示 confident
 * cstump: 连续型变量、不带置信度的决策树桩。
 * 	   此类决策树桩接受连续型变量，并选择其中一个特征确定一个阈值来将样本划
 * 	   分为正例或负例；输出 -1 或 +1。
 * cstump_cf: 连续型变量、带有置信度的决策树桩。
 * 	     此类决策树桩接受连续型变量，并选择其中一个特征确定一个阈值来将样本
 * 	     划分为正例或负例；输出一实数。
 * dstump: 离散型变量、不带置信度的决策树桩。
 * 	   此类决策树桩接受离散型变量，并选择其中一个特征，给出每个可能的特征值
 * 	   所对应的分类。输出 -1 或 +1。
 * dstump_cf: 离散型变量、带有置信度的决策树桩。
 * 	     此类决策树桩接受离散型变量，并选择其中一个特征，给出每个可能的特征
 * 	     值所对应的分类。输出一实数。
 ******************************************************************************/

/// 支持连续型变量、不带置信度的决策树桩
struct cstump_base {
	flt_t value;		///< 用于分类的特征的分割值
	label_t output[2];	///< 第一个元素表示小于分割值时的输出值，
};				///< 第二个元素表示大于等于分割值时的输出值

/// 支持连续型变量、带有置信度的决策树桩
struct cstump_cf_base {
	flt_t value;		///< 用于分类的特征的分割值
	flt_t output[2];	///< 第一个元素表示小于分割值时的输出值，
};				///< 第二个元素表示大于等于分割值时的输出值

/// 支持离散型变量、不带置信度的决策树桩
struct dstump_base {
	sample_t *value;	///< 表示分类特征的取值（数组）
	label_t *output;	///< 对应于取值的输出值（-1 或 +1）
	label_t default_output;	///< 默认输出值
	num_t size;		///< value 数组元素数量
};

/// 支持离散型变量、带有置信度的决策树桩
struct dstump_cf_base {
	sample_t *value;	///< 表示分类特征的取值（数组）
	flt_t *output;		///< 对应于取值的输出值（实数值）
	flt_t default_output;	///< 默认输出值
	num_t size;		///< value 数组元素数量
};

/**
 * \brief 回调函数类型：初始化特征
 * \param[out] feature 表示一个特征，调用此函数后被初始化为第一个特征
 * \param[in] samples  指向用户定义的样本集
 */
typedef void (*st_init_feat_fn)(void *feature, const void *samples);

/**
 * \brief 回调函数类型：获取下一特征
 * \param[out] feature 每次调用时，都将 feature 指向的变量更新为下一个特征
 * \param[in] samples  指向用户定义的样本集
 * \return 当遍历完所有特征时，返回 NULL；否则返回非 NULL 值
 */
typedef void *(*st_next_feat_fn)(void *feature, const void *samples);

/**
 * \brief 回调函数类型：更新最优特征
 * \param[out] opt    指向最优特征，此函数执行后被更新
 * \param[in] feature 指向当前最优特征，此函数执行后被复制到 opt
 */
typedef void (*st_update_opt_fn)(void *opt, const void *feature);

/**
 * \brief 回调函数类型：返回一个特征数组，未排序
 * \param m       样本数量
 * \param samples 样本集
 * \param feature 当前特征
 * \return 返回一个 sample_t 型数组，各元素为样本集在 feature 上的取值，未排序
 */
typedef const sample_t *(*st_get_vals_fn)(num_t m, const void *samples,
					  const void *feature);

/**
 * \brief 返回排序后的特征数组索引
 * \param[in] m       样本数量
 * \param[in] samples 样本集
 * \param[in] feature 当前特征
 * \return 返回一个 num_t 型数组，各元素为样本的标号（从0开始），标号按样本在
 *      feature 上的取值从小到大排序
 */
typedef const num_t *(*st_get_sorted_fn)(num_t m, const void *samples,
					 const void *feature);

/// 获取决策树桩最优划分属性时，所使用的回调函数集合
struct stump_opt_handles {
	st_init_feat_fn init_feature;	///< 将feature指向的变量初始化为第一个特征
	st_next_feat_fn next_feature;	///< 获取下一特征
	st_update_opt_fn update_opt;	///< 设置当前最优特征
	struct {
		st_get_vals_fn raw;	///< 返回一个特征数组，未排序
		st_get_sorted_fn sort;	///< 返回排序后的特征数组索引
                                        /**< 可置为 NULL，此时将使用自带的排序方法 */
	} get_vals;			///< 结构体，获取样本集在当前特征上的取值
};

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/**
 * \brief 获取 cstump_base 类型决策树桩的最优划分属性
 * \param[out] stump  未初始化的决策树桩
 * \param[in] opt     用于保存最优划分属性的变量地址
 * \param[in] ft_size 单个属性变量的长度（字节），即特征类型的长度
 * \param[in] m       样本数量
 * \param[in] samples 指向样本集的指针
 * \param[in] D       样本集的概率分布
 * \param[in] handles 已初始化的回调函数集合
 * \return 训练成功则返回真，否则返回假
 */
bool cstump_opt(struct cstump_base *stump, void *opt, size_t ft_size,
		num_t m, const void *samples, const label_t * label,
		const flt_t * D, const struct stump_opt_handles *handles);

/**
 * \brief 获取 cstump_cf_base 类型决策树桩的最优划分属性
 * \details \copydetails cstump_opt()
 */
bool cstump_cf_opt(struct cstump_cf_base *stump, void *opt, size_t ft_size,
		   num_t m, const void *samples, const label_t * label,
		   const flt_t * D, const struct stump_opt_handles *handles);

/**
 * \brief 获取 dstump_base 类型决策树桩的最优划分属性
 * \details \copydetails cstump_opt()
 */
bool dstump_opt(struct dstump_base *stump, void *opt, size_t ft_size,
		num_t m, const void *samples, const label_t * label,
		const flt_t * D, const struct stump_opt_handles *handles);

/**
 * \brief 获取 dstump_cf_base 类型决策树桩的最优划分属性
 * \details \copydetails cstump_opt()
 */
bool dstump_cf_opt(struct dstump_cf_base *stump, void *opt, size_t ft_size,
		   num_t m, const void *samples, const label_t * label,
		   const flt_t * D, const struct stump_opt_handles *handles);

/**
 * \brief 为 struct dstump_base 类型申请空间，不初始化
 * \param[out] stump 未初始化的决策树桩
 * \param[in] n      离散取值的数量
 * \return 成功则返回真；失败则返回假
 */
bool dstump_alloc(struct dstump_base *stump, num_t n);

/**
 * \brief 为 struct dstump_cf_base 类型申请空间，不初始化
 * \details \copydetails dstump_alloc()
 */
bool dstump_cf_alloc(struct dstump_cf_base *stump, num_t n);

/**
 * \brief 为 struct dstump_base 类型重新申请空间（调用 realloc），尽量不改变数
 *      组内容
 * \param[in, out] stump 已初始化的决策树桩
 * \param[in] n          新的离散取值的数量
 * \return 成功则返回真；失败则返回假
 */
bool dstump_realloc(struct dstump_base *stump, num_t n);

/**
 * \brief 为 struct dstump_cf_base 类型重新申请空间（调用 realloc），尽量不改变
 *      数组内容
 * \details \copydetails dstump_realloc()
 */
bool dstump_cf_realloc(struct dstump_cf_base *stump, num_t n);

/**
 * \brief 为 struct dstump_base 类型释放空间
 * \param[in] stump 已分配内存的决策树桩
 */
void dstump_free(struct dstump_base *stump);

/**
 * \brief 为 struct dstump_cf_base 类型释放空间
 * \param[in] stump 已分配内存的决策树桩
 */
void dstump_cf_free(struct dstump_cf_base *stump);

/**
 * \brief 获取 struct dstump_base 类型决策树桩的分类结果
 * \param[in] stump 已初始化的决策树桩
 * \param[in] value 最优特征的取值
 * \return 分类结果（-1 或 +1）
 */
label_t dstump_h(const struct dstump_base *stump, sample_t value);

/**
 * \brief 获取 struct dstump_cf_base 类型决策树桩的分类结果
 * \param[in] stump 已初始化的决策树桩
 * \param[in] value 最优特征的取值
 * \return 分类结果的置信度
 */
flt_t dstump_cf_h(const struct dstump_cf_base *stump, sample_t value);

/**
 * \brief dstump_base 写入方法实现
 * \param[in] stump  已初始化的决策树桩
 * \param[out] file  用于保存 stump 的文件
 * \return 成功则返回真；失败则返回假
 */
bool dstump_write(const struct dstump_base *stump, FILE * file);

/**
 * \brief dstump_cf_base 写入方法实现
 * \details \copydetails dstump_write()
 */
bool dstump_cf_write(const struct dstump_cf_base *stump, FILE * file);

/**
 * \brief dstump_base 读取方法实现
 * \param[out] stump 未初始化的决策树桩
 * \param[in] file   保存了 stump 的文件，文件内容将被读入到 stump 中
 * \return 成功则返回真；失败则返回假
 */
// struct dstump_base 类型读取方法
bool dstump_read(struct dstump_base *stump, FILE * file);

/**
 * \brief dstump_cf_base 读取方法实现
 * \details \copydetails dstump_read()
 */
bool dstump_cf_read(struct dstump_cf_base *stump, FILE * file);

/**
 * \brief dstump_base 深度复制方法实现
 * \param[out] dst 未初始化的决策树桩
 * \param[in] src  已初始化的决策树桩，内容将被复制到 dst
 * \return 成功则 dst；失败则返回 NULL
 */
void *dstump_copy(struct dstump_base *dst, const struct dstump_base *src);

/**
 * \brief dstump_cf_base 深度复制方法
 * \details \copydetails dstump_copy()
 */
void *dstump_cf_copy(struct dstump_cf_base *dst,
		     const struct dstump_cf_base *src);

/**
 * \brief 比较两个 sample_t * 型变量（可用于 qsort() 函数）
 * \param[in] p1 sample_t * 型变量之一
 * \param[in] p2 sample_t * 型变量之二
 * \return *p1 > *p1，返回 1；*p1 < *p2，返回 -1；*p1 == *p2，返回 0
 */
int sample_cmp(const void *p1, const void *p2);

/**
 * \brief 比较两个 sample_t ** 型变量（可用于 qsort() 函数）
 * \param[in] p1 sample_t ** 型变量之一
 * \param[in] p2 sample_t ** 型变量之二
 * \return **p1 > **p1，返回 1；**p1 < **p2，返回 -1；**p1 == **p2，返回 0
 */
int sample_ptr_cmp(const void *p1, const void *p2);

/*******************************************************************************
 * 				  内联函数定义
 ******************************************************************************/
/**
 * \brief 获取 cstump_base 决策树桩的分类结果
 * \param[in] stump 已初始化的决策树桩
 * \param[in] value 特征的取值
 * \return 返回分类结果（-1 或 +1）
 */
static inline label_t cstump_h(const struct cstump_base *stump, sample_t value)
{
	return stump->output[value >= stump->value];
}

/**
 * \brief 获取 cstump_cf_base 决策树桩的分类结果
 * \param[in] stump 已初始化的决策树桩
 * \param[in] value 特征的取值
 * \return 返回分类结果，带置信度
 */
static inline flt_t cstump_cf_h(const struct cstump_cf_base *stump,
				sample_t value)
{
	return stump->output[value >= stump->value];
}

/**
 * \brief cstump_base 写入方法实现
 * \param[in] stump: 已初始化的决策树桩
 * \param[out] file: 用于保存 stump 的文件
 * \return 成功则返回真；失败则返回假
 */
static inline bool cstump_write(const struct cstump_base *stump, FILE * file)
{
	if (fwrite(stump, sizeof(struct cstump_base), 1, file) < 1)
		return false;
	return true;
}

/**
 * \brief cstump_cf_base 写入方法实现
 * \details \copydetails cstump_write()
 */
static inline bool cstump_cf_write(const struct cstump_cf_base *stump,
				   FILE * file)
{
	if (fwrite(stump, sizeof(struct cstump_cf_base), 1, file) < 1)
		return false;
	return true;
}

/**
 * \brief cstump_base 读取方法实现
 * \param[out] stump: 未初始化的决策树桩
 * \param[in] file: 保存了 stump 的文件，文件内容将被读入到 stump 中
 * \return 成功则返回真；失败则返回假
 */
static inline bool cstump_read(const struct cstump_base *stump, FILE * file)
{
	if (fread((void *)stump, sizeof(struct cstump_base), 1, file) < 1)
		return false;
	return true;
}

/**
 * \brief cstump_cf_base 读取方法实现
 * \details \copydetails cstump_read()
 */
static inline bool cstump_cf_read(const struct cstump_cf_base *stump,
				  FILE * file)
{
	if (fread((void *)stump, sizeof(struct cstump_cf_base), 1, file) < 1)
		return false;
	return true;
}

#endif
