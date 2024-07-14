#ifndef VEC_BASE_PVT_H
#define VEC_BASE_PVT_H
#include <string.h>
#include <stdlib.h>
#include "adaboost_base.h"
#include "WeakLearner/weaklearner.h"
#include "WeakLearner/stump/vec_stump.h"
/**
 * \file vec_base_pvt.h
 * \brief 样本集为向量集时共用的类型定义及函数声明。
 * 	包含样本集、Adaboost 结构体的包装，以及结构体共用的私有函数。
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/// 样本集结构体
struct sp_wrap {
	const void *sample;		///< 样本集地址, m*n 数组（m 为样本数量）
	void *cache;			///< 样本排序结果（按样本标号存储）, n*m
					///< 数组（m 为样本数量）
	const struct wl_handles *handles;
					///< 弱学习器的回调函数集合  
	dim_t n;			///< 每个样本向量的长度
};

/// Adaboost 结构体包装
struct ada_wrap {
	void *adaboost;			///< Adaboost 结构体地址
	size_t wl_size;			///< 弱学习器所需字节
	turn_t t;			///< 当前训练轮数
};

/// 训练设置集
struct train_setting {
	struct sp_wrap sp;		///< 样本集合
	struct ada_wrap ada;		///< Adaboost 分类器
};

/**
 * \brief 回调函数类型定义，用于处理全部样本分类成功的情形
 * \param[out] ada 样本全部分类成功时的 struct ada_wrap 结构体指针，含有当前轮数
 * \param[in] hl   已初始化的弱学习器回调函数集
 * \return 成功则返回真，否则返回假
 */
typedef bool (*all_pass_fn)(struct ada_wrap *ada, const struct wl_handles *hl);

/*******************************************************************************
 * 				   宏函数定义
 ******************************************************************************/
/**
 * \brief 弱学习器系数数组读写函数
 * \param[in, out] alpha flt_t 数组地址
 * \param[in] nmemb      是元素数量
 * \param[in, out] file  FILE* 类型，用于读取或写入弱学习器系数
 * \param[in] rw_fun     fread 或 fwrite
 * \return: 成功则返回假，否则返回真
 */
#define ALPHA_RW(alpha, nmemb, file, rw_fun)					\
	(rw_fun(alpha, sizeof(flt_t), nmemb, file) < nmemb)

/**
 * \brief 弱学习器系数复制函数
 * \param[out] dst  目标 alpha 数组地址
 * \param[in] src   为源 alpha 数组地址
 * \param[in] nmemb 为元素个数
 */
#define ALPHA_COPY(dst, src, nmemb)						\
	memcpy(dst, src, nmemb * sizeof(flt_t))

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/**
 * \brief 弱学习器读取函数
 * \param[out] weaklearner 要保存弱学习器的目标数组地址
 * \param[in] nmemb        弱学习器数量
 * \param[in] handles      弱学习器的回调函数集合
 * \param[in] file         保存有弱学习器的文件
 * \return 返回成功读取的弱学习器数量
 */
turn_t vec_wl_read(unsigned char *weaklearner, turn_t nmemb,
		   const struct wl_handles *handles, FILE * file);

/**
 * \brief 弱学习器写入函数
 * \param[in] weaklearner 弱学习器的数组地址
 * \param[in] nmemb       弱学习器数量
 * \param[in] handles     弱学习器的回调函数集合
 * \param[out] file       目标文件
 * \return 返回成功写入的弱学习器数量
 */
turn_t vec_wl_write(const unsigned char *weaklearner, turn_t nmemb,
		    const struct wl_handles *handles, FILE * file);

/**
 * \brief 弱学习器数组复制函数
 * \param[out] dst  目标弱学习器数组地址
 * \param[in] src   源弱学习器数组地址
 * \param[in] nmemb 元素数量
 * \param[in] hl    struct wl_handles *类型的回调函数集
 * \return 返回成功复制的弱学习器数量
 */
turn_t vec_wl_copy(unsigned char *dst, const unsigned char *src,
		   turn_t nmemb, const struct wl_handles *hl);

/**
 * \brief 弱学习器数组释放函数
 * \param[in] wl      弱学习器数组地址
 * \param[in] nmemb   元素数量
 * \param[in] handles struct wl_handles *类型的回调函数集
 */
void vec_wl_free(unsigned char *wl, turn_t nmemb,
		 const struct wl_handles *handles);

/*******************************************************************************
 * 				  静态函数定义
 ******************************************************************************/
/**
 * \brief 训练的初始化操作
 * \param[out] st       指向未初始化的训练设置集
 * \param[out] adaboost 指向 struct vec_adaboost 结构体
 * \param[in] m         样本数量
 * \param[in] n         样本特征数量，即样本向量的长度
 * \param[in] X         样本集
 * \param[in] cache_on  表示是否启用缓存，真值表示启用
 * \param[in] wl_hl     弱学习器回调函数集合
 * \return 成功则返回真，否则返回假
 */
static inline bool init_setting(struct train_setting *st, void *adaboost,
				num_t m, dim_t n, const sample_t X[m][n],
				bool cache_on, const struct wl_handles *wl_hl)
{
	st->sp.sample = X;
	st->sp.cache = NULL;
	st->sp.handles = wl_hl;
	st->sp.n = n;

	st->ada.adaboost = adaboost;
	st->ada.t = 0;
	st->ada.wl_size = wl_hl->size;

	if (cache_on && (st->sp.cache = vec_new_cache(m, n, X)) == NULL)
		return false;
	return true;
}

/**
 * \brief 训练设置集的内存释放操作
 * \param[in] st 指向已使用 init_setting() 设置的结构体
 */
static inline void free_setting(struct train_setting *st)
{
	free(st->sp.cache);
}

#endif
