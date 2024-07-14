#include <stdlib.h>
#include "vec_stump.h"
#include "stump_base.h"
/**
 * \file vec_stump.c
 * \brief 决策树桩子类，从样本特征构成的向量中构造弱学习器（函数实现）
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/// 样本集结构体
struct sp_wrap {
	const void *samples;	///< 样本集，m*n sample_t 矩阵（行表示样本）
	const void *sorted_sp;	///< n*m num_t 矩阵，一行表示样本标号在某特征上的排序
	sample_t *vector;	///< 列向量，保存样本集在某一特征上的取值
	dim_t n;		///< 样本维度
};

/*******************************************************************************
 * 				   宏函数定义
 ******************************************************************************/
/**
 * \brief 决策树桩读写
 * \param[in, out] stump 决策树桩指针
 * \param[in, out] file  已打开的文件
 * \param[in] type       stump 的类型
 * \param[in] fun        fread 或 fwrite 函数的地址
 * \param[in] base_fun   父类的读函数或写函数的地址
 * \return 成功则返回真，失败返回假
 */
#define STUMP_RW(stump, file, type, fun, base_fun)                              \
({				                                                \
	bool done = false;							\
	do {									\
		type * ptr = stump;						\
		if (fun (&ptr->feature, sizeof(dim_t), 1, file) < 1)		\
			break;							\
		if (!base_fun (&ptr->base, file))				\
			break;							\
		done = true;							\
	} while (0);								\
	done = true;								\
})

/**
 * 训练模板
 * \param[in] stump_type: 即 stump 实际上的类型
 * \param[in] fun_opt: 基类选择最优划分属性函数的函数名，如 cstump_opt、cstump_cf_opt 等等
 * \details \copydetails vec_cstump_train()
 */
#define TRAIN(stump, m, n, X, Y, D, cache, stump_type, fun_opt)			\
({										\
 	bool status;								\
	do {									\
		struct sp_wrap sp;						\
		struct stump_opt_handles handles;				\
		if (!init_train(&sp, &handles, stump, X, m, n, cache)) {	\
			status = false;						\
			break;							\
		}								\
										\
		stump_type ptr_stump = stump;					\
		status = fun_opt(&ptr_stump->base, &ptr_stump->feature,		\
				sizeof(dim_t), m, &sp, Y, D, &handles);		\
		free_train (&sp, &handles);					\
	} while (0);								\
	status;									\
})

/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
/**
 * \brief 特征初始化函数
 * \param[out] feature 实际类型为 dim_t *
 * \param[in] samples  实际类型为 struct sp_wrap *
 */
static void init_feature(void *feature, const void *samples);

/**
 * \brief 获取下一特征
 * \details \copydetails init_feature()
 */
static void *next_feature(void *feature, const void *samples);

/// 最优特征更新函数
static void update_opt(void *opt, const void *feature);

/// 获取特征数组，传入的样本集为样本集（struct sp_wrap *）
static const sample_t *get_vals_raw(num_t m, const void *samples,
				    const void *feature);
/// 获取样本集标号在某特征上的排序结果，传入的样本集为排序结果缓存
static const num_t *get_vals_sort(num_t m, const void *samples,
				  const void *feature);

/**
 * \brief 训练的初始化操作
 * \param[out] sp      指向未初始化的 struct sp_wrap 结构体
 * \param[out] handles 指向未初始化的回调函数集
 * \param[in] stump    已初始化的决策树桩
 * \param[in] X        样本集
 * \param[in] m        样本数量
 * \param[in] n        样本特征数量
 * \return 成功则返回真，失败则返回假
 */
static bool init_train(struct sp_wrap *sp, struct stump_opt_handles *handles,
		       const void *stump, const void *X, num_t m, dim_t n,
		       const void *cache);

/**
 * \brief 训练资源释放操作
 * \param[out] sp     指向已初始化的 struct sp_wrap 结构体
 * \param[in] handles 指向已初始化的回调函数集
 */
static inline void free_train(struct sp_wrap *sp,
			      const struct stump_opt_handles *handles);

/*******************************************************************************
 * 				    函数实现
 ******************************************************************************/
bool vec_cstump_train(void *stump, num_t m, dim_t n, const sample_t X[m][n],
		      const label_t Y[], const flt_t D[], const void *cache)
{
	return TRAIN(stump, m, n, X, Y, D, cache, struct vec_cstump *,
		     cstump_opt);
}

bool vec_cstump_cf_train(void *stump, num_t m, dim_t n, const sample_t X[m][n],
			 const label_t Y[], const flt_t D[], const void *cache)
{
	return TRAIN(stump, m, n, X, Y, D, cache, struct vec_cstump_cf *,
		     cstump_cf_opt);
}

bool vec_dstump_train(void *stump, num_t m, dim_t n, const sample_t X[m][n],
		      const label_t Y[], const flt_t D[], const void *cache)
{
	return TRAIN(stump, m, n, X, Y, D, cache, struct vec_dstump *,
		     dstump_opt);
}

bool vec_dstump_cf_train(void *stump, num_t m, dim_t n, const sample_t X[m][n],
			 const label_t Y[], const flt_t D[], const void *cache)
{
	return TRAIN(stump, m, n, X, Y, D, cache, struct vec_dstump_cf *,
		     dstump_cf_opt);
}

label_t vec_cstump_h(const void *stump, const sample_t x[], dim_t n)
{
	const struct vec_cstump *ptr = stump;
	return cstump_h(&ptr->base, x[ptr->feature]);
}

flt_t vec_cstump_cf_h(const void *stump, const sample_t x[], dim_t n)
{
	const struct vec_cstump_cf *ptr = stump;
	return cstump_cf_h(&ptr->base, x[ptr->feature]);
}

label_t vec_dstump_h(const void *stump, const sample_t x[], dim_t n)
{
	const struct vec_dstump *ptr = stump;
	return dstump_h(&ptr->base, x[ptr->feature]);
}

flt_t vec_dstump_cf_h(const void *stump, const sample_t x[], dim_t n)
{
	const struct vec_dstump_cf *ptr = stump;
	return dstump_cf_h(&ptr->base, x[ptr->feature]);
}

bool vec_cstump_read(void *stump, FILE * file)
{
	return STUMP_RW(stump, file, struct vec_cstump, fread, cstump_read);
}

bool vec_cstump_cf_read(void *stump, FILE * file)
{
	return STUMP_RW(stump, file, struct vec_cstump_cf, fread,
			cstump_cf_read);
}

bool vec_dstump_read(void *stump, FILE * file)
{
	return STUMP_RW(stump, file, struct vec_dstump, fread, dstump_read);
}

bool vec_dstump_cf_read(void *stump, FILE * file)
{
	return STUMP_RW(stump, file, struct vec_dstump_cf, fread,
			dstump_cf_read);
}

bool vec_cstump_write(const void *stump, FILE * file)
{
	return STUMP_RW(stump, file, const struct vec_cstump, fwrite,
			cstump_write);
}

bool vec_cstump_cf_write(const void *stump, FILE * file)
{
	return STUMP_RW(stump, file, const struct vec_cstump_cf, fwrite,
			cstump_cf_write);
}

bool vec_dstump_write(const void *stump, FILE * file)
{
	return STUMP_RW(stump, file, const struct vec_dstump, fwrite,
			dstump_write);
}

bool vec_dstump_cf_write(const void *stump, FILE * file)
{
	return STUMP_RW(stump, file, const struct vec_dstump_cf, fwrite,
			dstump_cf_write);
}

void *vec_dstump_copy(void *dst, const void *src)
{
	struct vec_dstump *stump_dst = dst;
	const struct vec_dstump *stump_src = src;
	stump_dst->feature = stump_src->feature;

	if (!dstump_copy(&stump_dst->base, &stump_src->base))
		return NULL;
	return dst;
}

void *vec_dstump_cf_copy(void *dst, const void *src)
{
	struct vec_dstump_cf *stump_dst = dst;
	const struct vec_dstump_cf *stump_src = src;
	stump_dst->feature = stump_src->feature;

	if (!dstump_cf_copy(&stump_dst->base, &stump_src->base))
		return NULL;
	return dst;
}

void vec_dstump_free(void *stump)
{
	struct vec_dstump *ptr = stump;
	dstump_free(&ptr->base);
}

void vec_dstump_cf_free(void *stump)
{
	struct vec_dstump_cf *ptr = stump;
	dstump_cf_free(&ptr->base);
}

void *vec_new_cache(num_t m, dim_t n, const sample_t X[m][n])
{
	dim_t(*ids)[m] = malloc(sizeof(dim_t) * m * n);
	const sample_t **ptrs = malloc(sizeof(sample_t *) * m);
	if (ptrs == NULL || ids == NULL) {
		free(ids);
		free(ptrs);
		return NULL;
	}

	dim_t i;
	num_t j;
	for (i = 0; i < n; ++i) {
		for (j = 0; j < m; ++j)
			ptrs[j] = &X[j][i];
		qsort(ptrs, m, sizeof(sample_t *), sample_ptr_cmp);
		for (j = 0; j < m; ++j)
			ids[i][j] = (ptrs[j] - &X[0][i]) / n;
	}
	free(ptrs);
	return ids;
}

/*******************************************************************************
 * 				  静态函数定义
 ******************************************************************************/
void init_feature(void *feature, const void *samples)
{
	*(dim_t *) feature = 0;
}

void *next_feature(void *feature, const void *samples)
{
	dim_t *ft_ptr = feature;
	const struct sp_wrap *sp_ptr = samples;

	++(*ft_ptr);
	if (*ft_ptr >= sp_ptr->n)
		return NULL;

	return feature;
}

void update_opt(void *opt, const void *feature)
{
	*(dim_t *) opt = *(const dim_t *)feature;
}

const sample_t *get_vals_raw(num_t m, const void *samples, const void *feature)
{
	const struct sp_wrap *sp_ptr = samples;
	const dim_t *ft_ptr = feature;
	const sample_t(*sp_mat)[sp_ptr->n] = sp_ptr->samples;

	for (num_t i = 0; i < m; ++i)
		sp_ptr->vector[i] = sp_mat[i][*ft_ptr];
	return sp_ptr->vector;
}

const num_t *get_vals_sort(num_t m, const void *samples, const void *feature)
{
	const struct sp_wrap *sp_ptr = samples;
	const dim_t *ft_ptr = feature;
	const num_t(*sp_mat)[m] = sp_ptr->sorted_sp;

	return sp_mat[*ft_ptr];
}

bool init_train(struct sp_wrap *sp, struct stump_opt_handles *handles,
		const void *stump, const void *X, num_t m, dim_t n,
		const void *cache)
{
	const struct vec_cstump *cstump = stump;
	sp->samples = X;
	sp->n = n;
	handles->init_feature = init_feature;
	handles->next_feature = next_feature;
	handles->update_opt = update_opt;
	handles->get_vals.raw = get_vals_raw;

	if ((sp->vector = malloc(sizeof(sample_t) * m)) == NULL)
		return false;
	if (cache == NULL) {
		sp->sorted_sp = NULL;
		handles->get_vals.sort = NULL;
	} else {
		sp->sorted_sp = cache;
		handles->get_vals.sort = get_vals_sort;
	}

	return true;
}

void free_train(struct sp_wrap *sp, const struct stump_opt_handles *handles)
{
	free(sp->vector);
}
