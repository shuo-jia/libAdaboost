#include <float.h>
#include <stdlib.h>
#include <string.h>
#include "stump_base.h"
#include "stump_base_pvt.h"
/**
 * \file stump_base.c
 * \brief 决策树桩基类实现
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-13
 */

/*******************************************************************************
 * 				   宏函数定义
 ******************************************************************************/
/**
 * \brief 从文件读取 struct dstump_base 或 struct dstump_cf_base 类型变量
 * \param[out] stump    未初始化的决策树桩
 * \param[in] file      已打开的文件
 * \param[in] op_len    输出值（output）字段类型的长度
 * \param[in] alloc_fun dstump_alloc 或 dstump_cf_alloc
 * \param[in] free_fun  dstump_free 或 dstump_cf_free
 * \return 成功则返回真，否则返回假
 */
#define DSTUMP_READ(stump, file, op_len, alloc_fun, free_fun)			\
({										\
	bool done = false;							\
	do {									\
		stump->value = NULL;						\
		stump->output = NULL;						\
		if (fread (&stump->size, sizeof(num_t), 1, file) < 1)		\
			break;							\
		if (!alloc_fun (stump, stump->size))				\
			break;							\
		if (fread (&stump->default_output, op_len, 1, file) < 1)	\
			break;							\
		if (fread (stump->value, sizeof(sample_t), stump->size, file)	\
				< stump->size)					\
			break;							\
		if (fread (stump->output, op_len, stump->size, file)		\
				< stump->size)					\
			break;							\
		done = true;							\
	} while (0);								\
	if (!done)								\
		free_fun (stump);						\
	done;									\
})

/**
 * \brief 向文件写入 struct dstump_base 或 struct dstump_cf_base 类型变量
 * \param[in] stump  已初始化的决策树桩
 * \param[out] file  已打开的文件
 * \param[in] op_len 输出值（output）字段类型的长度
 * \return 成功则返回真，否则返回假
 */
#define DSTUMP_WRITE(stump, file, op_len)					\
({										\
										\
	bool done = false;							\
	do {									\
		if (fwrite (&stump->size, sizeof(num_t), 1, file) < 1)		\
			break;							\
		if (fwrite (&stump->default_output, op_len, 1, file) < 1)	\
			break;							\
		if (fwrite (stump->value, sizeof(sample_t), stump->size, file)	\
				< stump->size)					\
			break;							\
		if (fwrite (stump->output, op_len, stump->size, file)		\
				< stump->size)					\
			break;							\
		done = true;							\
	} while (0);								\
	done;									\
})

/**
 * \brief 决策树桩训练，将训练结果保存至决策树桩，并更新最优划分属性
 * \param[out] stump    决策树桩基类，用于保存划分属性的划分值及输出值
 * \param[out] opt      变量地址，用于保存最优划分属性
 * \param[out] feature  变量地址，指向的变量与 opt 所指向变量有相同类型，用作临
 *                      时变量
 * \param[out] seg      变量地址，用于保存划分值及权重信息，可为
 *                      struct cstump_segment *
 * \param[in] m         样本数量
 * \param[in] samples   样本集
 * \param[in] label     样本标签
 * \param[in] D         样本概率分布
 * \param[in] handles   操作当前决策树桩的回调函数集合，
 *                      struct stump_opt_handles * 类型
 * \param[in] get_z     获取划分值及权重信息并保存到 seg 变量的回调函数，
 *                      如 cstump_raw_get_z 
 * \param[in] update    更新决策树桩的回调函数，如 cstump_update
 */
#define TRAIN(stump, opt, feature, seg, m, samples, label, D, handles, get_z,	\
	      update)								\
do {										\
	flt_t min_z = DBL_MAX;							\
	handles->init_feature (feature, samples);				\
	do {									\
		get_z (seg, feature, m, samples, label, D, handles);		\
		if ((seg)->z < min_z) {						\
			min_z = (seg)->z;					\
			update (stump, seg);					\
			handles->update_opt (opt, feature);			\
		}								\
	} while (handles->next_feature (feature, samples));			\
} while(0)

/*******************************************************************************
 * 				    函数实现
 ******************************************************************************/
bool cstump_opt(struct cstump_base *stump, void *opt, size_t ft_size,
		num_t m, const void *samples, const label_t * label,
		const flt_t * D, const struct stump_opt_handles *handles)
{
	typeof(&cstump_raw_get_z) get_z = (handles->get_vals.sort == NULL) ?
	    cstump_raw_get_z : cstump_sort_get_z;

	char feature[ft_size];
	struct cstump_segment seg;
	TRAIN(stump, opt, feature, &seg, m, samples, label, D, handles, get_z,
	      cstump_update);
	return true;
}

bool cstump_cf_opt(struct cstump_cf_base *stump, void *opt, size_t ft_size,
		   num_t m, const void *samples, const label_t * label,
		   const flt_t * D, const struct stump_opt_handles *handles)
{
	typeof(&cstump_raw_get_z) get_z = (handles->get_vals.sort == NULL) ?
	    cstump_raw_get_z : cstump_sort_get_z;

	char feature[ft_size];
	struct cstump_segment seg;
	TRAIN(stump, opt, feature, &seg, m, samples, label, D, handles, get_z,
	      cstump_cf_update);
	return true;
}

bool dstump_opt(struct dstump_base *stump, void *opt, size_t ft_size,
		num_t m, const void *samples, const label_t * label,
		const flt_t * D, const struct stump_opt_handles *handles)
{
	typeof(&dstump_raw_get_z) get_z = (handles->get_vals.sort == NULL) ?
	    dstump_raw_get_z : dstump_sort_get_z;
	if (dstump_alloc(stump, m) == false)
		return false;

	char feature[ft_size];
	struct dstump_segment *seg = init_dseg(m);
	if (seg == NULL) {
		dstump_free(stump);
		return false;
	}
	TRAIN(stump, opt, feature, seg, m, samples, label, D, handles, get_z,
	      dstump_update);
	free(seg);
	return dstump_realloc(stump, stump->size);
}

bool dstump_cf_opt(struct dstump_cf_base *stump, void *opt, size_t ft_size,
		   num_t m, const void *samples, const label_t * label,
		   const flt_t * D, const struct stump_opt_handles *handles)
{
	typeof(&dstump_raw_get_z) get_z = (handles->get_vals.sort == NULL) ?
	    dstump_raw_get_z : dstump_sort_get_z;
	if (dstump_cf_alloc(stump, m) == false)
		return false;

	char feature[ft_size];
	struct dstump_segment *seg = init_dseg(m);
	if (seg == NULL) {
		dstump_cf_free(stump);
		return false;
	}
	TRAIN(stump, opt, feature, seg, m, samples, label, D, handles, get_z,
	      dstump_cf_update);
	free(seg);
	return dstump_cf_realloc(stump, stump->size);
}

bool dstump_alloc(struct dstump_base *stump, num_t n)
{
	if ((stump->value = malloc(sizeof(sample_t) * n)) == NULL)
		return false;
	if ((stump->output = malloc(sizeof(label_t) * n)) == NULL) {
		free(stump->value);
		return false;
	}
	stump->size = n;
	return true;
}

bool dstump_cf_alloc(struct dstump_cf_base *stump, num_t n)
{
	if ((stump->value = malloc(sizeof(sample_t) * n)) == NULL)
		return false;
	if ((stump->output = malloc(sizeof(flt_t) * n)) == NULL) {
		free(stump->value);
		return false;
	}
	stump->size = n;
	return true;
}

bool dstump_realloc(struct dstump_base *stump, num_t n)
{
	void *ptr;
	if ((ptr = realloc(stump->value, sizeof(sample_t) * n)) == NULL)
		return false;
	stump->value = ptr;
	if ((ptr = realloc(stump->output, sizeof(label_t) * n)) == NULL)
		return false;
	stump->output = ptr;
	stump->size = n;
	return true;
}

bool dstump_cf_realloc(struct dstump_cf_base *stump, num_t n)
{
	void *ptr;
	if ((ptr = realloc(stump->value, sizeof(sample_t) * n)) == NULL)
		return false;
	stump->value = ptr;
	if ((ptr = realloc(stump->output, sizeof(flt_t) * n)) == NULL)
		return false;
	stump->output = ptr;
	stump->size = n;
	return true;
}

void dstump_free(struct dstump_base *stump)
{
	free(stump->value);
	free(stump->output);
}

void dstump_cf_free(struct dstump_cf_base *stump)
{
	free(stump->value);
	free(stump->output);
}

label_t dstump_h(const struct dstump_base *stump, sample_t value)
{
	const sample_t *ptr;
	ptr = bsearch(&value, stump->value, stump->size, sizeof(sample_t),
		      sample_cmp);
	if (ptr != NULL)
		return stump->output[ptr - stump->value];
	else
		return stump->default_output;
}

flt_t dstump_cf_h(const struct dstump_cf_base *stump, sample_t value)
{
	const sample_t *ptr;
	ptr = bsearch(&value, stump->value, stump->size, sizeof(sample_t),
		      sample_cmp);
	if (ptr != NULL)
		return stump->output[ptr - stump->value];
	else
		return stump->default_output;
}

bool dstump_write(const struct dstump_base *stump, FILE * file)
{
	return DSTUMP_WRITE(stump, file, sizeof(label_t));
}

bool dstump_cf_write(const struct dstump_cf_base *stump, FILE * file)
{
	return DSTUMP_WRITE(stump, file, sizeof(flt_t));
}

bool dstump_read(struct dstump_base *stump, FILE * file)
{
	return DSTUMP_READ(stump, file, sizeof(label_t), dstump_alloc,
			   dstump_free);
}

bool dstump_cf_read(struct dstump_cf_base *stump, FILE * file)
{
	return DSTUMP_READ(stump, file, sizeof(flt_t), dstump_cf_alloc,
			   dstump_cf_free);
}

void *dstump_copy(struct dstump_base *dst, const struct dstump_base *src)
{
	if (!dstump_alloc(dst, src->size))
		return NULL;
	memcpy(dst->value, src->value, sizeof(sample_t) * src->size);
	memcpy(dst->output, src->output, sizeof(label_t) * src->size);
	dst->default_output = src->default_output;
	dst->size = src->size;
	return dst;
}

// struct dstump_cf_base 类型深度复制方法
void *dstump_cf_copy(struct dstump_cf_base *dst,
		     const struct dstump_cf_base *src)
{
	if (!dstump_cf_alloc(dst, src->size))
		return NULL;
	memcpy(dst->value, src->value, sizeof(sample_t) * src->size);
	memcpy(dst->output, src->output, sizeof(flt_t) * src->size);
	dst->default_output = src->default_output;
	dst->size = src->size;
	return dst;
}

int sample_cmp(const void *p1, const void *p2)
{
	sample_t result = *(const sample_t *)p1 - *(const sample_t *)p2;
	if (result > 0)
		return 1;
	else if (result < 0)
		return -1;
	else
		return 0;
}

int sample_ptr_cmp(const void *p1, const void *p2)
{
	sample_t result = **(const sample_t **)p1 - **(const sample_t **)p2;
	if (result > 0)
		return 1;
	else if (result < 0)
		return -1;
	else
		return 0;
}
