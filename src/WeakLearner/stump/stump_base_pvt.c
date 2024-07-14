#include <math.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>
#include "stump_base_pvt.h"
/**
 * \file stump_base_pvt.c
 * \brief stump_base 的私有部分实现
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-13
 */

/*******************************************************************************
 * 				   宏函数定义
 ******************************************************************************/
/**
 * \brief 计算 cstump 系列的 Z/2 值
 * \param[in] W0 负例权重，W0[0]、W0[1] 分别为左、右两侧的负例权重
 * \param[in] W1 正例权重，W1[0]、W1[1] 分别为左、右两侧的正例权重
 * \return 返回 Z 值的二分之一
 */
#define CSTUMP_Z(W0, W1) (sqrt ((W0)[0] * (W1)[0]) + sqrt ((W0)[1] * (W1)[1]))

/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
/**
 * \brief 将数组分成左右两组，左侧元素的值小于右侧元素的值
 * \param[out] seg 用于保存分割值
 * \param[out] p   用于保存分位置的索引（X[*p..] > *seg）
 * \param[in]  X   要进行划分的指针数组
 * \param[in]  m   数组长度
 */
static void partition(flt_t * seg, num_t * p, const sample_t * X[], num_t m);

/**
 * \brief 获取单个属性的最优划分值
 * 	类似于快速排序算法，但估算了 Z 值的下界并剪枝，以此提速
 * \param[out] best  保存最优划分值
 * \param[in] X0     保存参数 X 指针数组中最小的地址
 * \param[in, out] X sample_t 型指针数组，每个元素指向 sample_t 数组元素
 *      （样本在单个属性上的取值），X 将被排序。
 * \param[in] Y      样本集标签，不进行排序
 * \param[in] D      样本概率分布，不进行排序
 * \param[in] m      样本数量
 * \param[in] left   在 X 最左侧进行划分的划分值
 * \param[in] right  在 X 最右侧进行划分的划分值
 */
static void quick_get_segment(struct cstump_segment *best, const sample_t * X0,
			      const sample_t * X[], const label_t Y[],
			      const flt_t D[], num_t m,
			      const struct cstump_segment *left,
			      const struct cstump_segment *right);

/*******************************************************************************
 * 				    函数实现
 ******************************************************************************/
void cstump_raw_get_z(struct cstump_segment *seg, const void *feature, num_t m,
		      const void *samples, const label_t * label,
		      const flt_t D[], const struct stump_opt_handles *handles)
{
	const flt_t epsilon = 1.0 / m;
	const sample_t *values = handles->get_vals.raw(m, samples, feature);
	const sample_t *X[m];
	for (num_t i = 0; i < m; ++i)
		X[i] = values + i;

	struct cstump_segment left = {
		.value = DBL_MAX,
		.W = { { epsilon, epsilon}, { epsilon, epsilon} },
	};
	struct cstump_segment right = {
		.value = -DBL_MAX,
		.W = { { epsilon, epsilon}, { epsilon, epsilon} },
	};

	bool p_or_n;
	// 计算最左、最右两侧的划分值以及在最左侧划分时的权重
	for (num_t i = 0; i < m; ++i) {
		if (values[i] < left.value)
			left.value = values[i];
		if (values[i] > right.value)
			right.value = values[i];
		p_or_n = (bool)(label[i] > 0);
		left.W[p_or_n][1] += D[i];
	}
	left.value -= VEC_SEG_INTERVAL;
	right.value += VEC_SEG_INTERVAL;
	// 计算左、右侧划分产生的 Z 值
	right.W[0][0] = left.W[0][1];
	right.W[1][0] = left.W[1][1];
	left.z = CSTUMP_Z(left.W[0], left.W[1]);
	right.z = left.z;
	// 当前最优划分位置为左侧（左、右侧 Z 值相同）
	*seg = left;
	quick_get_segment(seg, values, X, label, D, m, &left, &right);
}

void cstump_sort_get_z(struct cstump_segment *seg, const void *feature, num_t m,
		       const void *samples, const label_t * label,
		       const flt_t D[], const struct stump_opt_handles *handles)
{
	flt_t z;
	const flt_t epsilon = 1.0 / m;
	flt_t W[2][2] = { { epsilon, epsilon }, { epsilon, epsilon } };
	const num_t *ids = handles->get_vals.sort(m, samples, feature);
	const sample_t *values = handles->get_vals.raw(m, samples, feature);

	num_t best_posi, i;
	bool p_or_n;
	for (i = 0; i < m; ++i)	// 分割位置在最左侧的情形
		W[(bool)(label[i] > 0)][1] += D[i];
	seg->z = CSTUMP_Z(W[0], W[1]);
	memcpy(seg->W, W, sizeof(flt_t) * 2 * 2);
	best_posi = 0;

	for (i = 0; i < m - 1; ++i) {	// 分割位置在中间的情形
		// 逐步移动分割位置，更新 W_+ 和 W_-
		p_or_n = (bool)(label[ids[i]] > 0);
		W[p_or_n][1] -= D[ids[i]];
		W[p_or_n][0] += D[ids[i]];
		z = CSTUMP_Z(W[0], W[1]);
		if (z < seg->z && values[ids[i]] != values[ids[i + 1]]) {
			seg->z = z;
			memcpy(seg->W, W, sizeof(flt_t) * 2 * 2);
			best_posi = i + 1;
		}
	}

	p_or_n = (bool)(label[ids[m - 1]] > 0);	// 分割位置在最右侧的情形
	W[p_or_n][1] -= D[ids[m - 1]];
	W[p_or_n][0] += D[ids[m - 1]];
	z = CSTUMP_Z(W[0], W[1]);
	if (z < seg->z) {
		seg->z = z;
		memcpy(seg->W, W, sizeof(flt_t) * 2 * 2);
		best_posi = m;
	}
	// 计算分割值
	if (best_posi == 0)
		seg->value = values[ids[best_posi]] - VEC_SEG_INTERVAL;
	else if (best_posi < m)
		seg->value = (flt_t) (values[ids[best_posi - 1]] +
				      values[ids[best_posi]]) / 2.0;
	else
		seg->value = values[ids[best_posi]] + VEC_SEG_INTERVAL;
}

void dstump_raw_get_z(struct dstump_segment *seg, const void *feature, num_t m,
		      const void *samples, const label_t * label,
		      const flt_t D[], const struct stump_opt_handles *handles)
{
	const flt_t epsilon = 1.0 / m;
	const sample_t *values = handles->get_vals.raw(m, samples, feature);
	const sample_t *ptrs[m];
	for (num_t i = 0; i < m; ++i)
		ptrs[i] = values + i;
	qsort(ptrs, m, sizeof(sample_t *), sample_ptr_cmp);
	memset(seg->W[0], 0, m * sizeof(flt_t));
	memset(seg->W[1], 0, m * sizeof(flt_t));
	seg->g_W[0] = seg->g_W[1] = seg->z = seg->len = 0;

	num_t id = ptrs[0] - values;
	bool p_or_n = (bool)(label[id] > 0);
	seg->W[p_or_n][0] += D[id];
	seg->g_W[p_or_n] += D[id];
	for (num_t i = 1; i < m; ++i) {
		if (ptrs[i][0] != ptrs[i - 1][0]) {
			seg->z +=
			    sqrt(seg->W[0][seg->len] * seg->W[1][seg->len]);
			seg->value[seg->len] = ptrs[i - 1][0];
			++seg->len;
		}
		id = ptrs[i] - values;
		p_or_n = (bool)(label[id] > 0);
		seg->W[p_or_n][seg->len] += D[id];
		seg->g_W[p_or_n] += D[id];
	}
	seg->z += sqrt(seg->W[0][seg->len] * seg->W[1][seg->len]);
	seg->value[seg->len] = ptrs[m - 1][0];
	++seg->len;
	for (num_t i = 0; i < seg->len; ++i) {
		seg->W[0][i] += epsilon;
		seg->W[1][i] += epsilon;
	}
	seg->g_W[0] += epsilon;
	seg->g_W[1] += epsilon;
}

void dstump_sort_get_z(struct dstump_segment *seg, const void *feature,
		       num_t m, const void *samples, const label_t * label,
		       const flt_t D[], const struct stump_opt_handles *handles)
{
	const flt_t epsilon = 1.0 / m;
	const num_t *ids = handles->get_vals.sort(m, samples, feature);
	const sample_t *values = handles->get_vals.raw(m, samples, feature);
	memset(seg->W[0], 0, m * sizeof(flt_t));
	memset(seg->W[1], 0, m * sizeof(flt_t));
	seg->g_W[0] = seg->g_W[1] = seg->z = seg->len = 0;

	bool p_or_n = (bool)(label[ids[0]] > 0);
	seg->W[p_or_n][0] += D[ids[0]];
	for (num_t i = 1; i < m; ++i) {
		if (values[ids[i - 1]] != values[ids[i]]) {
			seg->z +=
			    sqrt(seg->W[0][seg->len] * seg->W[1][seg->len]);
			seg->value[seg->len] = values[ids[i - 1]];
			++seg->len;
		}
		p_or_n = (bool)(label[ids[i]] > 0);
		seg->W[p_or_n][seg->len] += D[ids[i]];
		seg->g_W[p_or_n] += D[ids[i]];
	}
	seg->z += sqrt(seg->W[0][seg->len] * seg->W[1][seg->len]);
	seg->value[seg->len] = values[ids[m - 1]];
	++seg->len;
	for (num_t i = 0; i < seg->len; ++i) {
		seg->W[0][i] += epsilon;
		seg->W[1][i] += epsilon;
	}
	seg->g_W[0] += epsilon;
	seg->g_W[1] += epsilon;
}

void cstump_update(struct cstump_base *stump, const struct cstump_segment *seg)
{
	stump->value = seg->value;
	stump->output[0] = (seg->W[1][0] > seg->W[0][0]) ? 1 : -1;
	stump->output[1] = (seg->W[1][1] > seg->W[0][1]) ? 1 : -1;
}

void cstump_cf_update(struct cstump_cf_base *stump,
		      const struct cstump_segment *seg)
{
	stump->value = seg->value;
	stump->output[0] = log(seg->W[1][0] / seg->W[0][0]) / 2.0;
	stump->output[1] = log(seg->W[1][1] / seg->W[0][1]) / 2.0;
}

void dstump_update(struct dstump_base *stump, const struct dstump_segment *seg)
{
	stump->size = seg->len;
	memcpy(stump->value, seg->value, sizeof(sample_t) * seg->len);
	for (num_t i = 0; i < seg->len; ++i)
		stump->output[i] = (seg->W[1][i] > seg->W[0][i]) ? 1 : -1;
	stump->default_output = (seg->g_W[1] > seg->g_W[0]) ? 1 : -1;
}

void dstump_cf_update(struct dstump_cf_base *stump,
		      const struct dstump_segment *seg)
{
	stump->size = seg->len;
	memcpy(stump->value, seg->value, sizeof(sample_t) * seg->len);
	for (num_t i = 0; i < seg->len; ++i)
		stump->output[i] = log(seg->W[1][i] / seg->W[0][i]) / 2.0;
	stump->default_output = log(seg->g_W[1] / seg->g_W[0]) / 2.0;
}

/*******************************************************************************
 * 				  静态函数实现
 ******************************************************************************/
void partition(flt_t * seg, num_t * p, const sample_t * X[], num_t m)
{
	num_t i, j;
	const sample_t *temp;

	// 找出不同的两个数，取中间值作为划分值
	i = rand() % m;
	temp = X[0];
	X[0] = X[i];
	X[i] = temp;
	for (i = 1; i < m; ++i)
		if (*X[0] != *X[i])
			break;
	if (i >= m) {
		*p = 0;
		return;
	}
	// 计算划分值。对于整数型属性，划分值容易出现整数，故减去一个固定小数
	*seg = (flt_t) (*X[0] + *X[i]) / 2 - VEC_SEG_INTERVAL;

	// 重新排列数组，使得划分值左侧的值小于划分值，右侧则大于划分值
	i = 0;
	for (j = 0; j < m; j++)
		if (*X[j] <= *seg) {
			temp = X[i];
			X[i] = X[j];
			X[j] = temp;
			++i;
		}
	*p = i;			// 从 *p 开始，X[i] 的值 > *seg
}

void quick_get_segment(struct cstump_segment *best, const sample_t * X0,
		       const sample_t * X[], const label_t Y[], const flt_t D[],
		       num_t m, const struct cstump_segment *left,
		       const struct cstump_segment *right)
{
	if (m <= 1)
		return;

	num_t p;		// 划分位置（索引）
	num_t id;		// 保存排序后元素的索引
	bool p_or_n;		// 表示正例(1)或负例(0)
	struct cstump_segment curr = *left;
	partition(&curr.value, &p, X, m);	// 获取划分位置 p
	if (p == 0 || p == m)
		return;
	// 将划分位置从最左侧移动到索引 p
	for (num_t i = 0; i < p; ++i) {
		id = X[i] - X0;
		p_or_n = (bool)(Y[id] > 0);
		curr.W[p_or_n][1] -= D[id];
		curr.W[p_or_n][0] += D[id];
	}
	curr.z = CSTUMP_Z(curr.W[0], curr.W[1]);

	if (curr.z < best->z)
		*best = curr;

	flt_t W[2][2] = {
		{ left->W[0][0], curr.W[0][1] },
		{ left->W[1][0], curr.W[1][1] }
	};
	if (CSTUMP_Z(W[0], W[1]) < best->z)
		quick_get_segment(best, X0, X, Y, D, p, left, &curr);
	W[0][0] = curr.W[0][0];
	W[1][0] = curr.W[1][0];
	W[0][1] = right->W[0][1];
	W[1][1] = right->W[1][1];
	if (CSTUMP_Z(W[0], W[1]) < best->z)
		quick_get_segment(best, X0, X + p, Y, D, m - p, &curr, right);
}
