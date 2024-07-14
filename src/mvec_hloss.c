#include <math.h>
#include <float.h>
#include <stdlib.h>
#include "vec_base_pvt.h"
#include "mvec_hloss.h"
#include "AlphaCalc/alpha.h"

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
// 标签集的结构体包装
struct label_wrap {
	label_t *labels;	// 标签集（label_t 数组，长度为 dim * m，
				//         m 为样本数量）
	mlabel_t dim;		// 不同标签的数量
};

/*******************************************************************************
 * 				   宏函数定义
 ******************************************************************************/
// 计算中间值 vals[j][i] = h(X[i])[l] * Y[i][l]
// vals: 保存中间值的数组
// wl: 弱学习器数组地址；      wl_size: 弱学习器长度（字节）
// m: 样本数量；           n: 样本维度；
// X: 样本集；                      Y: 标签集；         dim: 标签集维度；
// h_fun: 假设函数的名称
#define VALS_CALC(vals, wl, wl_size, m, n, X, Y, dim, h_fun)			\
do {										\
	const unsigned char *wl_ptr = wl;					\
	const label_t (*Y_ptr) [m] = (const label_t (*) [m]) (Y);		\
	flt_t (*vals_ptr) [m] = (flt_t (*) [m]) (vals);				\
	for (mlabel_t j = 0; j < dim; ++j) {					\
		for (num_t i = 0; i < m; ++i)					\
			vals_ptr[j][i] =h_fun(wl_ptr, X[i], n) * Y_ptr[j][i];	\
		wl_ptr += wl_size;						\
	}									\
} while(0)

// 假设器计算函数模板
// ada, ..., hl: 与 mvec_ada_h_xxx() 函数参数意义相同
// output_fun: OUTPUT_XXX() 系列宏函数名
#define H_CALC(ada, x, n, hl, output_fun)					\
({										\
	flt_t output [ada->dim];						\
	const unsigned char * wl_ptr = ada->weaklearner;			\
	memset (output, 0, sizeof(flt_t) * ada->dim);				\
	for (turn_t i = 0; i < ada->group_len; ++i)				\
		for (mlabel_t j = 0; j < ada->dim; ++j) {			\
			output[j] += output_fun(wl_ptr, ada->alpha, i, x, n,	\
					hl);					\
			wl_ptr += hl->size;					\
		}								\
										\
	argmax (output, ada->dim);						\
})

// 输出值计算方法
// wl 为当前弱学习器地址，alpha 为系数数组地址，i 为 alpha 对应元素索引，
// x 为样本数组，n 为样本维度，hl 为弱学习器的回调函数集
#define OUTPUT(wl, alpha, i, x, n, hl)						\
	((alpha)[i] * (hl)->hypothesis.vec(wl, x, n))

#define OUTPUT_FOLD(wl, alpha, i, x, n, hl)					\
	((hl)->hypothesis.vec_cf(wl, x, n))

/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
/*
 * 回调函数集合
 * sample: 实际类型为 struct sp_wrap *
 * adaboost: 实际类型为 struct ada_wrap *
 */
// 弱学习器训练函数
static bool wl_train(void *weaklearner, num_t m, const void *sample,
		     const void *label, const flt_t D[]);
// 获取下一弱学习器
static bool wl_next(struct ada_item *item, void *adaboost,
		    const flt_t vals[], num_t vals_len);
// 获取弱学习器输出
static enum ada_result get_vals(flt_t vals[], num_t vals_len,
				const void *weaklearner, num_t m,
				const void *sample, const void *label,
				const flt_t D[]);
// alpha 计算方法（使用近似方法），包装函数
static flt_t alpha_approx_wrap(const flt_t vals[], num_t vals_len, num_t m,
			       const void *label, const flt_t D[]);
// alpha 计算方法（使用牛顿迭代），包装函数
static flt_t alpha_newton_wrap(const flt_t vals[], num_t vals_len, num_t m,
			       const void *label, const flt_t D[]);
// 权重初始化方法
static void init_D(flt_t D[], num_t m, const void *label);
// 权重更新方法
static void update_D(flt_t D[], flt_t vals[], num_t vals_len, num_t m,
		     const void *label, flt_t alpha);

// 当全部样本分类成功时将被调用，参数 ada 内将保存已训练的轮数
// 弱学习器系数不并入弱学习器
static bool all_pass(struct ada_wrap *ada, const struct wl_handles *hl);
// 弱学习器系数并入弱学习器，无需复制弱学习器系数
static bool all_pass_fold(struct ada_wrap *ada, const struct wl_handles *hl);

/*
 * 初始化标签集包装结构体，为标签集编码
 * lb: 要进行初始化的结构体
 * label: 样本集的标签
 * m: 样本数量（即 label 数组的元素个数）
 */
static bool init_label_wrap(struct label_wrap *lb, const mlabel_t * label,
			    num_t m);

/*
 * 释放 struct label_wrap 结构体内部空间
 * lb: 已初始化的结构体
 */
static inline void free_label_wrap(struct label_wrap *lb);

/*
 * 为采用 hloss 的 Adaboost 设置回调函数集
 * handles: Adaboost 训练所需回调函数集合
 * m: 样本数量
 * dim: 不同标签的数量
 * get_alpha: 计算 alpha 的值（回调函数）
 */
static inline void ada_hl_init(struct ada_handles *handles, num_t m,
			       mlabel_t dim, ada_alpha_fn get_alpha);

/*
 * 输出数组最大值的索引
 * output: 待检查最大值的数组
 * n: 数组元素个数
 * 返回值：返回最大元素的索引
 */
static mlabel_t argmax(flt_t output[], mlabel_t n);

// 训练模板
// adaboost, ..., handles: 与 mvec_ada_xxx_train() 系列函数参数意义相同
// get_alpha: 弱学习器系数计算函数（回调函数）
// all_pass: 样本全部分类成功时调用的回调函数（all_pass_fn 类型）
// 返回值：成功返回真，否则返回假
static inline bool train_framework(struct mvec_adaboost *adaboost, turn_t T,
				   num_t m, dim_t n, const sample_t X[m][n],
				   const mlabel_t Y[], bool cache_on,
				   const struct wl_handles *handles,
				   ada_alpha_fn get_alpha,
				   all_pass_fn all_pass);

/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
// 近似方法，利用不等式缩放获取系数 alpha
bool mvec_ada_approx_train(struct mvec_adaboost *adaboost, turn_t T, num_t m,
			   dim_t n, const sample_t X[m][n], const mlabel_t Y[],
			   bool cache_on, const struct wl_handles *handles)
{
	return train_framework(adaboost, T, m, n, X, Y, cache_on, handles,
			       alpha_approx_wrap, all_pass);
}

// 将 alpha 合并到弱学习器 h 中，即 alpha 恒为 1
bool mvec_ada_fold_train(struct mvec_adaboost *adaboost, turn_t T, num_t m,
			 dim_t n, const sample_t X[m][n], const mlabel_t Y[],
			 bool cache_on, const struct wl_handles *handles)
{
	return train_framework(adaboost, T, m, n, X, Y, cache_on, handles,
			       alpha_eq_1, all_pass_fold);
}

// 数值方法，应用牛顿二分法求系数 alpha
bool mvec_ada_newton_train(struct mvec_adaboost *adaboost, turn_t T, num_t m,
			   dim_t n, const sample_t X[m][n], const mlabel_t Y[],
			   bool cache_on, const struct wl_handles *handles)
{
	return train_framework(adaboost, T, m, n, X, Y, cache_on, handles,
			       alpha_newton_wrap, all_pass);
}

// 输出分类结果，弱学习器系数不并入弱学习器
mlabel_t mvec_ada_h(const struct mvec_adaboost *adaboost, const sample_t x[],
		    dim_t n, const struct wl_handles *handles)
{
	return H_CALC(adaboost, x, n, handles, OUTPUT);
}

// 输出分类结果，弱学习器系数并入弱学习器
mlabel_t mvec_ada_fold_h(const struct mvec_adaboost *adaboost,
			 const sample_t x[], dim_t n,
			 const struct wl_handles *handles)
{
	return H_CALC(adaboost, x, n, handles, OUTPUT_FOLD);
}

/*******************************************************************************
 * 				  静态函数定义
 ******************************************************************************/
bool wl_train(void *weaklearner, num_t m, const void *sample,
	      const void *label, const flt_t D[])
{
	mlabel_t n;
	const struct sp_wrap *sp = sample;
	const struct label_wrap *lb = label;
	unsigned char *wl_ptr = weaklearner;
	const label_t *label_ptr = lb->labels;
	const flt_t *D_ptr = D;
	for (n = 0; n < lb->dim; ++n) {	// 分别对每种分类的学习器训练
		if (!sp->handles->train.vec(wl_ptr, m, sp->n, sp->sample,
					    label_ptr, D_ptr, sp->cache))
			break;
		wl_ptr += sp->handles->size;
		label_ptr += m;
		D_ptr += m;
	}
	if (n < lb->dim) {
		vec_wl_free(weaklearner, n, sp->handles);
		return false;
	}
	return true;
}

bool wl_next(struct ada_item *item, void *adaboost, const flt_t vals[],
	     num_t vals_len)
{
	struct ada_wrap *wrap = adaboost;
	struct mvec_adaboost *mvec_ada = wrap->adaboost;
	if (wrap->t >= mvec_ada->group_len)
		return false;
	item->weaklearner = mvec_ada->weaklearner +
	    wrap->t * (wrap->wl_size * mvec_ada->dim);
	// 一组弱学习器对应一个系数
	item->alpha = mvec_ada->alpha + wrap->t;
	item->status = true;
	++wrap->t;

	return true;
}

enum ada_result get_vals(flt_t vals[], num_t vals_len, const void *weaklearner,
			 num_t m, const void *sample, const void *label,
			 const flt_t D[])
{
	const struct sp_wrap *sp = sample;
	const struct label_wrap *lb = label;
	const sample_t(*X)[sp->n] = sp->sample;
	// 计算 h(X[i])[l] * Y[i][l] 的值，l = 0, 1, ..., dim-1
	if (sp->handles->using_confident)
		VALS_CALC(vals, weaklearner, sp->handles->size, m, sp->n, X,
			  lb->labels, lb->dim, sp->handles->hypothesis.vec_cf);
	else
		VALS_CALC(vals, weaklearner, sp->handles->size, m, sp->n, X,
			  lb->labels, lb->dim, sp->handles->hypothesis.vec);
	flt_t err = 0;
	for (long_num_t i = 0; i < m * lb->dim; ++i)
		err += D[i] * (vals[i] < 0);
	if (err > 0.5)
		return ADA_FAILURE;
	else if (err == 0)
		return ADA_ALL_PASS;
	else
		return ADA_SUCCESS;
}

flt_t alpha_approx_wrap(const flt_t vals[], num_t vals_len, num_t m,
			const void *label, const flt_t D[])
{
	const struct label_wrap *lb = label;
	return alpha_approx(vals, vals_len, vals_len, lb->labels, D);
}

flt_t alpha_newton_wrap(const flt_t vals[], num_t vals_len, num_t m,
			const void *label, const flt_t D[])
{
	const struct label_wrap *lb = label;
	return alpha_newton(vals, vals_len, vals_len, lb->labels, D);
}

void init_D(flt_t D[], num_t m, const void *label)
{
	const struct label_wrap *lb = label;
	long_num_t n = lb->dim * m;
	flt_t val = (flt_t) 1.0 / n;
	for (long_num_t i = 0; i < n; ++i)
		D[i] = val;
}

void update_D(flt_t D[], flt_t vals[], num_t vals_len, num_t m,
	      const void *label, flt_t alpha)
{
	flt_t sum = 0;
	const struct label_wrap *lb = label;

	// 这里 D、vals、的长度都为 vals_len
	for (num_t i = 0; i < vals_len; ++i) {
		D[i] *= exp(-alpha * vals[i]);
		sum += D[i];
	}

	// 标准化
	for (num_t i = 0; i < vals_len; ++i)
		D[i] /= sum;
}

bool all_pass(struct ada_wrap *ada, const struct wl_handles *hl)
{
	struct mvec_adaboost *mvec_ada = ada->adaboost;
	turn_t n = mvec_ada->group_len - ada->t;
	turn_t t = ada->t;
	if (!all_pass_fold(ada, hl))
		return false;

	for (turn_t i = t; i < mvec_ada->group_len; ++i)
		mvec_ada->alpha[i] = mvec_ada->alpha[t - 1];
	return true;
}

bool all_pass_fold(struct ada_wrap *ada, const struct wl_handles *hl)
{
	struct mvec_adaboost *mvec_ada = ada->adaboost;
	turn_t group_size = hl->size * mvec_ada->dim;
	unsigned char *wl_src = mvec_ada->weaklearner +
	    (ada->t - 1) * group_size;
	unsigned char *wl_dst = wl_src + group_size;

	turn_t i;
	mlabel_t j;
	turn_t n = mvec_ada->group_len - ada->t;
	if (hl->copy != NULL) {
		for (i = 0; i < n; ++i)
			for (j = 0; j < mvec_ada->dim; ++j) {
				if (!hl->copy(wl_dst, wl_src + j * hl->size))
					goto err;
				wl_dst += hl->size;
			}
	} else
		for (i = 0; i < n; ++i) {
			memcpy(wl_dst, wl_src, group_size);
			wl_dst += group_size;
		}
	ada->t += n;
	return true;
err:
	// 失败时，仅保留完整的弱学习器组（若有 n 个分类，则一组有 n 个弱学习器
	vec_wl_free(wl_src + (i + 1) * group_size, j, hl);
	ada->t += i;
	return false;
}

bool init_label_wrap(struct label_wrap *lb, const mlabel_t * label, num_t m)
{
	// 计算所有可能的标签数量
	mlabel_t max = 0;
	for (num_t i = 0; i < m; ++i)
		if (label[i] > max)
			max = label[i];
	lb->dim = max + 1;
	// 对分类进行编码
	// 如 0 表为 -1-1-1...，1 表为 -1 1 -1...，2 表为 -1 -1 1...
	size_t size = sizeof(label_t) * m * (max + 1);
	if ((lb->labels = malloc(size)) == NULL)
		return false;
	memset(lb->labels, -1, size);
	label_t(*ptr)[m] = (label_t(*)[m]) lb->labels;
	for (num_t i = 0; i < m; ++i)
		ptr[label[i]][i] = 1;

	return true;
}

void free_label_wrap(struct label_wrap *lb)
{
	free(lb->labels);
}

void ada_hl_init(struct ada_handles *handles, num_t m, mlabel_t dim,
		 ada_alpha_fn get_alpha)
{
	handles->D_len = m * dim;
	handles->vals_len = m * dim;
	handles->train = wl_train;
	handles->get_vals = get_vals;
	handles->get_alpha = get_alpha;
	handles->next = wl_next;
	handles->init_D = init_D;
	handles->update_D = update_D;
}

mlabel_t argmax(flt_t output[], mlabel_t n)
{
	mlabel_t index;
	flt_t max = -DBL_MAX;
	for (mlabel_t i = 0; i < n; ++i)
		if (output[i] > max) {
			max = output[i];
			index = i;
		}

	return index;
}

bool train_framework(struct mvec_adaboost *adaboost, turn_t T, num_t m,
		     dim_t n, const sample_t X[m][n], const mlabel_t Y[],
		     bool cache_on, const struct wl_handles *handles,
		     ada_alpha_fn get_alpha, all_pass_fn all_pass)
{
	struct label_wrap lb;
	struct ada_handles ada_hl;
	struct train_setting st;

	if (!init_label_wrap(&lb, Y, m))
		return false;
	if (!init_setting(&st, adaboost, m, n, X, cache_on, handles))
		goto init_st_err;
	if (!mvec_ada_init(adaboost, T, lb.dim, false, handles))
		goto ada_init_err;
	ada_hl_init(&ada_hl, m, lb.dim, get_alpha);
	switch (ada_framework(&st.ada, m, &st.sp, &lb, &ada_hl)) {
	case ADA_FAILURE:
		goto train_err;
	case ADA_ALL_PASS:
		if (!all_pass(&st.ada, handles))
			goto train_err;
	case ADA_SUCCESS:
	default:
		break;
	}

	free_setting(&st);
	free_label_wrap(&lb);
	return true;

train_err:
	adaboost->group_len = st.ada.t;
	mvec_ada_free(adaboost, handles);
ada_init_err:
	free_setting(&st);
init_st_err:
	free_label_wrap(&lb);
	return false;
}
