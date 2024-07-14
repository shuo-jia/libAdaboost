#include <math.h>
#include <stdlib.h>
#include "vec_adaboost.h"
#include "vec_base_pvt.h"
#include "AlphaCalc/alpha.h"

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
// 初始化概率分布数组
static void init_D(flt_t D[], num_t m, const void *label);
// 更新概率分布数组
static void update_D(flt_t D[], flt_t vals[], num_t vals_len, num_t m,
		     const void *label, flt_t alpha);

/*
 * 初始化 Adaboost 回调函数集
 * handles: 需要初始化的 Adaboost 回调函数集
 * m: 样本数量
 * train: 函数指针，用于对单个弱学习器进行训练，见 adaboost_base.h 说明
 * get_vals: 计算中间值 Y[i] * h_t(x[i]) 并保存到数组中，见 adaboost_base.h 说明
 * get_alpha: 计算 alpha 的值，可为 AlphaCalc/alpha.h 中的函数
 * next: 函数指针，用于获取下一轮弱学习器及其系数的地址，见 adaboost_base.h 说明
 */
static inline void ada_hl_init(struct ada_handles *handles, num_t m,
			       ada_train_fn train, ada_vals_fn get_vals,
			       ada_alpha_fn get_alpha, ada_next_fn next);

/*
 * 初始化 Adaboost
 * ada: 指向未初始化的结构体
 * T: 训练轮数
 * using_fold: 表示是否将弱学习器系数并入弱学习器
 * handles: 弱学习器回调函数集合
 * 返回值：内存分配成功返回真，否则返回假
 */
static inline bool vec_ada_init(struct vec_adaboost *ada, turn_t T,
				bool using_fold,
				const struct wl_handles *handles);

// 当全部样本分类成功时将被调用，参数 ada 内将保存已训练的轮数
// 弱学习器系数不并入弱学习器
static bool all_pass(struct ada_wrap *ada, const struct wl_handles *hl);
// 弱学习器系数并入弱学习器，无需复制弱学习器系数
static bool all_pass_fold(struct ada_wrap *ada, const struct wl_handles *hl);

// 训练模板
// adaboost, ..., handles: 与 vec_ada_xxx() 系列函数参数意义相同
// get_alpha: 弱学习器系数计算函数（回调函数）
// all_pass: 样本全部分类成功时调用的回调函数（all_pass_fn 类型）
static inline bool train_framework(struct vec_adaboost *adaboost, turn_t T,
				   num_t m, dim_t n, const sample_t X[m][n],
				   const label_t Y[], bool cache_on,
				   const struct wl_handles *handles,
				   ada_alpha_fn get_alpha,
				   all_pass_fn all_pass);

/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
bool vec_ada_approx_train(struct vec_adaboost *adaboost, turn_t T, num_t m,
			  dim_t n, const sample_t X[m][n], const label_t Y[],
			  bool cache_on, const struct wl_handles *handles)
{
	return train_framework(adaboost, T, m, n, X, Y, cache_on, handles,
			       alpha_approx, all_pass);
}

bool vec_ada_fold_train(struct vec_adaboost *adaboost, turn_t T, num_t m,
			dim_t n, const sample_t X[m][n], const label_t Y[],
			bool cache_on, const struct wl_handles *handles)
{
	return train_framework(adaboost, T, m, n, X, Y, cache_on, handles,
			       alpha_eq_1, all_pass_fold);
}

bool vec_ada_newton_train(struct vec_adaboost *adaboost, turn_t T, num_t m,
			  dim_t n, const sample_t X[m][n], const label_t Y[],
			  bool cache_on, const struct wl_handles *handles)
{
	return train_framework(adaboost, T, m, n, X, Y, cache_on, handles,
			       alpha_newton, all_pass);
}

label_t vec_ada_h(const struct vec_adaboost *adaboost, const sample_t x[],
		  dim_t n, const struct wl_handles *handles)
{
	flt_t total = 0;
	const unsigned char *wl = adaboost->weaklearner;
	for (turn_t i = 0; i < adaboost->size; ++i, wl += handles->size)
		total += adaboost->alpha[i] * handles->hypothesis.vec(wl, x, n);
	return (total > 0) ? 1 : -1;
}

flt_t vec_ada_cf_h(const struct vec_adaboost *adaboost, const sample_t x[],
		   dim_t n, const struct wl_handles *handles)
{
	flt_t total = 0;
	const unsigned char *wl = adaboost->weaklearner;
	for (turn_t i = 0; i < adaboost->size; ++i, wl += handles->size)
		total += adaboost->alpha[i] * handles->hypothesis.vec(wl, x, n);
	return total;
}

label_t vec_ada_fold_h(const struct vec_adaboost *adaboost, const sample_t x[],
		       dim_t n, const struct wl_handles *handles)
{
	flt_t total = 0;
	const unsigned char *wl = adaboost->weaklearner;
	for (turn_t i = 0; i < adaboost->size; ++i, wl += handles->size)
		total += handles->hypothesis.vec_cf(wl, x, n);
	return (total > 0) ? 1 : -1;
}

flt_t vec_ada_fold_cf_h(const struct vec_adaboost *adaboost, const sample_t x[],
			dim_t n, const struct wl_handles *handles)
{
	flt_t total = 0;
	const unsigned char *wl = adaboost->weaklearner;
	for (turn_t i = 0; i < adaboost->size; ++i, wl += handles->size)
		total += handles->hypothesis.vec_cf(wl, x, n);
	return total;
}

bool vec_ada_read(struct vec_adaboost *adaboost, FILE * file,
		  const struct wl_handles *handles)
{
	turn_t t = 0;
	bool using_fold;
	if (fread(&using_fold, sizeof(bool), 1, file) < 1)
		return false;
	if (fread(&adaboost->size, sizeof(turn_t), 1, file) < 1)
		return false;
	if (!vec_ada_init(adaboost, adaboost->size, using_fold, handles))
		return false;
	if (!using_fold
	    && ALPHA_RW(adaboost->alpha, adaboost->size, file, fread))
		goto err;
	if ((t = vec_wl_read(adaboost->weaklearner, adaboost->size, handles,
			     file)) < adaboost->size)
		goto err;
	return true;
err:
	adaboost->size = t;	// 设置需释放内存的弱学习器数量
	vec_ada_free(adaboost, handles);
	return false;
}

bool vec_ada_write(const struct vec_adaboost *adaboost, FILE * file,
		   const struct wl_handles *handles)
{
	bool using_fold = (adaboost->alpha == NULL);
	if (fwrite(&using_fold, sizeof(bool), 1, file) < 1)
		return false;
	if (fwrite(&adaboost->size, sizeof(turn_t), 1, file) < 1)
		return false;
	if (!using_fold
	    && ALPHA_RW(adaboost->alpha, adaboost->size, file, fwrite))
		return false;
	if (vec_wl_write(adaboost->weaklearner, adaboost->size, handles, file)
	    < adaboost->size)
		return false;
	return true;
}

void *vec_ada_copy(struct vec_adaboost *dst, const struct vec_adaboost *src,
		   const struct wl_handles *handles)
{
	bool using_fold = (src->alpha == NULL);
	if (!vec_ada_init(dst, src->size, using_fold, handles))
		return NULL;

	if ((dst->size = vec_wl_copy(dst->weaklearner, src->weaklearner,
				     src->size, handles)) < src->size) {
		// dst->size 被置为已复制的弱学习器数量，可调用 vec_ada_free()
		vec_ada_free(dst, handles);
		return NULL;
	}

	if (!using_fold)	// 复制弱学习器系数
		ALPHA_COPY(dst->alpha, src->alpha, src->size);
	return dst;
}

void vec_ada_free(struct vec_adaboost *adaboost,
		  const struct wl_handles *handles)
{
	vec_wl_free(adaboost->weaklearner, adaboost->size, handles);
	free(adaboost->weaklearner);
	free(adaboost->alpha);
	adaboost->weaklearner = NULL;
	adaboost->alpha = NULL;
}

/*******************************************************************************
 * 				  静态函数定义
 ******************************************************************************/
bool wl_train(void *weaklearner, num_t m, const void *sample,
	      const void *label, const flt_t D[])
{
	const struct sp_wrap *sp = sample;
	return sp->handles->train.vec(weaklearner, m, sp->n, sp->sample, label,
				      D, sp->cache);
}

bool wl_next(struct ada_item *item, void *adaboost, const flt_t vals[],
	     num_t vals_len)
{
	struct ada_wrap *wrap = adaboost;
	struct vec_adaboost *vec_ada = wrap->adaboost;
	if (wrap->t >= vec_ada->size)
		return false;
	item->weaklearner = vec_ada->weaklearner + wrap->t * wrap->wl_size;
	item->alpha = vec_ada->alpha + wrap->t;
	item->status = true;
	++wrap->t;

	return true;
}

enum ada_result get_vals(flt_t vals[], num_t vals_len, const void *weaklearner,
			 num_t m, const void *sample, const void *label,
			 const flt_t D[])
{
	num_t i;
	const struct sp_wrap *sp = sample;
	const sample_t(*X)[sp->n] = sp->sample;
	const label_t *Y = label;
	if (sp->handles->using_confident)
		for (i = 0; i < m; ++i)
			vals[i] = sp->handles->hypothesis.vec_cf(weaklearner,
								 X[i],
								 sp->n) * Y[i];
	else
		for (i = 0; i < m; ++i)
			vals[i] = sp->handles->hypothesis.vec(weaklearner,
							      X[i],
							      sp->n) * Y[i];
	flt_t err = 0;
	for (i = 0; i < m; ++i)
		err += D[i] * (vals[i] < 0);
	if (err > 0.5)
		return ADA_FAILURE;
	else if (err == 0)
		return ADA_ALL_PASS;
	else
		return ADA_SUCCESS;
}

void init_D(flt_t D[], num_t m, const void *label)
{
	flt_t val = (flt_t) 1.0 / m;
	for (num_t i = 0; i < m; ++i)
		D[i] = val;
}

void update_D(flt_t D[], flt_t vals[], num_t vals_len, num_t m,
	      const void *label, flt_t alpha)
{
	num_t i;
	flt_t Z = 0;
	for (i = 0; i < m; ++i) {
		D[i] *= exp(-alpha * vals[i]);
		Z += D[i];
	}

	for (i = 0; i < m; ++i)
		D[i] /= Z;
	const label_t *Y = label;
}

void ada_hl_init(struct ada_handles *handles, num_t m, ada_train_fn train,
		 ada_vals_fn get_vals, ada_alpha_fn get_alpha, ada_next_fn next)
{
	handles->D_len = m;
	handles->vals_len = m;
	handles->train = train;
	handles->get_vals = get_vals;
	handles->get_alpha = get_alpha;
	handles->next = next;
	handles->init_D = init_D;
	handles->update_D = update_D;
}

bool vec_ada_init(struct vec_adaboost *ada, turn_t T, bool using_fold,
		  const struct wl_handles *handles)
{
	if ((ada->weaklearner = malloc(handles->size * T)) == NULL)
		return false;
	ada->alpha = NULL;
	if (!using_fold && (ada->alpha = malloc(sizeof(flt_t) * T)) == NULL) {
		free(ada->weaklearner);
		ada->weaklearner = NULL;
		return false;
	}
	ada->size = T;
	return true;
}

bool all_pass_fold(struct ada_wrap *ada, const struct wl_handles *hl)
{
	struct vec_adaboost *vec_ada = ada->adaboost;
	unsigned char *wl_src = vec_ada->weaklearner + (ada->t - 1) * hl->size;
	unsigned char *wl_dst = wl_src + hl->size;

	turn_t i;
	turn_t n = vec_ada->size - ada->t;
	if (hl->copy != NULL) {
		for (i = 0; i < n; ++i, wl_dst += hl->size)
			if (!hl->copy(wl_dst, wl_src))
				break;
	} else
		for (i = 0; i < n; ++i, wl_dst += hl->size)
			memcpy(wl_dst, wl_src, hl->size);
	ada->t += i;
	if (i < n)
		return false;
	return true;
}

bool all_pass(struct ada_wrap *ada, const struct wl_handles *hl)
{
	struct vec_adaboost *vec_ada = ada->adaboost;
	turn_t n = vec_ada->size - ada->t;
	turn_t t = ada->t;
	if (!all_pass_fold(ada, hl))
		return false;

	for (turn_t i = 0; i < vec_ada->size; ++i)
		vec_ada->alpha[i] = vec_ada->alpha[t - 1];
	return true;
}

bool train_framework(struct vec_adaboost *adaboost, turn_t T, num_t m,
		     dim_t n, const sample_t X[m][n], const label_t Y[],
		     bool cache_on, const struct wl_handles *handles,
		     ada_alpha_fn get_alpha, all_pass_fn all_pass)
{
	struct ada_handles ada_hl;
	struct train_setting st;
	ada_hl_init(&ada_hl, m, wl_train, get_vals, get_alpha, wl_next);
	if (!init_setting(&st, adaboost, m, n, X, cache_on, handles))
		return false;
	if (!vec_ada_init(adaboost, T, false, handles))
		goto ada_init_err;
	switch (ada_framework(&st.ada, m, &st.sp, Y, &ada_hl)) {
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
	return true;

train_err:
	adaboost->size = st.ada.t;
	vec_ada_free(adaboost, handles);
ada_init_err:
	free_setting(&st);
	return false;
}
