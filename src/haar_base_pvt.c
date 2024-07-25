#include <math.h>
#include <stdlib.h>
#include "haar_base_pvt.h"

/**
 * \file haar_base_pvt.c
 * \brief haar_base 的私有部分，包括函数实现
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/// 回调函数类型：此类函数将样本集在指定弱学习器上的输出保存到数组中
typedef void (*output_fn)(flt_t *, num_t, const void *, const struct sp_wrap *);

/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
/// struct ada_handles 的回调函数，对弱学习器进行训练
static bool wl_train(void *weaklearner, num_t m, const void *sample,
		     const void *label, const flt_t D[]);

/// 用于 qsort 比较（struct sort_item * 指针比较）
static int sort_cmp(const void *item1, const void *item2);

/// 将样本集在弱学习器 wl 上的输出保存到 vals 数组中（输出值不带置信度）
static void wl_output(flt_t vals[], num_t vals_len, const void *wl,
		      const struct sp_wrap *sp);

/// 将样本集在弱学习器 wl 上的输出保存到 vals 数组中（输出值带置信度）
static void wl_output_cf(flt_t vals[], num_t vals_len, const void *wl,
			 const struct sp_wrap *sp);

/// 将 h(X[i]) * alpha 的值保存到 vals 数组中，wl 为 struct haar_wl * 类型
static void wl_alpha(flt_t vals[], num_t vals_len, const void *wl,
			 const struct sp_wrap *sp);

/// get_vals() 函数框架，使用回调函数 op() 改变行为；op 可为 wl_output 或 wl_output_cf
static enum ada_result get_vals_framework(flt_t vals[], num_t vals_len,
					  const void *weaklearner, num_t m,
					  const void *sample, const void *label,
					  const flt_t D[], output_fn op);

/// all_pass 函数框架，使用回调函数 op() 改变行为；op 可为 wl_alpha 或 wl_output_cf
static bool all_pass_framework (struct train_setting *st, output_fn op);

/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
enum ada_result haar_get_vals(flt_t vals[], num_t vals_len,
			      const void *weaklearner, num_t m,
			      const void *sample, const void *label,
			      const flt_t D[])
{
	return get_vals_framework(vals, vals_len, weaklearner, m, sample, label,
				  D, wl_output);
}

enum ada_result haar_get_vals_cf(flt_t vals[], num_t vals_len,
			      const void *weaklearner, num_t m,
			      const void *sample, const void *label,
			      const flt_t D[])
{
	return get_vals_framework(vals, vals_len, weaklearner, m, sample, label,
				  D, wl_output_cf);
}

void ada_hl_init(struct ada_handles *ada_hl, num_t l, num_t m,
		 ada_vals_fn get_vals, ada_alpha_fn get_alpha, ada_next_fn next,
		 ada_init_D_fn init_D, ada_update_D_fn update_D)
{
	ada_hl->D_len = m;
	ada_hl->vals_len = l + m;
	ada_hl->train = wl_train;
	ada_hl->get_vals = get_vals;
	ada_hl->get_alpha = get_alpha;
	ada_hl->next = next;
	ada_hl->init_D = init_D;
	ada_hl->update_D = update_D;
}

bool init_setting(struct train_setting *st, struct haar_adaboost *adaboost,
		  flt_t d, flt_t f, num_t l, imgsz_t h, imgsz_t w,
		  const sample_t * const X[], const sample_t * const X2[],
		  const label_t Y[], const struct wl_handles *wl_hl)
{
	st->sp.l = l;
	st->sp.h = h;
	st->sp.w = w;
	st->sp.X = X;
	st->sp.X2 = X2;
	st->sp.handles = wl_hl;

	st->ada.adaboost = adaboost;
	if ((st->ada.output = malloc(sizeof(struct sort_item) * l)) == NULL)
		return false;
	if ((st->ada.op_ptrs = malloc(sizeof(struct sort_item *) * l)) == NULL) {
		free(st->ada.output);
		return false;
	}
	st->ada.positive_ct = 0;
	for (num_t i = 0; i < l; ++i) {
		st->ada.output[i].id = i;
		st->ada.output[i].val = 0;
		st->ada.op_ptrs[i] = &st->ada.output[i];
		if (Y[i] > 0)
			++st->ada.positive_ct;
	}
	st->ada.l = l;
	st->ada.d = d;
	st->ada.f = f;
	st->ada.Y = Y;
	st->ada.wl_size = wl_hl->size;
	return true;
}

void free_setting(struct train_setting *st)
{
	free(st->ada.output);
	free(st->ada.op_ptrs);
}

void get_ratio(flt_t * d, flt_t * f, struct ada_wrap *ada, const flt_t vals[])
/*
{
	num_t i;
	num_t false_n = 0;		// 假阴性样本数量
	num_t max_f_n;			// 所允许的最大假阴性样本数量
	max_f_n = ada->positive_ct * (1 - ada->d);

	// 计算当前分类结果（vals数组表示 alpha * h(X[i])）
	for (num_t i = 0; i < ada->l; ++i)
		ada->output[i].val += vals[i];
	// 寻找满足刚好达到最大假阴性样本数量的数组位置（即 i-1 到 i 之间）
	qsort (ada->op_ptrs, ada->l, sizeof(struct sort_item *), sort_cmp);
	for (i = 0; i < ada->l; ++i)
		if (ada->Y[ada->op_ptrs[i]->id] > 0 && ++false_n > max_f_n)
			break;

	// 最大假阴性，意味着刚好达到最小检测率，在此位置设置正、负例样本阈值
	if (i == 0)
		ada->adaboost->threshold = ada->op_ptrs[0]->val - MIN_INTERVAL;
	else if (i < ada->l)
		ada->adaboost->threshold = (ada->op_ptrs[i]->val +
					    ada->op_ptrs[i-1]->val) / 2;
	else
		ada->adaboost->threshold = ada->op_ptrs[ada->l-1]->val
					   + MIN_INTERVAL;

	// 检测率、假阳率计算
	if (ada->positive_ct > 0)
		*d = (flt_t)(ada->positive_ct + 1 - false_n)
			  / ada->positive_ct;
	else
		*d = 1;
	if (ada->l > ada->positive_ct)
		*f = (flt_t)(ada->l + false_n + 1 - i
			       -ada->positive_ct) / (ada->l - ada->positive_ct);
	else
		*f = 0;

}
*/
{
	num_t i;
	num_t det_ct = 0;	// 真阳性样本数量
	num_t min_det;		// 真阳性样本最少数量
	min_det = ceil(ada->positive_ct * ada->d);

	// 计算当前分类结果（vals数组表示 alpha * h(X[i])）
	for (i = 0; i < ada->l; ++i)
		ada->output[i].val += vals[i];
	// 寻找刚好满足检测率的位置（即 i-1 到 i 之间）
	qsort(ada->op_ptrs, ada->l, sizeof(struct sort_item *), sort_cmp);
	for (i = ada->l - 1; i > 0; --i)
		if (ada->Y[ada->op_ptrs[i]->id] > 0 && ++det_ct >= min_det)
			break;
	if (i == 0 && ada->Y[ada->op_ptrs[i]->id] > 0)
		++det_ct;
	// 用作属性划分，i-1 与 i 处的输出值必须不同
	while (i > 0 && ada->op_ptrs[i]->val == ada->op_ptrs[i - 1]->val)
		if (ada->Y[ada->op_ptrs[--i]->id] > 0)
			++det_ct;
	// 在此位置设置正、负例样本阈值
	if (i > 0)
		ada->adaboost->threshold = (ada->op_ptrs[i]->val +
					    ada->op_ptrs[i - 1]->val) / 2;
	else			// i == 0 的情形
		ada->adaboost->threshold = ada->op_ptrs[0]->val - MIN_INTERVAL;

	// 检测率、假阳率计算
	if (ada->positive_ct > 0)
		*d = (flt_t) det_ct / ada->positive_ct;
	else
		*d = 1;
	if (ada->l > ada->positive_ct)
		*f = (flt_t) (ada->l - i - det_ct) / (ada->l -
						      ada->positive_ct);
	else
		*f = 0;
}

bool haar_all_pass (struct train_setting *st)
{
	return all_pass_framework(st, wl_alpha);
}

bool haar_all_pass_cf (struct train_setting *st)
{
	return all_pass_framework(st, wl_output_cf);
}

/*******************************************************************************
 * 				  静态函数定义
 ******************************************************************************/
bool wl_train(void *weaklearner, num_t m, const void *sample,
	      const void *label, const flt_t D[])
{
	const struct sp_wrap *sp = sample;
	return sp->handles->train.haar(weaklearner, m, sp->h, sp->w,
				       sp->X + sp->l, sp->X2 + sp->l, label, D);
}

int sort_cmp(const void *item1, const void *item2)
{
	flt_t result = (*(struct sort_item **)item1)->val -
	    (*(struct sort_item **)item2)->val;
	if (result < 0)
		return -1;
	else if (result > 0)
		return 1;
	else
		return 0;
}

void wl_output(flt_t vals[], num_t vals_len, const void *wl,
	       const struct sp_wrap *sp)
{
	for (num_t i = 0; i < vals_len; ++i)
		vals[i] =
		    sp->handles->hypothesis.haar(wl, sp->h, sp->w, sp->w,
						 (void *)sp->X[i],
						 (void *)sp->X2[i], 1);
}

void wl_output_cf(flt_t vals[], num_t vals_len, const void *wl,
		  const struct sp_wrap *sp)
{
	for (num_t i = 0; i < vals_len; ++i)
		vals[i] =
		    sp->handles->hypothesis.haar_cf(wl, sp->h, sp->w, sp->w,
						    (void *)sp->X[i],
						    (void *)sp->X2[i], 1);
}

void wl_alpha(flt_t vals[], num_t vals_len, const void *weaklearner,
	      const struct sp_wrap *sp)
{
	const struct haar_wl *wl = weaklearner;
	for (num_t i = 0; i < vals_len; ++i)
		vals[i] =
		    sp->handles->hypothesis.haar(wl->weaklearner, sp->h, sp->w,
						 sp->w, (void *)sp->X[i],
						 (void *)sp->X2[i],
						 1) * wl->alpha;
}

enum ada_result get_vals_framework(flt_t vals[], num_t vals_len,
				   const void *weaklearner, num_t m,
				   const void *sample, const void *label,
				   const flt_t D[], output_fn op)
{
	num_t i;
	const struct sp_wrap *sp = sample;
	const label_t *Y = label;
	op(vals, vals_len, weaklearner, sp);	// 计算 h(X[i])

	flt_t err = 0;
	for (i = sp->l; i < vals_len; ++i) {	// 在训练集计算 h(X[i]) * Y[i]
		vals[i] *= Y[i - sp->l];
		if (vals[i] <= 0)
			err += D[i - sp->l];
	}

	if (err == 0)
		return ADA_ALL_PASS;
	else if (err < 0.5)
		return ADA_SUCCESS;
	else
		return ADA_FAILURE;
}

bool all_pass_framework (struct train_setting *st, output_fn op)
{
	flt_t * vals = malloc (sizeof(flt_t) * st->ada.l);
	if (vals == NULL)
		return false;
	op(vals, st->ada.l, st->ada.adaboost->wl.end_ptr->data, &st->sp);
	get_ratio (&(st->ada.d), &(st->ada.f), &st->ada, vals);         
	free (vals);
	return true;
}
