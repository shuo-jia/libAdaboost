#include <math.h>
#include <stdlib.h>
#include "haar_base_pvt.h"
#include "AlphaCalc/alpha.h"

/*******************************************************************************
 * 				   宏函数定义
 ******************************************************************************/
// op_val: （宏）函数名，接收 st、样本索引 i 作为参数，返回 h(X[i]) * alpha 的值
#define OP_VAL(st, i)								\
({										\
	const struct haar_wl * wl = st->ada.adaboost->wl.end_ptr->data;		\
	st->sp.handles->hypothesis.haar((wl)->weaklearner, (st)->sp.h,		\
			(st)->sp.w, (st)->sp.w, (void *)(st)->sp.X[i],		\
			(void *)(st)->sp.X2[i], 1) * wl->alpha;			\
 })

/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
// struct ada_handles 的回调函数
// 对于验证集（前 l 个元素），计算弱学习器输出值 h(X[i])
// 对于训练集（后 m 各元素），计算 h(X[i]) * Y[i]
static enum ada_result get_vals(flt_t vals[], num_t vals_len,
				const void *weaklearner, num_t m,
				const void *sample, const void *label,
				const flt_t D[]);

// struct ada_handles 的回调函数，初始化概率分布（正例样本总概率=负例样本总概率）
static void init_D(flt_t D[], num_t m, const void *label);

// struct ada_handles 的回调函数，使用常规方法更新概率分布，
// 对于验证集（前 l 个元素），计算 alpha * h(X[i])
// 对于训练集（后 m 各元素），计算 alpha * h(X[i]) * Y[i]
void update_D(flt_t D[], flt_t vals[], num_t vals_len, num_t m,
	      const void *label, flt_t alpha);

// struct ada_handles 的回调函数，计算检测率、假阳率，据此判断是否继续训练
static bool wl_next(struct ada_item *item, void *adaboost,
		    const flt_t vals[], num_t vals_len);

// 当全部训练样本分类成功时执行的函数
// 判断验证集的假阳率、检测率是否满足要求
static bool all_pass(struct train_setting *st);

/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
bool haar_ada_approx_train(struct haar_adaboost *adaboost, flt_t * d,
			   flt_t * f, num_t l, num_t m, imgsz_t h, imgsz_t w,
			   const sample_t * X[], const sample_t * X2[],
			   const label_t Y[], const struct wl_handles *handles)
{
	struct ada_handles ada_hl;
	ada_hl_init(&ada_hl, l, m, get_vals, alpha_approx, wl_next, init_D,
		    update_D);
	return train_framework(adaboost, d, f, l, m, h, w, X, X2, Y, all_pass,
			       handles, &ada_hl);
}

bool haar_ada_newton_train(struct haar_adaboost *adaboost, flt_t * d,
			   flt_t * f, num_t l, num_t m, imgsz_t h, imgsz_t w,
			   const sample_t * X[], const sample_t * X2[],
			   const label_t Y[], const struct wl_handles *handles)
{
	struct ada_handles ada_hl;
	ada_hl_init(&ada_hl, l, m, get_vals, alpha_newton, wl_next, init_D,
		    update_D);
	return train_framework(adaboost, d, f, l, m, h, w, X, X2, Y, all_pass,
			       handles, &ada_hl);
}

flt_t haar_ada_h(const struct haar_adaboost *adaboost, imgsz_t h, imgsz_t w,
		 imgsz_t wid, const sample_t x[h][wid],
		 const sample_t x2[h][wid], flt_t scale,
		 const struct wl_handles *handles)
{
	flt_t total = 0;
	struct haar_wl *wl;
	link_iter iter = link_list_start_iter(&adaboost->wl);
	while (link_list_check_end(iter)) {
		wl = link_list_get_data(iter);
		total += wl->alpha * handles->hypothesis.haar(wl->weaklearner,
							      h, w, wid, x, x2,
							      scale);
		link_list_next_iter(&iter);
	}

	return total - adaboost->threshold;
}

/*******************************************************************************
 * 				  静态函数定义
 ******************************************************************************/
// 中间值数组 vals 将保存弱学习器输出值 h(X[i])
enum ada_result get_vals(flt_t vals[], num_t vals_len,
			 const void *weaklearner, num_t m, const void *sample,
			 const void *label, const flt_t D[])
{
	num_t i;
	const struct sp_wrap *sp = sample;
	const label_t *Y = label;
	for (i = 0; i < sp->l; ++i)	// 验证集输出值
		vals[i] = sp->handles->hypothesis.haar(weaklearner, sp->h,
						       sp->w, sp->w,
						       (void *)sp->X[i],
						       (void *)sp->X2[i], 1);

	flt_t err = 0;
	for (i = sp->l; i < vals_len; ++i) {	// 计算训练集输出值 * Y[i]
		vals[i] =
		    sp->handles->hypothesis.haar(weaklearner, sp->h, sp->w,
						 sp->w, (void *)sp->X[i],
						 (void *)sp->X2[i],
						 1) * Y[i - sp->l];
		if (vals[i] <= 0)	// 训练误差计算
			err += D[i - sp->l];
	}
#ifdef LOG
	printf("Weaklearner error rate: %f\n", err);
#endif
	if (err == 0)
		return ADA_ALL_PASS;
	else if (err < 0.5)
		return ADA_SUCCESS;
	else
		return ADA_FAILURE;
}

void init_D(flt_t D[], num_t m, const void *label)
{
	num_t i;
	num_t positive_ct = 0;
	const label_t *Y = label;
	for (i = 0; i < m; ++i)
		if (Y[i] > 0)
			++positive_ct;
	for (i = 0; i < m; ++i)
		if (Y[i] > 0)
			D[i] = (flt_t) 1.0 / (2 * positive_ct);
		else
			D[i] = (flt_t) 1.0 / (2 * (m - positive_ct));
}

void update_D(flt_t D[], flt_t vals[], num_t vals_len, num_t m,
	      const void *label, flt_t alpha)
{
	num_t i;
	num_t start = vals_len - m;
	flt_t Z = 0;
	const label_t *Y = label;

	// 使用弱学习器系数调整分类结果
	for (i = 0; i < vals_len; ++i)
		vals[i] *= alpha;
	// 更新训练集分布概率
	for (i = 0; i < m; ++i) {
		D[i] *= exp(-vals[start + i]);
		Z += D[i];
	}

	for (i = 0; i < m; ++i)
		D[i] /= Z;
}

bool wl_next(struct ada_item *item, void *adaboost, const flt_t vals[],
	     num_t vals_len)
{
	flt_t det_rto, fal_pos_rto;
	struct ada_wrap *ada = adaboost;
	if (link_list_size(&ada->adaboost->wl) > 0) {
		get_ratio(&det_rto, &fal_pos_rto, adaboost, vals);
#ifdef LOG
		printf("Current AdaBoost false positive ratio: %f\n",
		       fal_pos_rto);
		printf("Target AdaBoost false positive ratio: %f\n", ada->f);
#endif
		// 达到训练要求，结束训练
		if (fal_pos_rto <= ada->f) {
			ada->d = det_rto;
			ada->f = fal_pos_rto;
			return false;
		}
	}
#ifdef LOG
	printf("Create new weaklearner.\n");
#endif
	// 创建新的弱学习器，准备下一轮训练
	struct haar_wl *wl = malloc(sizeof(struct haar_wl) + ada->wl_size);
	if (wl == NULL)
		goto malloc_err;
	if (!link_list_append(&ada->adaboost->wl, wl))
		goto link_list_err;
	item->alpha = &wl->alpha;
	item->weaklearner = wl->weaklearner;
	item->status = true;
	return true;

link_list_err:
	free(wl);
malloc_err:
	item->status = false;
	return true;
}

bool all_pass(struct train_setting *st)
{
	return ALL_PASS(st, OP_VAL);
}
