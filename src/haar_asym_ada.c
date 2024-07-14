#include <math.h>
#include <stdlib.h>
#include "haar_asym_ada.h"
#include "haar_base_pvt.h"
#include "AlphaCalc/alpha.h"

/*******************************************************************************
 * 				   宏函数定义
 ******************************************************************************/
// op_val: （宏）函数名，接收 st、样本索引 i 作为参数，返回 h(X[i]) * alpha 的值
#define OP_VAL(st, i)								\
({										\
	void * wl = st->ada.adaboost->wl.end_ptr->data;				\
	st->sp.handles->hypothesis.haar_cf(wl, (st)->sp.h, (st)->sp.w,		\
			(st)->sp.w, (void *)(st)->sp.X[i],			\
			(void *)(st)->sp.X2[i], 1);				\
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

// struct ada_handles 的回调函数，初始化概率分布（加入非对称损失）
static void init_D(flt_t D[], num_t m, const void *label);

// 改进的概率分布初始化方法，避免非对称损失的效果迅速消失
static void init_D_imp(flt_t D[], num_t m, const void *label);

// struct ada_handles 的回调函数，使用常规方法更新概率分布
// 与 init_D() 一同使用
static void update_D(flt_t D[], flt_t vals[], num_t vals_len, num_t m,
		     const void *label, flt_t alpha);

// 改进的概率分布更新方法，与 init_D_imp() 一同使用
static void update_D_imp(flt_t D[], flt_t vals[], num_t vals_len, num_t m,
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
bool haar_ada_asym_train(struct haar_adaboost *adaboost, flt_t * d, flt_t * f,
			 num_t l, num_t m, imgsz_t h, imgsz_t w,
			 const sample_t * X[], const sample_t * X2[],
			 const label_t Y[], const struct wl_handles *handles)
{
	struct ada_handles ada_hl;
	ada_hl_init(&ada_hl, l, m, get_vals, alpha_eq_1, wl_next, init_D,
		    update_D);
	return train_framework(adaboost, d, f, l, m, h, w, X, X2, Y, all_pass,
			       handles, &ada_hl);
}

bool haar_ada_asym_imp_train(struct haar_adaboost *adaboost, flt_t * d,
			     flt_t * f, num_t l, num_t m, imgsz_t h, imgsz_t w,
			     const sample_t * X[], const sample_t * X2[],
			     const label_t Y[],
			     const struct wl_handles *handles)
{
	struct ada_handles ada_hl;
	ada_hl_init(&ada_hl, l, m, get_vals, alpha_eq_1, wl_next, init_D_imp,
		    update_D_imp);
	return train_framework(adaboost, d, f, l, m, h, w, X, X2, Y, all_pass,
			       handles, &ada_hl);
}

flt_t haar_ada_fold_h(const struct haar_adaboost *adaboost, imgsz_t h,
		      imgsz_t w, imgsz_t wid, const sample_t x[h][wid],
		      const sample_t x2[h][wid], flt_t scale,
		      const struct wl_handles *handles)
{
	flt_t total = 0;
	void *wl;
	link_iter iter = link_list_start_iter(&adaboost->wl);
	while (link_list_check_end(iter)) {
		wl = link_list_get_data(iter);
		total +=
		    handles->hypothesis.haar_cf(wl, h, w, wid, x, x2, scale);
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
	for (i = 0; i < vals_len; ++i)		// 验证集和训练集的输出值
		vals[i] = sp->handles->hypothesis.haar_cf(weaklearner, sp->h,
							  sp->w, sp->w,
							  (void *)sp->X[i],
							  (void *)sp->X2[i], 1);

	flt_t err = 0;
	for (i = sp->l; i < vals_len; ++i) {	// 训练误差计算
		vals[i] *= Y[i - sp->l];	// 训练集置为 h(X[i]) * Y[i]
		if (vals[i] < 0)
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
	flt_t Z = 0;
	const flt_t val_p = sqrt(ASYM_CONST);
	const flt_t val_n = 1.0 / val_p;
	const label_t *Y = label;
	for (i = 0; i < m; ++i) {
		D[i] = (Y[i] > 0) ? val_p : val_n;
		Z += D[i];
	}

	for (i = 0; i < m; ++i)
		D[i] /= Z;
}

void init_D_imp(flt_t D[], num_t m, const void *label)
{
	num_t i;
	flt_t Z = 0;
	const flt_t val_p = pow(ASYM_CONST, 1.0 / (2 * ASYM_TURN));
	const flt_t val_n = 1.0 / val_p;
	const label_t *Y = label;
	for (i = 0; i < m; ++i) {
		D[i] = (Y[i] > 0) ? val_p : val_n;
		Z += D[i];
	}

	for (i = 0; i < m; ++i)
		D[i] /= Z;
}

void update_D(flt_t D[], flt_t vals[], num_t vals_len, num_t m,
	      const void *label, flt_t alpha)
{
	num_t i;
	num_t start = vals_len - m;
	flt_t Z = 0;
	const label_t *Y = label;

	// 更新训练集分布概率
	for (i = 0; i < m; ++i) {
		D[i] *= exp(-vals[start + i]);
		Z += D[i];
	}

	for (i = 0; i < m; ++i)
		D[i] /= Z;
}

void update_D_imp(flt_t D[], flt_t vals[], num_t vals_len, num_t m,
		  const void *label, flt_t alpha)
{
	num_t i;
	num_t start = vals_len - m;
	flt_t Z = 0;
	const label_t *Y = label;
	const flt_t val_p = pow(ASYM_CONST, 1.0 / (2 * ASYM_TURN));
	const flt_t val_n = 1.0 / val_p;

	// 更新训练集分布概率
	for (i = 0; i < m; ++i) {
		D[i] *= exp(-vals[start + i]);
		D[i] *= (Y[i] > 0) ? val_p : val_n;
		Z += D[i];
	}

	for (i = 0; i < m; ++i)
		D[i] /= Z;
}

bool wl_next(struct ada_item *item, void *adaboost, const flt_t vals[],
	     num_t vals_len)
{
	static flt_t alpha;
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
	unsigned char *wl = malloc(ada->wl_size);
	if (wl == NULL)
		goto malloc_err;
	if (!link_list_append(&ada->adaboost->wl, wl))
		goto link_list_err;
	item->weaklearner = wl;
	item->alpha = &alpha;
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
