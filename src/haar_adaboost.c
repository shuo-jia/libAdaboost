#include <math.h>
#include <stdlib.h>
#include "haar_base_pvt.h"
#include "AlphaCalc/alpha.h"
/**
 * \file haar_adaboost.c
 * \brief 基于哈尔特征的 Adaboost 分类器函数定义--子类（增加训练方法）。
 *	主要实现 Paul Viola, Michael Jones 于 2001 年给出的方法
 *	（Robust Real-time Object Detection）
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
/// struct ada_handles 的回调函数，初始化概率分布（正例样本总概率=负例样本总概率）
static void init_D(flt_t D[], num_t m, const void *label);

/// struct ada_handles 的回调函数，更新概率分布。
/** 使用常规方法更新概率分布，并将中间值 vals 数组置为 alpha * h(X[i]), 
 * i = 0, 1, ..., vals_len - 1 */
void update_D(flt_t D[], flt_t vals[], num_t vals_len, num_t m,
	      const void *label, flt_t alpha);

/// struct ada_handles 的回调函数，计算检测率、假阳率，据此判断是否继续训练
static bool wl_next(struct ada_item *item, void *adaboost,
		    const flt_t vals[], num_t vals_len);

/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
bool haar_ada_approx_train(struct haar_adaboost *adaboost, flt_t * d,
			   flt_t * f, num_t l, num_t m, imgsz_t h, imgsz_t w,
			   const sample_t * X[], const sample_t * X2[],
			   const label_t Y[], const struct wl_handles *handles)
{
	struct ada_handles ada_hl;
	ada_hl_init(&ada_hl, l, m, haar_get_vals, alpha_approx, wl_next, init_D,
		    update_D);
	return train_framework(adaboost, d, f, l, m, h, w, X, X2, Y,
			       haar_all_pass, handles, &ada_hl);
}

bool haar_ada_newton_train(struct haar_adaboost *adaboost, flt_t * d,
			   flt_t * f, num_t l, num_t m, imgsz_t h, imgsz_t w,
			   const sample_t * X[], const sample_t * X2[],
			   const label_t Y[], const struct wl_handles *handles)
{
	struct ada_handles ada_hl;
	ada_hl_init(&ada_hl, l, m, haar_get_vals, alpha_newton, wl_next, init_D,
		    update_D);
	return train_framework(adaboost, d, f, l, m, h, w, X, X2, Y,
			       haar_all_pass_cf, handles, &ada_hl);
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
