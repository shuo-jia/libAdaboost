#include "adaboost.h"
#include "mvec_hloss.h"
#include "haar_adaboost.h"
#include "haar_asym_ada.h"

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
// 回调函数类型：弱学习器回调函数集初始化
typedef void (*wl_setting_fn)(struct wl_handles * handles);

/*******************************************************************************
 * 				    静态变量
 ******************************************************************************/
// 决策树桩回调函数集初始化方法
static wl_setting_fn wl_set_vec_arr[ADA_WL_END][2] = {
	[ADA_CONTINUOUS] = { wl_set_vec_cstump, wl_set_vec_cstump_cf },
	[ADA_DISCRETE] = { wl_set_vec_dstump, wl_set_vec_dstump_cf },
};

// struct vec_adaboost 的训练方法
static vec_ada_train_fn vec_ada_train_arr[ADA_ALPHA_END] = {
	[ADA_APPROX] = vec_ada_approx_train,
	[ADA_FOLD] = vec_ada_fold_train,
	[ADA_NEWTON] = vec_ada_newton_train,
};

// struct vec_adaboost 的假设器
static void *vec_ada_h_arr[ADA_H_END][2] = {
	[ADA_NO_CONFIDENT] = { vec_ada_h, vec_ada_fold_h },
	[ADA_CONFIDENT] = { vec_ada_cf_h, vec_ada_fold_cf_h },
};

// struct mvec_adaboost 的训练方法
static mvec_ada_train_fn mvec_ada_train_arr[ADA_MVEC_END][ADA_ALPHA_END] = {
	[ADA_HLOSS] = {
		       [ADA_APPROX] = mvec_ada_approx_train,
		       [ADA_FOLD] = mvec_ada_fold_train,
		       [ADA_NEWTON] = mvec_ada_newton_train,
		        },
};

// struct haar_adaboost 的训练方法
static haar_ada_train_fn haar_ada_train_arr[ADA_HAAR_END] = {
	[ADA_NM_APPROX] = haar_ada_approx_train,
	[ADA_NM_NEWTON] = haar_ada_newton_train,
	[ADA_ASYM] = haar_ada_asym_train,
	[ADA_ASYM_IMP] = haar_ada_asym_imp_train,
};

/*******************************************************************************
 * 			    vec_ada_handles 函数定义
 ******************************************************************************/
void ada_set_vec(struct vec_ada_handles *handles, enum ada_alpha_t alpha_type,
		 enum ada_hypothesis_t h_type, enum ada_wl_t wl_type)
{
	extern vec_ada_train_fn vec_ada_train_arr[ADA_ALPHA_END];
	extern void *vec_ada_h_arr[ADA_H_END][2];
	extern wl_setting_fn wl_set_vec_arr[ADA_WL_END][2];

	handles->train = vec_ada_train_arr[alpha_type];
	switch (h_type) {
	case ADA_NO_CONFIDENT:
		handles->hypothesis.h = vec_ada_h_arr[h_type]
		    [alpha_type == ADA_FOLD];
		break;
	case ADA_CONFIDENT:
		handles->hypothesis.cf_h = vec_ada_h_arr[h_type]
		    [alpha_type == ADA_FOLD];
		break;
	case ADA_H_END:
		break;
	}

	wl_set_vec_arr[wl_type][alpha_type == ADA_FOLD] (&handles->wl_hl);
	handles->read = vec_ada_read;
	handles->write = vec_ada_write;
	handles->copy = vec_ada_copy;
	handles->free = vec_ada_free;
}

/*******************************************************************************
 * 			   mvec_ada_handles 函数声明
 ******************************************************************************/
void ada_set_mvec(struct mvec_ada_handles *handles, enum ada_mvec_t mvec_type,
		  enum ada_alpha_t alpha_type, enum ada_wl_t wl_type)
{
	extern mvec_ada_train_fn
	    mvec_ada_train_arr[ADA_MVEC_END][ADA_ALPHA_END];
	extern wl_setting_fn wl_set_vec_arr[ADA_WL_END][2];
	handles->train = mvec_ada_train_arr[mvec_type][alpha_type];
	handles->h = (alpha_type == ADA_FOLD) ? mvec_ada_fold_h : mvec_ada_h;
	wl_set_vec_arr[wl_type][alpha_type == ADA_FOLD] (&handles->wl_hl);
	handles->read = mvec_ada_read;
	handles->write = mvec_ada_write;
	handles->copy = mvec_ada_copy;
	handles->free = mvec_ada_free;
}

/*******************************************************************************
 * 			   haar_ada_handles 函数声明
 ******************************************************************************/
void ada_set_haar(struct haar_ada_handles *handles, enum ada_haar_t haar_type,
		  enum ada_wl_train_t wl_train_type)
{
	extern haar_ada_train_fn haar_ada_train_arr[ADA_HAAR_END];

	handles->train = haar_ada_train_arr[haar_type];
	switch (haar_type) {
	case ADA_NM_APPROX:
	case ADA_NM_NEWTON:
		handles->h = haar_ada_h;
		if (wl_train_type == ADA_OPT)
			wl_set_haar(&handles->wl_hl);
		else
			wl_set_haar_ga(&handles->wl_hl);
		break;
	case ADA_ASYM:
	case ADA_ASYM_IMP:
		handles->h = haar_ada_fold_h;
		if (wl_train_type == ADA_OPT)
			wl_set_haar_cf(&handles->wl_hl);
		else
			wl_set_haar_ga_cf(&handles->wl_hl);
	case ADA_HAAR_END:
		break;
	}
	handles->read = haar_ada_read;
	handles->write = haar_ada_write;
	handles->copy = haar_ada_copy;
	handles->free = haar_ada_free;
}
