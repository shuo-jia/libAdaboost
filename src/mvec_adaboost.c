#include <stdio.h>
#include <stdlib.h>
#include "vec_base_pvt.h"
#include "mvec_adaboost.h"
/**
 * \file mvec_adaboost.c
 * \brief Adaboost 分类器函数实现：多分类任务、样本集为向量集
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
bool mvec_ada_init(struct mvec_adaboost *ada, turn_t group_len, mlabel_t dim,
		   bool using_fold, const struct wl_handles *handles)
{
	turn_t total = group_len * dim;
	if ((ada->weaklearner = malloc(handles->size * total)) == NULL)
		return false;
	ada->alpha = NULL;
	if (!using_fold && (ada->alpha = malloc(sizeof(flt_t) * group_len))
	    == NULL) {
		free(ada->weaklearner);
		ada->weaklearner = NULL;
		return false;
	}
	ada->group_len = group_len;
	ada->dim = dim;
	return true;
}

bool mvec_ada_read(struct mvec_adaboost *adaboost, FILE * file,
		   const struct wl_handles *handles)
{
	turn_t t = 0;
	bool using_fold;
	if (fread(&using_fold, sizeof(bool), 1, file) < 1)
		return false;
	if (fread(&adaboost->group_len, sizeof(turn_t), 1, file) < 1)
		return false;
	if (fread(&adaboost->dim, sizeof(mlabel_t), 1, file) < 1)
		return false;
	if (!mvec_ada_init(adaboost, adaboost->group_len, adaboost->dim,
			   using_fold, handles))
		return false;
	if (!using_fold && ALPHA_RW(adaboost->alpha, adaboost->group_len, file,
				    fread))
		goto read_alpha_err;
	turn_t total = adaboost->group_len * adaboost->dim;
	if ((t = vec_wl_read(adaboost->weaklearner, total, handles, file))
	    < total)
		goto read_wl_err;

	return true;
read_wl_err:
	vec_wl_free(adaboost->weaklearner, t, handles);
read_alpha_err:
	free(adaboost->alpha);
	free(adaboost->weaklearner);
	return false;
}

bool mvec_ada_write(const struct mvec_adaboost *adaboost, FILE * file,
		    const struct wl_handles *handles)
{
	bool using_fold = (adaboost->alpha == NULL);
	if (fwrite(&using_fold, sizeof(bool), 1, file) < 1)
		return false;
	if (fwrite(&adaboost->group_len, sizeof(turn_t), 1, file) < 1)
		return false;
	if (fwrite(&adaboost->dim, sizeof(mlabel_t), 1, file) < 1)
		return false;
	if (!using_fold && ALPHA_RW(adaboost->alpha, adaboost->group_len, file,
				    fwrite))
		return false;
	turn_t total = adaboost->group_len * adaboost->dim;
	if (vec_wl_write(adaboost->weaklearner, total, handles, file) < total)
		return false;

	return true;
}

void *mvec_ada_copy(struct mvec_adaboost *dst,
		    const struct mvec_adaboost *src,
		    const struct wl_handles *handles)
{
	turn_t t;
	bool using_fold = (src->alpha == NULL);
	turn_t total = src->group_len * src->dim;
	if (!mvec_ada_init(dst, src->group_len, src->dim, using_fold, handles))
		return NULL;
	if ((t = vec_wl_copy(dst->weaklearner, src->weaklearner, total,
			     handles)) < total) {
		vec_wl_free(dst->weaklearner, t, handles);
		free(dst->weaklearner);
		free(dst->alpha);
		return NULL;
	}

	ALPHA_COPY(dst->alpha, src->alpha, src->group_len);
	dst->group_len = src->group_len;
	dst->dim = src->dim;
	return dst;
}

void mvec_ada_free(struct mvec_adaboost *adaboost,
		   const struct wl_handles *handles)
{
	vec_wl_free(adaboost->weaklearner, adaboost->dim * adaboost->group_len,
		    handles);
	free(adaboost->weaklearner);
	free(adaboost->alpha);
}
