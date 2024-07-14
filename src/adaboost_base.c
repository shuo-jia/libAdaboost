#include <stdlib.h>
#include "adaboost_base.h"
/**
 * \file adaboost_base.c
 * \brief Adaboost 分类器函数实现
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
enum ada_result ada_framework(void *adaboost, num_t m, const void *sample,
			      const void *label,
			      const struct ada_handles *handles)
{
	flt_t *D = NULL;
	flt_t *vals = NULL;
	struct ada_item item = {
		.weaklearner = NULL,
		.alpha = NULL,
		.status = true,
	};

	if ((D = malloc(sizeof(flt_t) * handles->D_len)) == NULL)
		goto MEM_D_ERR;
	if ((vals = malloc(sizeof(flt_t) * handles->vals_len)) == NULL)
		goto MEM_VALS_ERR;

	handles->init_D(D, m, label);
	while (handles->next(&item, adaboost, vals, handles->vals_len) == true) {
		if (!item.status || !handles->train(item.weaklearner, m,
						    sample, label, D))
			goto TRAIN_ERR;
		switch (handles->get_vals(vals, handles->vals_len,
					  item.weaklearner, m, sample, label,
					  D)) {
		case ADA_FAILURE:
			goto TRAIN_ERR;
		case ADA_ALL_PASS:
			*item.alpha = 1;
			free(vals);
			free(D);
			return ADA_ALL_PASS;
		case ADA_SUCCESS:
		default:
			break;
		}
		*item.alpha = handles->get_alpha(vals, handles->vals_len, m,
						 label, D);
		// 更新样本分布
		handles->update_D(D, vals, handles->vals_len, m, label,
				  *item.alpha);
	}

	free(vals);
	free(D);
	return ADA_SUCCESS;

TRAIN_ERR:
	free(vals);
MEM_VALS_ERR:
	free(D);
MEM_D_ERR:
	return ADA_FAILURE;
}
