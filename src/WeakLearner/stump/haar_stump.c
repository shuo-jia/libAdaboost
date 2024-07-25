#include <math.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include "haar_stump.h"
#include "haar_stump_pvt.h"

/**
 * \file haar_stump.c
 * \brief 使用 Haar 特征的决策树桩（函数声明）
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */
/*******************************************************************************
 * 				   宏函数定义
 ******************************************************************************/
/**
 * \brief 训练模板
 * \param[in] stump_type 即 stump 实际上的类型
 * \param[in] fun_opt    基类选择最优划分属性函数的函数名，如 cstump_opt、cstump_cf_opt
 * \details \copydetails haar_stump_train()
 */
#define TRAIN(stump, m, h, w, X, X2, Y, D, stump_type, fun_opt)			\
({										\
 	bool status;								\
	do {									\
		stump_type ptr = stump;						\
		struct sp_wrap sp;						\
		struct stump_opt_handles handles;				\
		if (!init_train (&sp, &handles, X, X2, m, h, w)) {		\
			status = false;						\
 			break;							\
		}								\
		status = fun_opt (&ptr->base, &ptr->feature,			\
				sizeof(struct haar_feature), m, &sp, Y, D,	\
				&handles);					\
		free_train (&sp);						\
	} while (0);								\
	status;									\
})

/*******************************************************************************
 * 				  静态函数原型
 ******************************************************************************/
/**
 * \brief 训练的初始化操作
 * \param[out] sp      指向未初始化的 struct sp_wrap 结构体
 * \param[out] handles 指向未初始化的回调函数集
 * \param[in] X        样本集（积分图）
 * \param[in] X2       样本集（像素平方的积分图）
 * \param[in] m        样本数量
 * \param[in] h        训练图像高度
 * \param[in] w        训练图像宽度
 * \return 成功则返回真，失败则返回假
 */
static bool init_train(struct sp_wrap *sp, struct stump_opt_handles *handles,
		       const sample_t * const *X, const sample_t * const *X2,
		       num_t m, imgsz_t h, imgsz_t w);

/**
 * \brief 训练资源释放操作
 * \param[in] sp 指向已初始化的 struct sp_wrap 结构体
 */
static inline void free_train(struct sp_wrap *sp);

/*******************************************************************************
 *				    函数实现
 ******************************************************************************/
// 获取决策树桩弱学习器的分类结果
label_t haar_stump_h(const void *stump, imgsz_t h, imgsz_t w, imgsz_t wid,
		     const sample_t x[h][wid], const sample_t x2[h][wid],
		     flt_t scale)
{
	const struct haar_stump *cstump = stump;
	return cstump_h(&cstump->base,
			get_value(&cstump->feature, h, w, wid, x, x2, scale));
}

flt_t haar_stump_cf_h(const void *stump, imgsz_t h, imgsz_t w, imgsz_t wid,
		      const sample_t x[h][wid], const sample_t x2[h][wid],
		      flt_t scale)
{
	const struct haar_stump_cf *cstump = stump;
	return cstump_cf_h(&cstump->base,
			   get_value(&cstump->feature, h, w, wid, x, x2,
				     scale));
}

bool haar_stump_train(void *stump, num_t m, imgsz_t h, imgsz_t w,
		      const sample_t * const X[], const sample_t * const X2[],
		      const label_t Y[], const flt_t D[])
{
	return TRAIN(stump, m, h, w, X, X2, Y, D, struct haar_stump *, cstump_opt);
}

bool haar_stump_cf_train(void *stump, num_t m, imgsz_t h, imgsz_t w,
			 const sample_t * const X[],
			 const sample_t * const X2[], const label_t Y[],
			 const flt_t D[])
{
	return TRAIN(stump, m, h, w, X, X2, Y, D, struct haar_stump_cf *,
		     cstump_cf_opt);
}

/*******************************************************************************
 *				  静态函数实现
 ******************************************************************************/
bool init_train(struct sp_wrap *sp, struct stump_opt_handles *handles,
		const sample_t * const *X, const sample_t * const *X2, num_t m,
		imgsz_t h, imgsz_t w)
{
	if ((sp->vector = malloc(sizeof(sample_t) * m)) == NULL)
		return false;
	sp->X = X;
	sp->X2 = X2;
	sp->h = h;
	sp->w = w;

	handles->init_feature = init_feature;
	handles->next_feature = next_feature;
	handles->update_opt = update_opt;
	handles->get_vals.raw = get_vals_raw;
	handles->get_vals.sort = NULL;

	return true;
}

void free_train(struct sp_wrap *sp)
{
	free(sp->vector);
}
