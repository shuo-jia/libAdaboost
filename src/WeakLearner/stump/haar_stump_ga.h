#ifndef HAAR_STUMP_GA_H
#define HAAR_STUMP_GA_H
#include "haar_stump.h"
/**
 * \file haar_stump_ga.h
 * \brief haar_stump 训练函数重载，使用进化算法寻优（函数声明）
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/**
 * \brief 训练 haar_stump 决策树桩弱学习器，不带置信度。
 * 	使用进化算法进行训练，训练速率更快，但不保证所选取的特征为最优特征
 * \details \copydetails wl_train_haar_fn
 */
bool haar_stump_ga_train(void *stump, num_t m, imgsz_t h, imgsz_t w,
			 const sample_t * X[], const sample_t * X2[],
			 const label_t Y[], const flt_t D[]);

/**
 * \brief 训练 haar_stump_cf 决策树桩弱学习器，带置信度。
 * 	使用进化算法进行训练，训练速率更快，但不保证所选取的特征为最优特征
 * \details \copydetails wl_train_haar_fn
 */
bool haar_stump_ga_cf_train(void *stump, num_t m, imgsz_t h, imgsz_t w,
			    const sample_t * X[], const sample_t * X2[],
			    const label_t Y[], const flt_t D[]);

#endif
