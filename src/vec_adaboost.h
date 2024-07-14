#ifndef VEC_ADABOOST_H
#define VEC_ADABOOST_H
#include "boost_cfg.h"
#include "WeakLearner/weaklearner.h"
/**
 * \file vec_adaboost.h
 * \brief Adaboost 分类器定义与函数声明：二分类任务、样本集为向量集
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/// 输入样本为向量集的 Adaboost 学习器
struct vec_adaboost {
	turn_t size;			///< 弱学习器数量
	unsigned char *weaklearner;	///< 弱学习器数组地址
	flt_t *alpha;			///< 弱学习器系数数组的地址
};

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/**
 * \brief vec_adaboost 训练方法，使用近似方法，利用不等式缩放获取系数 alpha
 * \details \copydetails vec_ada_train_fn
 */
bool vec_ada_approx_train(struct vec_adaboost *adaboost, turn_t T, num_t m,
			  dim_t n, const sample_t X[m][n], const label_t Y[],
			  bool cache_on, const struct wl_handles *handles);

/**
 * \brief vec_adaboost 训练方法，将 alpha 合并到弱学习器 h 中，即 alpha 恒为 1
 * \details \copydetails vec_ada_train_fn
 */
bool vec_ada_fold_train(struct vec_adaboost *adaboost, turn_t T, num_t m,
			dim_t n, const sample_t X[m][n], const label_t Y[],
			bool cache_on, const struct wl_handles *handles);

/**
 * \brief vec_adaboost 训练方法，使用数值方法，应用牛顿二分法求系数 alpha
 * \details \copydetails vec_ada_train_fn
 */
bool vec_ada_newton_train(struct vec_adaboost *adaboost, turn_t T, num_t m,
			  dim_t n, const sample_t X[m][n], const label_t Y[],
			  bool cache_on, const struct wl_handles *handles);

/**
 * \brief vec_adaboost 分类方法，不带置信度（弱学习器系数不并入弱学习器）
 * \details \copydetails vec_ada_h_fn
 */
label_t vec_ada_h(const struct vec_adaboost *adaboost, const sample_t x[],
		  dim_t n, const struct wl_handles *handles);

/**
 * \brief vec_adaboost 分类方法，带置信度
 * \details \copydetails vec_ada_cf_h_fn
 */
flt_t vec_ada_cf_h(const struct vec_adaboost *adaboost, const sample_t x[],
		   dim_t n, const struct wl_handles *handles);

/**
 * \brief vec_adaboost 分类方法，不带置信度（弱学习器系数并入弱学习器）
 * \details \copydetails vec_ada_h_fn
 */
label_t vec_ada_fold_h(const struct vec_adaboost *adaboost, const sample_t x[],
		       dim_t n, const struct wl_handles *handles);

/**
 * \brief vec_adaboost 分类方法，带置信度（弱学习器系数并入弱学习器）
 * \details \copydetails vec_ada_h_fn
 */
flt_t vec_ada_fold_cf_h(const struct vec_adaboost *adaboost, const sample_t x[],
			dim_t n, const struct wl_handles *handles);

/**
 * \brief 从文件中读取 vec_adaboost
 * \details \copydetails vec_ada_read_fn
 */
bool vec_ada_read(struct vec_adaboost *adaboost, FILE * file,
		  const struct wl_handles *handles);

/**
 * \brief 将 vec_adaboost 写入文件
 * \details \copydetails vec_ada_write_fn
 */
bool vec_ada_write(const struct vec_adaboost *adaboost, FILE * file,
		   const struct wl_handles *handles);

/**
 * \brief 对 vec_adaboost 进行深度复制
 * \details \copydetails vec_ada_copy_fn
 */
void *vec_ada_copy(struct vec_adaboost *dst, const struct vec_adaboost *src,
		   const struct wl_handles *handles);

/**
 * \brief vec_adaboost 结构体内部空间
 * \details \copydetails vec_ada_free_fn
 */
void vec_ada_free(struct vec_adaboost *adaboost,
		  const struct wl_handles *handles);

#endif
