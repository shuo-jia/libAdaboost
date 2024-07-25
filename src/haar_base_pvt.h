#ifndef HAAR_BASE_PVT_H
#define HAAR_BASE_PVT_H
#include "haar_base.h"
/**
 * \file haar_base_pvt.h
 * \brief haar_base 的私有部分，包括共用的类型定义及函数声明
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/// 保存数组元素值及其索引，用于排序
struct sort_item {
	num_t id;		///< 元素索引
	flt_t val;		///< 元素值
};

/// Adaboost 包装，附加某些必要变量
struct ada_wrap {
	struct haar_adaboost *adaboost;	///< Adaboost 学习器
	struct sort_item *output;	///< 验证集分类结果
	struct sort_item **op_ptrs;	///< output 的指针数组排序结果
	num_t positive_ct;		///< 验证集正例数量
	num_t l;			///< 验证集样本数量
	flt_t d;			///< Adaboost 最小检测率
	flt_t f;			///< Adaboost 最大假阳率
	const label_t *Y;		///< 验证集样本标签
	size_t wl_size;			///< 弱学习器长度（字节）
};

/// 训练集结构体包装，包含额外的参数
struct sp_wrap {
	num_t l;			///< 验证集样本数量
	imgsz_t h;			///< 训练图片的高度
	imgsz_t w;			///< 训练图片的宽度
	const sample_t * const *X;	///< 积分图指针数组（前 l 个为验证集）
	const sample_t * const *X2;	///< 像素值平方的积分图指针数组
                                        /**<（前 l 个为验证集）*/
	const struct wl_handles *handles;	///< 弱学习器回调函数集
};

/// 训练设置集
struct train_setting {
	struct sp_wrap sp;		///< 样本集（结构体包装）
	struct ada_wrap ada;		///< Adaboost 分类器（结构体包装）
};

/// 回调函数类型：当全部训练样本分类成功时执行的函数
typedef bool (*all_pass_fn)(struct train_setting * st);

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/**
 * \brief 用作回调函数：计算中间值结果并保存到 vals 数组中（不带置信度）
 *      对于验证集，计算弱学习器输出值 h(X[i])；
 *      对于训练集，计算弱学习器输出值与标签的积 h(X[i]) * Y[i]
 * \details \copydetails ada_vals_fn
 */
enum ada_result haar_get_vals(flt_t vals[], num_t vals_len,
			      const void *weaklearner, num_t m,
			      const void *sample, const void *label,
			      const flt_t D[]);

/**
 * \brief 用作回调函数：计算中间值结果并保存到 vals 数组中（带置信度）
 *      对于验证集，计算弱学习器输出值 h(X[i])；
 *      对于训练集，计算弱学习器输出值与标签的积 h(X[i]) * Y[i]
 * \details \copydetails ada_vals_fn
 */
enum ada_result haar_get_vals_cf(flt_t vals[], num_t vals_len,
			      const void *weaklearner, num_t m,
			      const void *sample, const void *label,
			      const flt_t D[]);

/**
 * \brief 初始化 Adaboost 训练所需的回调函数集合
 * \param[out] ada_hl   待初始化结构体
 * \param[in] l         验证集样本数量
 * \param[in] m         训练集样本数量
 * \param[in] get_vals  回调函数，计算中间值，
 *                      可为 haar_get_vals 或 haar_get_vals_cf
 * \param[in] get_alpha 回调函数，计算 alpha 系数
 * \param[in] next      回调函数，获取下一个要进行训练的弱学习器
 * \param[in] init_D    回调函数，初始化概率分布
 * \param[in] update_D  回调函数，更新概率分布，同时将 vals 数组置为
 *                      alpha * h(X[i])
 */
void ada_hl_init(struct ada_handles *ada_hl, num_t l, num_t m,
		 ada_vals_fn get_vals, ada_alpha_fn get_alpha, ada_next_fn next,
		 ada_init_D_fn init_D, ada_update_D_fn update_D);

/**
 * \brief 训练的初始化操作
 * \param[out] st      指向未初始化的训练设置集
 * \param[in] adaboost 指向要进行训练的 Adaboost 结构体
 * \param[in] d        Adaboost 所需的最小检测率
 * \param[in] f        Adaboost 所需的最大假阳率
 * \param[in] l        验证集样本数量
 * \param[in] h        图像高度
 * \param[in] w        图像宽度
 * \param[in] X        积分图（样本集）
 * \param[in] X2       像素值平方的积分图（样本集）
 * \param[in] Y        样本标签集，长度为 l + m
 * \param[in] wl_hl    弱学习器回调函数集合
 * \return 成功则返回真，否则返回假
 */
bool init_setting(struct train_setting *st, struct haar_adaboost *adaboost,
		  flt_t d, flt_t f, num_t l, imgsz_t h, imgsz_t w,
		  const sample_t * const X[], const sample_t * const X2[],
		  const label_t Y[], const struct wl_handles *wl_hl);

/**
 * \brief 释放训练过程中所用到的临时空间
 * \param[in] st: 指向已初始化的训练设置集
 */
void free_setting(struct train_setting *st);

/**
 * \brief 计算当前检测率、假阳率
 * \param[out] d: 用于保存检测率
 * \param[out] f: 用于保存假阳率
 * \param[out] ada: Adaboost 包装结构体，output 字段将被更新
 * \param[in] vals: 中间值，保存 alpha * h(X[i]) 的值
 */
void get_ratio(flt_t * d, flt_t * f, struct ada_wrap *ada, const flt_t vals[]);

/**
 * \brief 当全部训练样本分类成功时执行的函数
 *      更新验证集的假阳率、检测率（不带置信度）
 * \param[in, out] st: 训练设置集，包含样本集及 Adaboost 分类器
 * \return 成功返回真，失败返回假
 */
bool haar_all_pass (struct train_setting *st);

/**
 * \brief 当全部训练样本分类成功时执行的函数
 *      更新验证集的假阳率、检测率（带置信度）
 * \details \copydetails haar_all_pass
 */
bool haar_all_pass_cf (struct train_setting *st);

/*******************************************************************************
 * 				  内联函数定义
 ******************************************************************************/
/**
 * \brief 初始化 Adaboost
 * \param[out] ada 指向未初始化的 struct haar_adaboost 结构体
 */
static inline void haar_ada_init(struct haar_adaboost *ada)
{
	ada->threshold = 0;
	ada->using_fold = false;
	link_list_init(&ada->wl);
}

/**
 * \brief 基于 Adaboost 的 Haar 特征选择器训练框架
 * \param[in] all_pass  当全部样本分类成功时执行的回调函数
 * \param[in] ada_hl    Adaboost 训练所需的回调函数集
 * \details \copydetails haar_ada_train_fn
 * \return 成功则返回真，且保存当前检测率到 *d，保存当前假阳率到 *f；否则返回假。
 * 	注：运行过程未出错，但在验证集上假阳率仍小于 *f的情形，同样返回真。
 */
static inline bool train_framework(struct haar_adaboost *adaboost, flt_t * d,
				   flt_t * f, num_t l, num_t m, imgsz_t h,
				   imgsz_t w, const sample_t * const X[],
				   const sample_t * const X2[],
				   const label_t Y[], all_pass_fn all_pass,
				   const struct wl_handles *handles,
				   const struct ada_handles *ada_hl)
{
	struct train_setting st;
	haar_ada_init(adaboost);
	if (!init_setting(&st, adaboost, *d, *f, l, h, w, X, X2, Y, handles))
		goto init_st_err;
	switch (ada_framework(&st.ada, m, &st.sp, Y + l, ada_hl)) {
	case ADA_ALL_PASS:
		if (all_pass(&st) == false)
			goto train_err;
	case ADA_SUCCESS:
		*d = st.ada.d;
		*f = st.ada.f;
		break;
	case ADA_FAILURE:
deafult:
		goto train_err;
	}

	free_setting(&st);
	return true;

train_err:
	free_setting(&st);
init_st_err:
	haar_ada_free(adaboost, handles);
	return false;
}

#endif
