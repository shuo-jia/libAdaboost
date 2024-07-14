#ifndef HAAR_BASE_PVT_H
#define HAAR_BASE_PVT_H
#include "haar_base.h"

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
// 保存数组元素值及其索引，用于排序
struct sort_item {
	num_t id;		// 元素索引
	flt_t val;		// 元素值
};

// Adaboost 包装，附加某些必要变量
struct ada_wrap {
	struct haar_adaboost *adaboost;	// Adaboost 学习器
	struct sort_item *output;	// 验证集分类结果
	struct sort_item **op_ptrs;	// output 的指针数组排序结果
	num_t positive_ct;		// 验证集正例数量
	num_t l;			// 验证集样本数量
	flt_t d;			// Adaboost 最小检测率
	flt_t f;			// Adaboost 最大假阳率
	const label_t *Y;		// 验证集样本标签
	size_t wl_size;			// 弱学习器长度（字节）
};

// 训练集结构体包装，包含额外的参数
struct sp_wrap {
	num_t l;			// 验证集样本数量
	imgsz_t h;			// 训练图片的高度
	imgsz_t w;			// 训练图片的宽度
	const sample_t **X;		// 积分图指针数组
	const sample_t **X2;		// 像素值平方的积分图指针数组
	const struct wl_handles *handles;	// 弱学习器回调函数集
};

// 训练设置集
struct train_setting {
	struct sp_wrap sp;
	struct ada_wrap ada;
};

// 回调函数类型：当全部训练样本分类成功时执行的函数（原型）
typedef bool (*all_pass_fn)(struct train_setting * st);

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/*
 * 初始化 Adaboost 训练所需的回调函数集合
 * ada_hl: 待初始化结构体
 * l: 验证集样本数量
 * m: 训练集样本数量
 * get_alpha: 回调函数，计算 alpha 系数
 * next: 回调函数，获取下一个要进行训练的弱学习器
 * init_D: 回调函数，初始化概率分布
 * update_D: 回调函数，更新概率分布，同时将 vals 数组置为 alpha * h(X[i])
 */
void ada_hl_init(struct ada_handles *ada_hl, num_t l, num_t m,
		 ada_vals_fn get_vals, ada_alpha_fn get_alpha, ada_next_fn next,
		 ada_init_D_fn init_D, ada_update_D_fn update_D);

/*
 * 训练的初始化操作
 * st: 指向未初始化的训练设置集
 * adaboost: 指向要进行训练的 Adaboost 结构体
 * d: Adaboost 所需的最小检测率
 * f: Adaboost 所需的最大假阳率
 * l: 验证集样本数量
 * h: 图像高度
 * w: 图像宽度
 * X: 积分图（样本集）
 * X2: 像素值平方的积分图（样本集）
 * Y: 样本标签集，长度为 l + m
 * wl_hl: 弱学习器回调函数集合
 * 返回值：成功则返回真，否则返回假
 */
bool init_setting(struct train_setting *st, struct haar_adaboost *adaboost,
		  flt_t d, flt_t f, num_t l, imgsz_t h, imgsz_t w,
		  const sample_t * X[], const sample_t * X2[],
		  const label_t Y[], const struct wl_handles *wl_hl);

/*
 * 释放训练过程中所用到的临时空间
 * st: 指向已初始化的训练设置集
 */
void free_setting(struct train_setting *st);

/*
 * 计算当前检测率、假阳率
 * d: 用于保存检测率
 * f: 用于保存假阳率
 * ada: Adaboost 包装结构体，output 字段将被更新
 * vals: 中间值，保存 alpha * h(X[i]) 的值
 */
void get_ratio(flt_t * d, flt_t * f, struct ada_wrap *ada, const flt_t vals[]);

/*******************************************************************************
 * 				   宏函数定义
 ******************************************************************************/
// 当全部训练样本分类成功时执行的函数（函数模板）
// 更新验证集的假阳率、检测率
// st: struct train_setting * 类型变量指针
// op_val: （宏）函数名，接收 st、样本索引 i 作为参数，返回 h(X[i]) * alpha 的值
#define ALL_PASS(st, op_val)							\
({										\
 	bool done = false;							\
	do {									\
		flt_t * vals = malloc (sizeof(flt_t) * st->ada.l);		\
		if (vals == NULL)						\
			break;							\
		for (num_t i = 0; i < st->ada.l; ++i)				\
			vals[i] = op_val(st, i);				\
										\
		get_ratio (&(st->ada.d), &(st->ada.f), &st->ada, vals);		\
		free (vals);							\
		done = true;							\
	} while (0);								\
	done;									\
})

/*******************************************************************************
 * 				  静态函数定义
 ******************************************************************************/
// 初始化 Adaboost
static inline void haar_ada_init(struct haar_adaboost *ada)
{
	ada->threshold = 0;
	ada->using_fold = false;
	link_list_init(&ada->wl);
}

/*
 * 基于 Adaboost 的 Haar 特征选择器训练框架
 * adaboost: 指向 struct haar_adaboost 结构体
 * d: 指向 AdaBoost 分类器的最小检测率
 * f: 指向 AdaBoost 分类器的最大假阳率
 * l: 验证集样本数量
 * m: 训练集样本数量
 * h: 图像高度
 * w: 图像宽度
 * X: 图像灰度值的积分图指针数组，每个指针指向 h*w 的图像区域
 * X2: 图像灰度值平方的积分图指针数组，每个指针指向 h*w 的图像区域
 * Y: 样本标签集，长度为 l + m
 * all_pass: 当全部训练样本分类成功时执行的函数，在本文件中以宏函数形式给出了一
 * 	个模板
 * handles: 弱学习器回调函数集合
 * ada_hl: Adaboost 训练所需的回调函数集
 * 返回值：成功则返回真，且保存当前检测率到 *d，保存当前假阳率到 *f；否则返回假。
 * 	注：运行过程未出错，但在验证集上假阳率仍小于 *f的情形，同样返回真。
 */
static inline bool train_framework(struct haar_adaboost *adaboost, flt_t * d,
				   flt_t * f, num_t l, num_t m, imgsz_t h,
				   imgsz_t w, const sample_t * X[],
				   const sample_t * X2[], const label_t Y[],
				   all_pass_fn all_pass,
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
