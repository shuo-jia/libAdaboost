// 使用汉明损失（Hamming loss）进行训练
#ifndef MVEC_HLOSS_H
#define MVEC_HLOSS_H
#include "mvec_adaboost.h"

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/*
 * Adaboost 训练方法
 * adaboost: 指向未初始化的 struct mvec_adaboost 结构体
 * T: 训练的总轮数
 * m: 样本数量
 * n: 样本特征数量，即样本向量的长度
 * X: 样本集
 * Y: 样本标签集，长度为 m；0 表示类别 0，1 表示类别 1，诸如此类
 * cache_on: 表示是否启用缓存，真值表示启用
 * handles: 弱学习器回调函数集合
 * 返回值：成功则返回真，否则返回假
 */
// 近似方法，利用不等式缩放获取系数 alpha
bool mvec_ada_approx_train(struct mvec_adaboost *adaboost, turn_t T, num_t m,
			   dim_t n, const sample_t X[m][n], const mlabel_t Y[],
			   bool cache_on, const struct wl_handles *handles);

// 将 alpha 合并到弱学习器 h 中，即 alpha 恒为 1
bool mvec_ada_fold_train(struct mvec_adaboost *adaboost, turn_t T, num_t m,
			 dim_t n, const sample_t X[m][n], const mlabel_t Y[],
			 bool cache_on, const struct wl_handles *handles);

// 数值方法，应用牛顿二分法求系数 alpha
bool mvec_ada_newton_train(struct mvec_adaboost *adaboost, turn_t T, num_t m,
			   dim_t n, const sample_t X[m][n], const mlabel_t Y[],
			   bool cache_on, const struct wl_handles *handles);

/*
 * 获取 Adaboost 分类结果
 * adaboost: 指向已保存训练结果的 struct mvec_adaboost 结构体
 * x: 样本向量
 * n: 样本向量的长度
 * handles: 弱学习器回调函数集合
 * 返回值：返回样本  x 在 Adaboost 上的输出结果
 */
// 输出分类结果，弱学习器系数不并入弱学习器
mlabel_t mvec_ada_h(const struct mvec_adaboost *adaboost, const sample_t x[],
		    dim_t n, const struct wl_handles *handles);

// 输出分类结果，弱学习器系数并入弱学习器
mlabel_t mvec_ada_fold_h(const struct mvec_adaboost *adaboost,
			 const sample_t x[], dim_t n,
			 const struct wl_handles *handles);

#endif
