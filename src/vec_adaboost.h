#ifndef VEC_ADABOOST_H
#define VEC_ADABOOST_H
#include "boost_cfg.h"
#include "WeakLearner/weaklearner.h"

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
/*
 * Adaboost 训练方法
 * adaboost: 指向未初始化的 struct vec_adaboost 结构体
 * T: 训练的总轮数
 * m: 样本数量
 * n: 样本特征数量，即样本向量的长度
 * X: 样本集
 * Y: 样本标签集，长度为 m
 * cache_on: 表示是否启用缓存，真值表示启用
 * handles: 弱学习器回调函数集合
 * 返回值：成功则返回真，否则返回假
 */
// 近似方法，利用不等式缩放获取系数 alpha
bool vec_ada_approx_train(struct vec_adaboost *adaboost, turn_t T, num_t m,
			  dim_t n, const sample_t X[m][n], const label_t Y[],
			  bool cache_on, const struct wl_handles *handles);

// 将 alpha 合并到弱学习器 h 中，即 alpha 恒为 1
bool vec_ada_fold_train(struct vec_adaboost *adaboost, turn_t T, num_t m,
			dim_t n, const sample_t X[m][n], const label_t Y[],
			bool cache_on, const struct wl_handles *handles);

// 数值方法，应用牛顿二分法求系数 alpha
bool vec_ada_newton_train(struct vec_adaboost *adaboost, turn_t T, num_t m,
			  dim_t n, const sample_t X[m][n], const label_t Y[],
			  bool cache_on, const struct wl_handles *handles);

/*
 * 获取 Adaboost 分类结果
 * adaboost: 指向已保存训练结果的 struct vec_adaboost 结构体
 * x: 样本向量
 * n: 样本向量的长度
 * handles: 弱学习器回调函数集合
 * 返回值：返回样本  x 在 Adaboost 上的输出结果，大于 0 表示判断为标签 +1，小于
 * 	0 表示判定为标签 -1
 */
// 不带置信度（弱学习器系数不并入弱学习器）
label_t vec_ada_h(const struct vec_adaboost *adaboost, const sample_t x[],
		  dim_t n, const struct wl_handles *handles);
// 带置信度（弱学习器系数不并入弱学习器）
flt_t vec_ada_cf_h(const struct vec_adaboost *adaboost, const sample_t x[],
		   dim_t n, const struct wl_handles *handles);
// 不带置信度（弱学习器系数并入弱学习器）
label_t vec_ada_fold_h(const struct vec_adaboost *adaboost, const sample_t x[],
		       dim_t n, const struct wl_handles *handles);
// 带置信度（弱学习器系数并入弱学习器）
flt_t vec_ada_fold_cf_h(const struct vec_adaboost *adaboost, const sample_t x[],
			dim_t n, const struct wl_handles *handles);

/*
 * 从文件中读取 Adaboost
 * adaboost: 指向未初始化的 struct vec_adaboost 结构体
 * file: 已打开的文件
 * handles: 弱学习器回调函数集合
 * 返回值：成功则返回真；失败则返回假
 */
bool vec_ada_read(struct vec_adaboost *adaboost, FILE * file,
		  const struct wl_handles *handles);

/*
 * 将 Adaboost 结构体写入到文件
 * adaboost: 指向已保存训练结果的 struct vec_adaboost 结构体
 * file: 已打开的文件
 * handles: 弱学习器回调函数集合
 * 返回值：成功则返回真；失败则返回假
 */
bool vec_ada_write(const struct vec_adaboost *adaboost, FILE * file,
		   const struct wl_handles *handles);

/*
 * 对 Adaboost 进行深度复制
 * dst: 指向未初始化的 struct vec_adaboost 结构体
 * src: 指向已初始化的 struct vec_adaboost 结构体，内容将被复制到 dst
 * handles: 弱学习器回调函数集合
 * 返回值：成功则返回 dst；失败则返回 NULL
 */
void *vec_ada_copy(struct vec_adaboost *dst, const struct vec_adaboost *src,
		   const struct wl_handles *handles);

/*
 * 内存释放方法
 * adaboost: 指向已保存训练结果的 struct vec_adaboost 结构体
 * handles: 弱学习器回调函数集合
 */
void vec_ada_free(struct vec_adaboost *adaboost,
		  const struct wl_handles *handles);

#endif
