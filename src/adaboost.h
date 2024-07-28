// adaboost 回调函数集设置
#ifndef ADABOOST_H
#define ADABOOST_H
#include "vec_adaboost.h"
#include "mvec_adaboost.h"
#include "haar_base.h"
#include "WeakLearner/weaklearner.h"
/**
 * \file adaboost.h
 * \brief Adaboost 分类器函数调用集及函数声明。
 * 	此文件定义了与 Adaboost 相关的一组回调函数集合，使用类似工厂模式的方法设
 * 	置不同种类的回调函数集。此文件为不同 Adaboost 分类器提供了一组通用的接口。
 * \author Shuojia
 * \version dd
 * \date 2024-07-14
 */

/*******************************************************************************
 * 			    vec_ada_handles 类型定义
 ******************************************************************************/
/**
 * \brief 回调函数类型：Adaboost 训练方法，支持单分类任务，输入样本为向量
 * \param[out] adaboost 指向未初始化的 struct vec_adaboost 结构体
 * \param[in] T         训练的总轮数
 * \param[in] m         样本数量
 * \param[in] n         样本特征数量，即样本向量的长度
 * \param[in] X         样本集
 * \param[in] Y         样本标签集，长度为 m
 * \param[in] cache_on  表示是否启用缓存，真值表示启用
 * \param[in] handles   弱学习器回调函数集合
 * \return 成功则返回真，否则返回假
 */
typedef bool (*vec_ada_train_fn)(struct vec_adaboost * adaboost, turn_t T,
				 num_t m, dim_t n, const sample_t X[m][n],
				 const label_t Y[], bool cache_on,
				 const struct wl_handles * handles);

/**
 * \brief 回调函数类型：Adaboost 分类方法，不带置信度
 * \param[in] adaboost 指向已保存训练结果的 struct vec_adaboost 结构体
 * \param[in] x        样本向量
 * \param[in] n        样本向量的长度
 * \param[in] handles  弱学习器回调函数集合
 * \return 返回样本  x 在 Adaboost 上的输出结果（+1 或 -1）
 */
typedef label_t(*vec_ada_h_fn) (const struct vec_adaboost * adaboost,
				const sample_t x[], dim_t n,
				const struct wl_handles * handles);

/**
 * \brief 回调函数类型：Adaboost 分类方法，带置信度
 * \param[in] adaboost 指向已保存训练结果的 struct vec_adaboost 结构体
 * \param[in] x        样本向量
 * \param[in] n        样本向量的长度
 * \param[in] handles  弱学习器回调函数集合
 * \return 返回样本  x 在 Adaboost 上的输出结果（置信度，大于 0 表示判定为正例，
 *      否则判定为负例）
 */
typedef flt_t(*vec_ada_cf_h_fn) (const struct vec_adaboost * adaboost,
				 const sample_t x[], dim_t n,
				 const struct wl_handles * handles);

/**
 * \brief 回调函数类型：从文件中读取 Adaboost
 * \param[out] adaboost 指向未初始化的 struct vec_adaboost 结构体
 * \param[in] file      已打开的文件
 * \param[in] handles   弱学习器回调函数集合
 * \return 成功则返回真；失败则返回假
 */
typedef bool (*vec_ada_read_fn)(struct vec_adaboost * adaboost, FILE * file,
				const struct wl_handles * handles);

/**
 * \brief 回调函数类型：将 Adaboost 结构体写入到文件
 * \param[in] adaboost  指向已保存训练结果的 struct vec_adaboost 结构体
 * \param[out] file     已打开的文件
 * \param[in] handles   弱学习器回调函数集合
 * \return 成功则返回真；失败则返回假
 */
typedef bool (*vec_ada_write_fn)(const struct vec_adaboost * adaboost,
				 FILE * file,
				 const struct wl_handles * handles);

/**
 * \brief 回调函数类型：对 Adaboost 进行深度复制
 * \param[out] dst    指向未初始化的 struct vec_adaboost 结构体
 * \param[in] src     指向已初始化的 struct vec_adaboost 结构体，内容将被复制到 dst
 * \param[in] handles 弱学习器回调函数集合
 * \return 成功则返回 dst；失败则返回 NULL
 */
typedef void *(*vec_ada_copy_fn)(struct vec_adaboost * dst,
				 const struct vec_adaboost * src,
				 const struct wl_handles * handles);

/**
 * \brief 回调函数类型：内存释放方法
 * \param[in] adaboost 指向已保存训练结果的 struct vec_adaboost 结构体
 * \param[in] handles  弱学习器回调函数集合
 */
typedef void (*vec_ada_free_fn)(struct vec_adaboost * adaboost,
				const struct wl_handles * handles);

/// 回调函数集定义
struct vec_ada_handles {
	vec_ada_train_fn train;		///< 训练方法
	union {				///< 联合，输出分类结果
		vec_ada_h_fn h;		///< 不带置信度，输出分类结果
		vec_ada_cf_h_fn cf_h;	///< 带置信度，输出分类结果
	};
	vec_ada_read_fn read;		///< 读取方法
	vec_ada_write_fn write;		///< 写入方法
	vec_ada_copy_fn copy;		///< 复制方法
	vec_ada_free_fn free;		///< 内存释放方法
	struct wl_handles wl_hl;	///< 弱学习器的回调函数集合
};

/*******************************************************************************
 * 			   mvec_ada_handles 类型定义
 ******************************************************************************/
/**
 * \brief 回调函数类型：Adaboost 训练方法，支持多分类任务，输入为样本向量
 * \param[out] adaboost 指向未初始化的 struct mvec_adaboost 结构体
 * \param[in] T         训练的总轮数
 * \param[in] m         样本数量
 * \param[in] n         样本特征数量，即样本向量的长度
 * \param[in] X         样本集
 * \param[in] Y         样本标签集，长度为 m；0 表示类别 0，1 表示类别 1，诸如此类
 * \param[in] cache_on  表示是否启用缓存，真值表示启用
 * \param[in] handles   弱学习器回调函数集合
 * \return 成功则返回真，否则返回假
 */
typedef bool (*mvec_ada_train_fn)(struct mvec_adaboost * adaboost, turn_t T,
				  num_t m, dim_t n, const sample_t X[m][n],
				  const mlabel_t Y[], bool cache_on,
				  const struct wl_handles * handles);

/**
 * \brief 获取 Adaboost 分类结果
 * \param[in] adaboost 指向已保存训练结果的 struct mvec_adaboost 结构体
 * \param[in] x        样本向量
 * \param[in] n        样本向量的长度
 * \param[in] handles  弱学习器回调函数集合
 * \return 返回样本  x 在 Adaboost 上的输出结果
 */
typedef mlabel_t(*mvec_ada_h_fn) (const struct mvec_adaboost * adaboost,
				  const sample_t x[], dim_t n,
				  const struct wl_handles * handles);

/**
 * \brief 回调函数类型：从文件中读取 Adaboost
 * \param[out] adaboost 指向未初始化的 struct mvec_adaboost 结构体
 * \param[in] file      已打开的文件
 * \param[in] handles   弱学习器回调函数集合
 * \return 成功则返回真；失败则返回假
 */
typedef bool (*mvec_ada_read_fn)(struct mvec_adaboost * adaboost, FILE * file,
				 const struct wl_handles * handles);

/**
 * \brief 将 Adaboost 结构体写入到文件
 * \param[in] adaboost 指向已保存训练结果的 struct mvec_adaboost 结构体
 * \param[out] file    已打开的文件
 * \param[in] handles  弱学习器回调函数集合
 * \return 成功则返回真；失败则返回假
 */
typedef bool (*mvec_ada_write_fn)(const struct mvec_adaboost * adaboost,
				  FILE * file,
				  const struct wl_handles * handles);

/**
 * \brief 对 Adaboost 进行深度复制
 * \param[out] dst    指向未初始化的 struct mvec_adaboost 结构体
 * \param[in] src     指向已初始化的 struct mvec_adaboost 结构体，内容将被复制到 dst
 * \param[in] handles 弱学习器回调函数集合
 * \return 成功则返回 dst；失败则返回 NULL
 */
typedef void *(*mvec_ada_copy_fn)(struct mvec_adaboost * dst,
				  const struct mvec_adaboost * src,
				  const struct wl_handles * handles);

/**
 * \brief Adaboost 内存释放方法
 * \param[in] adaboost: 指向已保存训练结果的 struct mvec_adaboost 结构体
 * \param[in] handles: 弱学习器回调函数集合
 */
typedef void (*mvec_ada_free_fn)(struct mvec_adaboost * adaboost,
				 const struct wl_handles * handles);

/// 回调函数集定义
struct mvec_ada_handles {
	mvec_ada_train_fn train;	///< 训练方法
	mvec_ada_h_fn h;		///< 输出分类结果
	mvec_ada_read_fn read;		///< 读取方法
	mvec_ada_write_fn write;	///< 写入方法
	mvec_ada_copy_fn copy;		///< 复制方法
	mvec_ada_free_fn free;		///< 内存释放方法
	struct wl_handles wl_hl;	///< 弱学习器的回调函数集合
};

/*******************************************************************************
 * 			   haar_ada_handles 类型定义
 ******************************************************************************/
/**
 * \brief Adaboost 训练方法，支持二分类任务，输入样本为积分图
 * \param[out] adaboost 指向未初始化的 struct haar_adaboost 结构体
 * \param[in, out] d    指向 AdaBoost 分类器的最小检测率
 * \param[in, out] f    指向 AdaBoost 分类器的最大假阳率
 * \param[in] l         验证集样本数量
 * \param[in] m         训练集样本数量
 * \param[in] h         图像高度
 * \param[in] w         图像宽度
 * \param[in] X         图像灰度值的积分图指针数组，每个指针指向 h*w 的图像区域
 * \param[in] X2        图像灰度值平方的积分图指针数组，每个指针指向 h*w 的图像区域
 * \param[in] Y         样本标签集，长度为 l + m
 * \param[in] handles   弱学习器回调函数集合
 * \return 成功则返回真，并设置 *d 为当前检测率，*f 为当前假阳率；否则返回假
 * 	注：运行过程未出错，但在验证集上假阳率仍小于 *f的情形，同样返回真。
 */
typedef bool (*haar_ada_train_fn)(struct haar_adaboost * adaboost, flt_t * d,
				  flt_t * f, num_t l, num_t m, imgsz_t h,
				  imgsz_t w, const sample_t * const X[],
				  const sample_t * const X2[], const label_t Y[],
				  const struct wl_handles * handles);

/**
 * \brief 获取分类结果
 * \param[in] adaboost 已训练完毕的 AdaBoost 学习器；
 * \param[in] h        窗口高度
 * \param[in] w        窗口宽度
 * \param[in] wid      图像实际宽度
 * \param[in] x        积分图
 * \param[in] x2       灰度值平方的积分图
 * \param[in] scale    与训练图片相比的尺度放大倍数
 * \param[in] handles  弱学习器回调函数集合
 * \return 输出分类结果（置信度）
 */
typedef flt_t(*haar_ada_h_fn) (const struct haar_adaboost * adaboost,
			       imgsz_t h, imgsz_t w, imgsz_t wid,
			       const double x[h][wid], const double x2[h][wid],
			       double scale, const struct wl_handles * handles);

/**
 * \brief 从文件中读取 Adaboost
 * \param[out] adaboost 指向未初始化的 struct haar_adaboost 结构体
 * \param[in] file      已打开的文件
 * \param[in] handles   弱学习器回调函数集合
 * \return 成功则返回真；失败则返回假
 */
typedef bool (*haar_ada_read_fn)(struct haar_adaboost * adaboost, FILE * file,
				 const struct wl_handles * handles);

/**
 * \brief 将 Adaboost 结构体写入到文件
 * \param[in] adaboost 指向已保存训练结果的 struct haar_adaboost 结构体
 * \param[out] file    已打开的文件
 * \param[in] handles  弱学习器回调函数集合
 * \return 成功则返回真；失败则返回假
 */
typedef bool (*haar_ada_write_fn)(const struct haar_adaboost *adaboost,
				  FILE *file, const struct wl_handles *handles);

/**
 * \brief 对 Adaboost 进行深度复制
 * \param[out] dst    指向未初始化的 struct haar_adaboost 结构体
 * \param[in] src     指向已初始化的 struct haar_adaboost 结构体，内容将被复制到 dst
 * \param[in] handles 弱学习器回调函数集合
 * \return 成功则返回 dst；失败则返回 NULL
 */
typedef void *(*haar_ada_copy_fn)(struct haar_adaboost * dst,
				  const struct haar_adaboost * src,
				  const struct wl_handles * handles);

/**
 * \brief 内存释放方法
 * \param[in] adaboost: 指向已保存训练结果的 struct haar_adaboost 结构体
 * \param[in] handles: 弱学习器回调函数集合
 */
typedef void (*haar_ada_free_fn)(struct haar_adaboost * adaboost,
				 const struct wl_handles * handles);

/// 回调函数集定义
struct haar_ada_handles {
	haar_ada_train_fn train;	///< 训练方法
	haar_ada_h_fn h;		///< 输出分类结果
	haar_ada_read_fn read;		///< 读取方法
	haar_ada_write_fn write;	///< 写入方法
	haar_ada_copy_fn copy;		///< 复制方法
	haar_ada_free_fn free;		///< 内存释放方法
	struct wl_handles wl_hl;	///< 弱学习器的回调函数集合
};

/*******************************************************************************
 * 				  常量类型定义
 ******************************************************************************/
/// 弱学习器类型定义
enum ada_wl_t {
	ADA_CONTINUOUS,		///< 连续型弱学习器，使用决策树桩
	ADA_DISCRETE,		///< 离散型弱学习器，使用决策树桩
	ADA_WL_END,		///< 结束符，该常量值等于常量数量
};

/// 弱学习器训练方式定义
enum ada_wl_train_t {
	ADA_OPT,		///< 寻找最优划分属性（大量特征耗时长）
	ADA_GA,			///< 使用进化算法寻找次优划分属性
};

/// 假设器类型定义
enum ada_hypothesis_t {
	ADA_NO_CONFIDENT,	///< 不使用置信度
	ADA_CONFIDENT,		///< 使用置信度
	ADA_H_END,		///< 结束符，该常量值等于常量数量
};

/// 弱学习器系数计算算法类型定义
enum ada_alpha_t {
	ADA_APPROX,		///< 使用近似算法
	ADA_FOLD,		///< 直接并入弱学习器，即系数恒为 1
	ADA_NEWTON,		///< 使用数值计算方法（牛顿二分法）
	ADA_ALPHA_END,		///< 结束符，该常量值等于常量数量
};

/// 多分类任务算法类型定义
enum ada_mvec_t {
	ADA_HLOSS,		///< 使用汉明损失
	ADA_MVEC_END,		///< 结束符，该常量值等于常量数量
};

/// 基于 Haar 特征的 Adaboost 算法类型定义
enum ada_haar_t {
	ADA_NM_APPROX,		///< 使用一般的方法，采用近似算法计算 alpha
	ADA_NM_NEWTON,		///< 使用一般的方法，采用数值算法计算 alpha
	ADA_ASYM,		///< 使用非对称损失
	ADA_ASYM_IMP,		///< 使用改进的非对称损失方法
	ADA_HAAR_END,		///< 结束符，该常量值等于常量数量
};

/*******************************************************************************
 * 			    vec_ada_handles 函数声明
 ******************************************************************************/
/**
 * \brief 设置回调函数集合，所有字段非空
 * \param[out] handles: 要初始化的结构体
 * \param[in] alpha_type: 指示弱学习器系数计算方法
 * \param[in] h_type: 指示假设器类型
 * 	（如为 ADA_CONFIDENT，则使用 handles->cf_h() 获取分类结果；
 * 	  如为 ADA_NO_CONFIDENT，则使用 handles->h() 获取分类结果）
 * \param[in] wl_type: 指示弱学习器类型
 */
void ada_set_vec(struct vec_ada_handles *handles, enum ada_alpha_t alpha_type,
		 enum ada_hypothesis_t h_type, enum ada_wl_t wl_type);

/*******************************************************************************
 * 			   mvec_ada_handles 函数声明
 ******************************************************************************/
/**
 * \brief 设置回调函数集合，所有字段非空
 * \param[out] handles: 要初始化的结构体
 * \param[in] mvec_type: 指示多分类算法
 * \param[in] alpha_type: 指示弱学习器系数计算方法
 * \param[in] wl_type: 指示弱学习器类型
 */
void ada_set_mvec(struct mvec_ada_handles *handles, enum ada_mvec_t mvec_type,
		  enum ada_alpha_t alpha_type, enum ada_wl_t wl_type);

/*******************************************************************************
 * 			   haar_ada_handles 函数声明
 ******************************************************************************/
/**
 * \brief 设置回调函数集合，所有字段非空
 * \param[out] handles: 要初始化的结构体
 * \param[in] haar_type: 指示训练方法
 * \param[in] wl_train_type: 指示弱学习器训练方法
 */
void ada_set_haar(struct haar_ada_handles *handles, enum ada_haar_t haar_type,
		  enum ada_wl_train_t wl_train_type);

/**
 * 一个训练 AdaBoost 分类器的例子
 * \note
 * 数据集采用手写字符数据集 DIGITS
 * (https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits)\n
 * 解压后放入 dataset 目录下。
 *
 * \par txt 表格的读取模块（类型及函数声明）
 * table.h
 * \include table.h
 * \par txt 表格的读取模块（函数实现）
 * \include table.c
 * \par 配置文件修改
 * boost_cfg.h
 * \include adaboost/boost_cfg.h
 *
 * \par 一个简单的二分类任务示例（对数字 0，数字 1 进行分类）
 * ada_train.c
 * \include ada_train.c
 *
 * \par 多分类任务示例
 * mada_train.c
 * \example mada_train.c
 */
#endif
