// adaboost 回调函数集设置
#ifndef ADABOOST_H
#define ADABOOST_H
#include "vec_adaboost.h"
#include "mvec_adaboost.h"
#include "haar_base.h"
#include "WeakLearner/weaklearner.h"

/*******************************************************************************
 * 			    vec_ada_handles 类型定义
 ******************************************************************************/
// 单分类任务，输入样本为向量的 Adaboost 函数类型定义
// 训练方法
typedef bool (*vec_ada_train_fn)(struct vec_adaboost * adaboost, turn_t T,
				 num_t m, dim_t n, const sample_t X[m][n],
				 const label_t Y[], bool cache_on,
				 const struct wl_handles * handles);
// 分类方法，不带置信度
typedef label_t(*vec_ada_h_fn) (const struct vec_adaboost * adaboost,
				const sample_t x[], dim_t n,
				const struct wl_handles * handles);
// 分类方法，带置信度
// 带置信度（弱学习器系数不并入弱学习器）
typedef flt_t(*vec_ada_cf_h_fn) (const struct vec_adaboost * adaboost,
				 const sample_t x[], dim_t n,
				 const struct wl_handles * handles);
// 读取方法，从文件读取分类器
typedef bool (*vec_ada_read_fn)(struct vec_adaboost * adaboost, FILE * file,
				const struct wl_handles * handles);
// 写入方法，向文件写入分类器
typedef bool (*vec_ada_write_fn)(const struct vec_adaboost * adaboost,
				 FILE * file,
				 const struct wl_handles * handles);
// 复制方法，将分类器复制到新的地址
typedef void *(*vec_ada_copy_fn)(struct vec_adaboost * dst,
				 const struct vec_adaboost * src,
				 const struct wl_handles * handles);
// 内存释放方法，用于释放保存训练数据的分类器
typedef void (*vec_ada_free_fn)(struct vec_adaboost * adaboost,
				const struct wl_handles * handles);

// 回调函数集定义
struct vec_ada_handles {
	vec_ada_train_fn train;		// 训练方法
	union {				// 联合，输出分类结果
		vec_ada_h_fn h;		// 不带置信度，输出分类结果
		vec_ada_cf_h_fn cf_h;	// 带置信度，输出分类结果
	} hypothesis;
	vec_ada_read_fn read;		// 读取方法
	vec_ada_write_fn write;		// 写入方法
	vec_ada_copy_fn copy;		// 复制方法
	vec_ada_free_fn free;		// 内存释放方法
	struct wl_handles wl_hl;	// 弱学习器的回调函数集合
};

/*******************************************************************************
 * 			   mvec_ada_handles 类型定义
 ******************************************************************************/
// 多标签分类任务，输入样本为向量的 Adaboost 函数类型定义
// 训练方法
typedef bool (*mvec_ada_train_fn)(struct mvec_adaboost * adaboost, turn_t T,
				  num_t m, dim_t n, const sample_t X[m][n],
				  const mlabel_t Y[], bool cache_on,
				  const struct wl_handles * handles);
// 分类方法
typedef mlabel_t(*mvec_ada_h_fn) (const struct mvec_adaboost * adaboost,
				  const sample_t x[], dim_t n,
				  const struct wl_handles * handles);
// 读取方法，从文件读取分类器
typedef bool (*mvec_ada_read_fn)(struct mvec_adaboost * adaboost, FILE * file,
				 const struct wl_handles * handles);
// 写入方法，向文件写入分类器
typedef bool (*mvec_ada_write_fn)(const struct mvec_adaboost * adaboost,
				  FILE * file,
				  const struct wl_handles * handles);
// 复制方法，将分类器复制到新的地址
typedef void *(*mvec_ada_copy_fn)(struct mvec_adaboost * dst,
				  const struct mvec_adaboost * src,
				  const struct wl_handles * handles);
// 内存释放方法，用于释放保存训练数据的分类器
typedef void (*mvec_ada_free_fn)(struct mvec_adaboost * adaboost,
				 const struct wl_handles * handles);

// 回调函数集定义
struct mvec_ada_handles {
	mvec_ada_train_fn train;	// 训练方法
	mvec_ada_h_fn h;		// 输出分类结果
	mvec_ada_read_fn read;		// 读取方法
	mvec_ada_write_fn write;	// 写入方法
	mvec_ada_copy_fn copy;		// 复制方法
	mvec_ada_free_fn free;		// 内存释放方法
	struct wl_handles wl_hl;	// 弱学习器的回调函数集合
};

/*******************************************************************************
 * 			   haar_ada_handles 类型定义
 ******************************************************************************/
// 单分类任务，输入样本为积分图的 Adaboost 函数类型定义
// 训练方法
typedef bool (*haar_ada_train_fn)(struct haar_adaboost * adaboost, flt_t * d,
				  flt_t * f, num_t l, num_t m, imgsz_t h,
				  imgsz_t w, const sample_t * X[],
				  const sample_t * X2[], const label_t Y[],
				  const struct wl_handles * handles);
// 分类方法
typedef flt_t(*haar_ada_h_fn) (const struct haar_adaboost * adaboost,
			       imgsz_t h, imgsz_t w, imgsz_t wid,
			       const double x[h][wid], const double x2[h][wid],
			       double scale, const struct wl_handles * handles);
// 读取方法，从文件读取分类器
typedef bool (*haar_ada_read_fn)(struct haar_adaboost * adaboost, FILE * file,
				 const struct wl_handles * handles);
// 写入方法，向文件写入分类器
typedef bool (*haar_ada_write_fn)(const struct haar_adaboost * adaboost,
				  FILE * file,
				  const struct wl_handles * handles);
// 复制方法，将分类器复制到新的地址
typedef void *(*haar_ada_copy_fn)(struct haar_adaboost * dst,
				  const struct haar_adaboost * src,
				  const struct wl_handles * handles);
// 内存释放方法，用于释放保存训练数据的分类器
typedef void (*haar_ada_free_fn)(struct haar_adaboost * adaboost,
				 const struct wl_handles * handles);

// 回调函数集定义
struct haar_ada_handles {
	haar_ada_train_fn train;	// 训练方法
	haar_ada_h_fn h;		// 输出分类结果
	haar_ada_read_fn read;		// 读取方法
	haar_ada_write_fn write;	// 写入方法
	haar_ada_copy_fn copy;		// 复制方法
	haar_ada_free_fn free;		// 内存释放方法
	struct wl_handles wl_hl;	// 弱学习器的回调函数集合
};

/*******************************************************************************
 * 				  常量类型定义
 ******************************************************************************/
// 弱学习器类型定义
enum ada_wl_t {
	ADA_CONTINUOUS,		// 连续型弱学习器，使用决策树桩
	ADA_DISCRETE,		// 离散型弱学习器，使用决策树桩
	ADA_WL_END,		// 结束符，该常量值等于常量数量
};

// 弱学习器训练方式定义
enum ada_wl_train_t {
	ADA_OPT,		// 寻找最优划分属性（大量特征耗时长）
	ADA_GA,			// 使用进化算法寻找次优划分属性
};

// 假设器类型定义
enum ada_hypothesis_t {
	ADA_NO_CONFIDENT,	// 不使用置信度
	ADA_CONFIDENT,		// 使用置信度
	ADA_H_END,		// 结束符，该常量值等于常量数量
};

// 弱学习器系数计算算法类型定义
enum ada_alpha_t {
	ADA_APPROX,		// 使用近似算法
	ADA_FOLD,		// 直接并入弱学习器，即系数恒为 1
	ADA_NEWTON,		// 使用数值计算方法（牛顿二分法）
	ADA_ALPHA_END,		// 结束符，该常量值等于常量数量
};

// 多分类任务算法类型定义
enum ada_mvec_t {
	ADA_HLOSS,		// 使用汉明损失
	ADA_MVEC_END,		// 结束符，该常量值等于常量数量
};

// 基于 Haar 特征的 Adaboost 算法类型定义
enum ada_haar_t {
	ADA_NM_APPROX,		// 使用一般的方法，采用近似算法计算 alpha
	ADA_NM_NEWTON,		// 使用一般的方法，采用数值算法计算 alpha
	ADA_ASYM,		// 使用非对称损失
	ADA_ASYM_IMP,		// 使用改进的非对称损失方法
	ADA_HAAR_END,		// 结束符，该常量值等于常量数量
};

/*******************************************************************************
 * 			    vec_ada_handles 函数声明
 ******************************************************************************/
// 设置回调函数集合，所有字段非空
// handles: 要初始化的结构体
// alpha_type: 指示弱学习器系数计算方法
// h_type: 指示假设器类型
// wl_type: 指示弱学习器类型
void ada_set_vec(struct vec_ada_handles *handles, enum ada_alpha_t alpha_type,
		 enum ada_hypothesis_t h_type, enum ada_wl_t wl_type);

/*******************************************************************************
 * 			   mvec_ada_handles 函数声明
 ******************************************************************************/
// 设置回调函数集合，所有字段非空
// handles: 要初始化的结构体
// mvec_type: 指示多分类算法
// alpha_type: 指示弱学习器系数计算方法
// h_type: 指示假设器类型
// wl_type: 指示弱学习器类型
void ada_set_mvec(struct mvec_ada_handles *handles, enum ada_mvec_t mvec_type,
		  enum ada_alpha_t alpha_type, enum ada_wl_t wl_type);

/*******************************************************************************
 * 			   haar_ada_handles 函数声明
 ******************************************************************************/
// 设置回调函数集合，所有字段非空
// handles: 要初始化的结构体
// haar_type: 指示训练方法
void ada_set_haar(struct haar_ada_handles *handles, enum ada_haar_t haar_type,
		  enum ada_wl_train_t wl_train_type);

#endif
