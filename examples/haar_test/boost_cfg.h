// 类型定义配置
#ifndef BOOST_CFG_H
#define BOOST_CFG_H

/*******************************************************************************
 * 				弱学习器类型配置
 ******************************************************************************/
// 常数弱学习器类型
typedef double constant;

// 决策树桩（cstump系列）属性划分值的最小间隔，
// 当划分位置在数组最左、最右侧时使用
#define VEC_SEG_INTERVAL 1E-3

// 进化算法设置
// 交叉概率
#define P_C 0.9
// 变异概率
#define P_M 0.1
// 迭代次数
#define GEN 50
// 种群大小
#define POP_SIZE 10

/*******************************************************************************
 * 			     Haar Adaboost 系列配置
 ******************************************************************************/
// Adaboost 正例、负例输出值之间的最小间隔
// 用于 Haar 特征选择器阈值计算
#define MIN_INTERVAL 1E-3

// 非对称 Adaboost 设置，假阴性的损失将是假阳性损失的 ASYM_CONST 倍
// ASYM_CONST > 0
#define ASYM_CONST 2

// 改进的非对称 ASYM_CONST 设置，将非对称损失的优化延迟到前 ASYM_TURN 轮训练中
// 可避免非对称损失函数的作用迅速消失
#define ASYM_TURN 50

/*******************************************************************************
 * 				    全局配置
 ******************************************************************************/
// 样本数量类型定义
typedef int num_t;

// 样本维度类型定义（向量型样本表示）
typedef int dim_t;

// 样本单个元素的类型定义
typedef double sample_t;

// 样本标签的类型定义
typedef int label_t;

// 多分类任务标签的类型定义
typedef int mlabel_t;

// 用于计算及保存的浮点数类型
typedef double flt_t;

// 图像尺寸类型定义（haar 特征，二维数组样本表示）
typedef int imgsz_t;

// 训练总次数的类型定义（对于多标签问题，需能容纳 T * 标签数量）
typedef unsigned int turn_t;

// 保存中间值数组的长度（对于多标签问题，需能容纳 样本数量 * 标签数量）
typedef unsigned int long_num_t;

#endif
