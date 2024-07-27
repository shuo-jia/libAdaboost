#ifndef SAMPLE_H
#define SAMPLE_H

#include <stdbool.h>
#include "cascade.h"
#include "image_type.h"
#include "boost_cfg.h"

/*******************************************************************************
 * 				   宏常量定义
 ******************************************************************************/
// 文件名最大长度
#define MAX_FILENAME 128

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
// 样本类型
struct sample {
	sample_t ** X;			// 积分图指针数组
	sample_t ** X2;			// （像素值平方的）积分图指针数组
	label_t * Y;			// 样本标签（+1 或 -1）数组
	num_t m;			// 样本数量
};

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/*
 * 构造样本集，返回一个已初始化的结构体
 * 从标注文件（mark）中读取正例样本，并随机抽取背景作为负例样本
 * sample: 未初始化的结构体
 * max_posi: 阳性样本的最大数量
 * neg_per_posi: 一张图片包含一张人脸（阳性样本），该值指示每张图片生成的非人脸
 * 	（阴性）样本的最大数量
 * face_size: 人脸样本的尺寸，即图片长或宽（正方形）
 * p_mark: 标注文件。标注文件给出人脸矩形框的位置和大小，一行表示一个矩形框，每
 * 	一行的格式为如下字符串：
 * 	文件名 左上角横坐标 左上角纵坐标 矩形框高度 矩形框宽度
 * n_mark: 保存随机产生的非人脸图片标注信息，该参数为 NULL 时不保存信息
 * dir_path: 指示 fmark 标注的图片文件所在目录，结尾需有分隔符"/"
 * 返回值：成功则返回真，否则返回假
 */
bool get_rand_sample (struct sample * sample, num_t max_posi,
		num_t neg_per_posi, imgsz_t face_size, FILE * p_mark,
		FILE * n_mark, const char * dir_path);

/*
 * 构造样本集，返回一个已初始化的结构体
 * 从标注文件中读取阳性和阴性样本
 * sample: 未初始化的结构体
 * max_posi: 阳性样本的最大数量
 * face_size: 人脸样本的尺寸，即图片长或宽（正方形）
 * p_mark: 标注文件，仅前 max_posi 个标注框被使用。标注文件给出人脸矩形框的位置
 * 	和大小，一行表示一个矩形框，每一行的格式为如下字符串：
 * 	文件名 左上角横坐标 左上角纵坐标 矩形框高度 矩形框宽度
 * n_mark: 标注文件，给出非人脸矩形框的信息，格式与 p_mark 相同
 * dir_path: 指示 fmark 标注的图片文件所在目录，结尾需有分隔符"/"
 * 返回值：成功则返回真，否则返回假
 */
bool get_mark_sample (struct sample * sample, num_t max_posi, imgsz_t face_size,
		FILE * p_mark, FILE * n_mark, const char * dir_path);

/*
 * 释放结构体内存
 * sample: 已初始化的结构体
 */
void free_sample (struct sample * sample);

/*
 * 将 struct cas_det_rect 类型转换为 struct rectangle 类型
 * rect: 目标变量
 * cas_rect: 源变量
 */
void rect_cast (struct rectangle *rect, const struct cas_rect * cas_rect);

/*
 * 将 struct cas_det_rect 类型转换为 struct rectangle 类型
 * rect: 目标变量
 * cas_rect: 源变量
 */
void rect_cast_r (struct cas_rect * cas_rect, const struct rectangle *rect);

#endif
