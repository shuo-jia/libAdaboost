#ifndef PRETRAIN_H
#define PRETRAIN_H
#include "cascade.h"
#include "boost_cfg.h"

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/*
 * 预训练函数，进行初步的训练，在训练过程中逐步构造非人脸样本
 * pcascade: 未初始化的结构体地址，指向一个级联分类器
 * max_posi: 指定阳性样本的最大数量
 * face_size: 人脸尺寸，即训练图片的长或宽
 * err_rate: 当在训练集上的错误率小于该值时，训练停止
 * posi_mark: 阳性样本标注文件的路径，标注文件的每行表示一个矩形框，格式如下
 * 	文件名 左上角横坐标 左上角纵坐标 矩形框高度 矩形框宽度
 * neg_mark: 阴性样本标注文件的路径，文件格式同 posi_mark。训练时所产生的假阳
 * 	性样本将作为阴性样本写入标注文件中。
 * dir_path: 图片文件所在目录（末尾需带有“/”符号）
 * 返回值：成功则返回真，否则返回假
 */
bool pre_train (struct cascade * pcascade, num_t max_posi, imgsz_t face_size,
		flt_t err_rate, const char * posi_mark, const char * neg_mark,
		const char * dir_path, const struct haar_ada_handles *hl);

#endif
