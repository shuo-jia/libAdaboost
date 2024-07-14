#ifndef MATLAB_LIB_H
#define MATLAB_LIB_H
#include <stddef.h>

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
typedef struct {
	void *array;		// 数组地址
	size_t width;		// 单个元素的长度
	size_t size;		// 数组容量
	size_t len;		// 当前数组长度
} Vector;

typedef struct {
	Vector *row_vec;	// 行向量指针数组
	size_t max_len;		// 行向量的最大长度
} Table;

/*******************************************************************************
 * 			   函数声明（Vector 结构体）
 ******************************************************************************/
// 初始化，分配初始空间
// width 是 Vector 每个元素的长度，n 是初始元素数量，使用 0 设置默认数量
Vector *newVector(size_t width, size_t n);

// 增加一个元素，空间不足则会重新分配空间（成功则返回 vector 地址，否则返回 NULL）
Vector *pushVector(Vector * vector, const void *element);

// 在末尾处删除一个元素，并返回指向该元素的指针
void *popVector(Vector * vector);

// 根据索引获取元素
static inline void *getVectorEle(const Vector * vector, unsigned int index)
{
	if (index > vector->len)
		return NULL;

	return vector->array + vector->width * index;
}

// 获取容量
static inline size_t getVectorCap(const Vector * vector)
{
	return vector->size;
}

// 获取长度
static inline size_t getVectorLen(const Vector * vector)
{
	return vector->len;
}

// 获取数据宽度
static inline size_t getVectorWidth(const Vector * vector)
{
	return vector->width;
}

// 改变 vector 的容量，使之恰好能容纳已有内容
void setFitVector(Vector * vector);

// 释放空间
void freeVector(Vector * vector);

/*******************************************************************************
 * 			    函数声明（Table 结构体）
 ******************************************************************************/
// 创建 Table，row 是初始行数（0 表示采用默认初始行数）
Table *newTable(size_t row);

// 添加行向量 vector 到 table，注意将由 table 管理 vector 的内存
Table *pushTable(Table * table, Vector * vector);

// 在末尾处删除一个行向量，并返回该行向量的指针，同时 table 取消对该向量的内存
// 管理
Vector *popTable(Table * table);

// 根据索引获取 table 的某个行向量的地址
static inline Vector **getTableVec(const Table * table, size_t r)
{
	return getVectorEle(table->row_vec, r);
}

// 根据索引获取指向 table 的某个元素的指针，如果超过范围则返回 NULL
void *getTableEle(const Table * table, size_t r, size_t c);

// 删除 table
void destroyTable(Table * Table);

// 获取 table 的最大宽度
static inline size_t getTableCol(const Table * table)
{
	return table->max_len;
}

// 获取 table 的长度（行数）
static inline size_t getTableRow(const Table * table)
{
	return getVectorLen(table->row_vec);
}

// 设置 table 的行向量容量，使之恰好能容纳已有行向量
static inline void setFitTable(Table * table)
{
	setFitVector(table->row_vec);
}

// 格式化读取文本文件，filename 为文件名，保存有若干行、列的表格
Table *load(const char *filename);

// 将 table 转换为二维数组，列数为最大宽度（缺失的数据保存为 nan）
// 注意每个元素须有相同的长度，如果某行的数据少于最大列数，则在末尾将缺失值补充
// 为 nan 指向的变量值
void *table2Array(const Table * table, const void *nan, size_t *row,
		  size_t *col);

/*******************************************************************************
 * 				    常用函数
 ******************************************************************************/

#endif
