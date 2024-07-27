#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include "table.h"

/*******************************************************************************
 * 				   宏常量定义
 ******************************************************************************/
#define BASE_SIZE 10

/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
// 确定空白字符类型，如果后跟换行符，则返回真
static bool is_new_line(FILE * file);

/*******************************************************************************
 * 			   函数定义（Vector 结构体）
 ******************************************************************************/
// 初始化，分配初始空间
Vector *newVector(size_t width, size_t n)
{
	if (n == 0)
		n = BASE_SIZE;
	Vector *vector = malloc(sizeof(Vector));
	if (vector == NULL)
		return NULL;

	vector->array = malloc(width * n);
	if (vector->array == NULL) {
		free(vector);
		return NULL;
	}

	vector->width = width;
	vector->size = n;
	vector->len = 0;
	return vector;
}

// 增加一个元素，空间不足则会重新分配空间
Vector *pushVector(Vector * vector, const void *element)
{
	if (vector->len + 1 > vector->size) {
		void *new_ptr;
		new_ptr =
		    realloc(vector->array, vector->width * vector->size * 2);
		if (new_ptr == NULL)
			return NULL;
		else {
			vector->array = new_ptr;
			vector->size *= 2;
		}
	}

	memcpy(vector->array + vector->len * vector->width, element,
	       vector->width);
	++vector->len;

	return vector;
}

// 在末尾处删除一个元素
void *popVector(Vector * vector)
{
	if (vector->len == 0)
		return NULL;
	else {
		--vector->len;
		return vector->array + vector->len * vector->width;
	}
}

// 改变 vector 的容量，使之恰好能容纳已有内容
void setFitVector(Vector * vector)
{
	if (vector->len == 0) {
		free(vector->array);
		vector->array = NULL;
		vector->size = 0;
	} else if (vector->len < vector->size) {
		void *new_ptr =
		    realloc(vector->array, vector->len * vector->width);
		if (new_ptr != NULL) {
			vector->size = vector->len;
			vector->array = new_ptr;
		}
	}
}

// 释放空间
void freeVector(Vector * vector)
{
	free(vector->array);
	free(vector);
}

/*******************************************************************************
 * 			    函数定义（Table 结构体）
 ******************************************************************************/
// 创建 Table，row 是初始行数
Table *newTable(size_t row)
{
	Table *table = malloc(sizeof(Table));
	if (table == NULL)
		return NULL;

	// 使用向量指针表示每一行
	table->row_vec = newVector(sizeof(Vector *), row);
	if (table->row_vec == NULL) {
		free(table);
		return NULL;
	}

	return table;
}

// 添加行向量 vector 到 table，注意将由 table 管理 vector 的内存
Table *pushTable(Table * table, Vector * vector)
{
	if (pushVector(table->row_vec, &vector) == NULL)
		return NULL;

	// 更新 table 中的最大列数
	if (vector->len > table->max_len)
		table->max_len = vector->len;
	return table;
}

// 在末尾处删除一个行向量，并返回该行向量的指针，同时 table 取消对该向量的内存管理
Vector *popTable(Table * table)
{
	Vector **pvector = popVector(table->row_vec);
	if (pvector == NULL)
		return NULL;
	else
		return *pvector;
}

// 根据索引获取指向 table 的某个元素的指针
void *getTableEle(const Table * table, size_t r, size_t c)
{
	Vector **prow_vec;

	// 超出表格范围，返回 NULL
	if ((prow_vec = getVectorEle(table->row_vec, r)) == NULL)
		return NULL;
	return getVectorEle(*prow_vec, c);
}

// 删除 table
void destroyTable(Table * table)
{
	Vector *to_delete = popTable(table);
	while (to_delete != NULL) {
		freeVector(to_delete);
		to_delete = popTable(table);
	}

	freeVector(table->row_vec);
	free(table);
}

// 格式化读取文本文件，filename 为文件名，保存有若干行、列的表格
Table *load(const char *filename)
{
	Table *table;
	if ((table = newTable(0)) == NULL)
		return NULL;

	FILE *file = fopen(filename, "r");
	Vector *row_vec;
	double digit;

	// 初始化第一个行向量
	if ((row_vec = newVector(sizeof(double), 0)) == NULL)
		goto ERROR_LABEL;
	while (fscanf(file, "%lf,", &digit) > 0) {
		if (pushVector(row_vec, &digit) == NULL)
			goto ERROR_LABEL;
		if (is_new_line(file)) {
			// 缩小原行向量的冗余空间
			setFitVector(row_vec);
			// 为新行分配空间
			if (pushTable(table, row_vec) == NULL)
				goto ERROR_LABEL;
			if ((row_vec =
			     newVector(sizeof(double),
				       getVectorCap(row_vec))) == NULL)
				goto ERROR_LABEL;
		}
	}

	// 缩小冗余的空间
	setFitVector(row_vec);
	if (getVectorCap(row_vec) == 0)
		freeVector(row_vec);
	else if (pushTable(table, row_vec) == NULL)
		goto ERROR_LABEL;
	setFitTable(table);

	fclose(file);
	return table;

// 错误处理，无法分配空间（行向量）或遇到非数字字符
// 此处直接删除已有空间
ERROR_LABEL:
	fclose(file);
	destroyTable(table);
	return NULL;
}

// 将 table （load() 返回的表格）转换为二维数组，列数为最大宽度（缺失的数据保存为 nan）
void *table2Array(const Table * table, const void *nan, size_t *row,
		  size_t *col)
{
	// 取得最大列数，以此申请内存空间
	size_t max_len = getTableCol(table);
	size_t row_ct = getTableRow(table);
	Vector **prow_vec;
	// 获取数据的长度
	size_t width;
	prow_vec = getTableVec(table, 0);
	width = getVectorWidth(*prow_vec);

	// 二维数组
	void *array;
	if ((array = malloc(width * max_len * row_ct)) == NULL)
		return NULL;

	void *ptr = array;
	size_t i, j;
	size_t len;
	for (i = 0; i < row_ct; ++i) {
		prow_vec = getTableVec(table, i);
		len = width * getVectorLen(*prow_vec);
		memcpy(ptr, getVectorEle(*prow_vec, 0), len);
		printf("%zd\n", i);
		ptr += len;
		// 填充缺失值
		for (j = len; j < max_len; ++j, ptr += width)
			memcpy(ptr, nan, width);
	}

	*row = row_ct;
	*col = max_len;
	return array;
}

/*******************************************************************************
 * 				  静态函数定义
 ******************************************************************************/
bool is_new_line(FILE * file)
{
	int ch;
	while (isspace(ch = fgetc(file)))
		if (ch == '\n')
			return true;
	ungetc(ch, file);
	return false;
}
