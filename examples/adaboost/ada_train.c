#include <stdio.h>
#include <stdlib.h>
#include "table.h"
#include "adaboost.h"

// 从 table 中构建样本（数字 0 及数字 1 两类）
// 返回值：返回读取的样本数量
int getSamples(const Table * table, int m, int n, double X[m][n - 1], int Y[m]);

// 获取错误率
double getErrRate(const struct vec_adaboost *ada, bool using_cf, int m, int n,
		  double X[m][n - 1], int Y[m],
		  const struct vec_ada_handles *hl);

int main(void)
{
	// 读取训练集
	Table *table = load("./dataset/pendigits.tra");
	int m = getTableRow(table);
	int n = getTableCol(table);
	double (*X) [n - 1] = malloc(sizeof(double) * m * (n-1));
	int * Y = malloc(sizeof(int) * m);
	if (X == NULL || Y == NULL)
		goto malloc_err;
	m = getSamples(table, m, n, X, Y);
	destroyTable(table);

	// 配置 AdaBoost
	struct vec_adaboost adaboost;
	struct vec_ada_handles handles;
	ada_set_vec(&handles, ADA_FOLD, ADA_NO_CONFIDENT, ADA_DISCRETE);
	// 训练
	if (!handles.train(&adaboost, 100, m, n - 1, X, Y, true, &handles.wl_hl)) {
		fprintf(stderr, "Training Error.\n");
		return 1;
	}
	// 获取错误率，如果使用了 ADA_CONFIDENT 选项，应将 false 改为 true
	double err = getErrRate(&adaboost, false, m, n, X, Y, &handles);
	printf("Training Error Rate: %lf\n", err);
	free(X);
	free(Y);

	// 读取测试集
	table = load("./dataset/pendigits.tes");
	m = getTableRow(table);
	X= malloc(sizeof(double) * m * (n-1));
	Y = malloc(sizeof(int) * m);
	if (X == NULL || Y == NULL)
		goto malloc_err;
	m = getSamples(table, m, n, X, Y);
	destroyTable(table);
	// 获取错误率，如果使用了 ADA_CONFIDENT 选项，应将 false 改为 true
	err = getErrRate(&adaboost, false, m, n, X, Y, &handles);
	printf("Testing Error Rate: %lf\n", err);
	handles.free(&adaboost, &handles.wl_hl);
	free(X);
	free(Y);

	return 0;
malloc_err:
	free(X);
	free(Y);
	destroyTable(table);
	exit(EXIT_FAILURE);
}

int getSamples(const Table * table, int m, int n, double X[m][n - 1], int Y[m])
{
	int i, j;
	int curr = 0;
	for (i = 0; i < m; ++i) {
		Vector **row = getTableVec(table, i);
		Y[curr] = *(double *)getVectorEle(*row, n - 1);
		if (Y[curr] != 0 && Y[curr] != 1)
			continue;
		for (j = 0; j < n - 1; ++j)
			X[curr][j] = *(double *)getVectorEle(*row, j);
		if (Y[curr] != 1)
			Y[curr] = -1;
		curr++;
	}

	return curr;
}

double getErrRate(const struct vec_adaboost *ada, bool using_cf, int m, int n,
		  double X[m][n - 1], int Y[m],
		  const struct vec_ada_handles *hl)
{
	double err = 0;
	for (int i = 0; i < m; ++i)
		if (using_cf) {
			if (hl->cf_h(ada, X[i], n - 1, &hl->wl_hl) *
			    Y[i] <= 0)
				++err;
		} else if (hl->h(ada, X[i], n - 1, &hl->wl_hl) *
			   Y[i] <= 0)
			++err;

	return err / m;
}
