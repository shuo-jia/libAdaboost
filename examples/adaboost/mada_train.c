#include <stdio.h>
#include <stdlib.h>
#include "table.h"
#include "adaboost.h"

void getSamples(const Table * table, int m, int n, double X[m][n - 1],
		int Y[m]);

double getErrRate(const struct mvec_adaboost *ada, int m, int n,
		  double X[m][n - 1], int Y[m],
		  const struct mvec_ada_handles *hl);

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
	getSamples(table, m, n, X, Y);
	destroyTable(table);

	// 配置 AdaBoost
	struct mvec_adaboost adaboost;
	struct mvec_ada_handles handles;
	ada_set_mvec(&handles, ADA_HLOSS, ADA_FOLD, ADA_CONTINUOUS);
	// 训练并获取训练误差
	if (!handles.train(&adaboost, 350, m, n - 1, X, Y, true, &handles.wl_hl)) {
		fprintf(stderr, "Training Error.\n");
		return 1;
	}
	double err = getErrRate(&adaboost, m, n, X, Y, &handles);
	printf("Training Error rate: %lf\n", err);
	free(X);
	free(Y);

	// 读取测试集并输出测试误差
	table = load("./dataset/pendigits.tes");
	m = getTableRow(table);
	X = malloc(sizeof(double) * m * (n-1));
	Y = malloc(sizeof(int) * m);
	if (X == NULL || Y == NULL)
		goto malloc_err;
	getSamples(table, m, n, X, Y);
	destroyTable(table);
	err = getErrRate(&adaboost, m, n, X, Y, &handles);
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

void getSamples(const Table * table, int m, int n, double X[m][n - 1], int Y[m])
{
	int i, j;
	int curr = 0;
	for (i = 0; i < m; ++i) {
		Vector **row = getTableVec(table, i);
		Y[i] = *(double *)getVectorEle(*row, n - 1);
		for (j = 0; j < n - 1; ++j)
			X[i][j] = *(double *)getVectorEle(*row, j);
	}
}

double getErrRate(const struct mvec_adaboost *ada, int m, int n,
		  double X[m][n - 1], int Y[m],
		  const struct mvec_ada_handles *hl)
{
	double err = 0;
	for (int i = 0; i < m; ++i)
		if (hl->h(ada, X[i], n - 1, &hl->wl_hl) != Y[i])
			++err;

	return err / m;
}
