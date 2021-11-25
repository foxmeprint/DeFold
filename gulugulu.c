#include<stdio.h>
#include<malloc.h>
#include "test.h"


int iamA = 31456;

 double  chacheng(double p1x, double p2x, double p3x, double p1y, double p2y, double p3y) {
	return ((p2x - p1x) * (p3y - p1y)) - ((p2y - p1y) * (p3x - p1x));
}
_declspec(dllexport) float _stdcall test_mul(float x[2][5], int y) {
	return x[1][2] * y;
}
_declspec(dllexport) int _stdcall is_in_tri(double p1x, double p1y, double p2x, double p2y, double p3x, double p3y, double px, double py) {
	if (chacheng(p1x, p2x, p3x, p1y, p2y, p3y) < 0) {
		return is_in_tri(p1x,p1y, p3x,p3y, p2x,p2y, px, py);
	}
	if (chacheng(p1x, p2x, px, p1y, p2y, py) > 0) {
		if (chacheng(p2x, p3x, px, p2y, p3y, py) > 0) {
			if (chacheng(p3x, p1x, px, p3y, p1y, py) > 0) {
				return 1;
			}
		}
	}
	return 0;
}

_declspec(dllexport) void _stdcall hello(int** a,int m,int n) {

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			a[i][j]++;

		}
	}
}

_declspec(dllexport) double _stdcall hi(int* matrix, int rows, int columns) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			printf("matrix[%d][%d] = %d\n", i, j, matrix[i * rows + j]);
		}
	}
	return 0;
}

 int mul(int a, int b) {
	return a * b;
}

_declspec(dllexport) int _stdcall filter(float* matrix,int col,int row, float* core,int size, float* ans) {
	int s1 = col - size + 1, s2 = row - size + 1;
	
	for (int i = 0; i < s1; i++) {
		for (int j = 0; j < s2; j++) {
			float sum = 0;
			for (int m = 0; m < size; m++) {
				for (int n = 0; n < size; n++) {
					sum += (matrix[(i + m) * row + j + n] * core[m * size + n]);

				}
			}
			ans[(i + 1) * row + j + 1] = sum * sum;
		}
	}


	return 0;
}

_declspec(dllexport) int _stdcall im2col(float* arr, int col, int row, int h, int w, float* ans) {
	int step_col = 0, step_row = 0, step = 0;
	step_col = col - h + 1;
	step_row = row - w + 1;
	step = step_col * step_row;

	for (int c = 0; c < step_col; c++) {
		for (int r = 0; r < step_row; r++) {
			int i = c * step_row + r;
			for (int a = 0; a < h; a++) {
				for (int b = 0; b < w; b++) {
					int j = a * w + b;
					//ans[i][j] = arr[c + a][r + b];
					ans[i * h * w + j] = arr[(c + a) * row + (r + b)];

				}
			}

		}
	}

	return 1;
}
