#include<iostream>
#include<Windows.h>
using namespace std;

const int N = 4096;

int a[N][N], v[N], res[N];
void init() {
	for (int i = 0; i < N; i++) {
		//初始化向量与矩阵
		v[i] = i ^ 2 + 3 * i - 7;
		for (int j = 0;j < N;j++) {
			a[i][j] = i + j;
		}
	}
}

void normal() {
	for (int i = 0; i < N; i++) res[i] = 0;
	for (int i = 0;i < N;i++) {
		for (int j = 0;j < N;j++) {
			res[i] += a[j][i] * v[j];
		}
	}
	//for (int i = 0; i < N; i++) cout << res[i] << ' ';
	//cout << endl;
}

void cache() {
	for (int i = 0; i < N; i++) res[i] = 0;
	for (int i = 0;i < N;i++) {
		for (int j = 0;j < N;j++) {
			res[j] += a[i][j] * v[j];
		}
	}
	//for (int i = 0; i < N; i++) cout << res[i] << ' ';
	//cout << endl;
}

int main() {
	long long head, tail, freq;
	init();

	QueryPerformanceFrequency((LARGE_INTEGER*) &freq);
	QueryPerformanceCounter((LARGE_INTEGER*) &head);
	normal();
	QueryPerformanceCounter((LARGE_INTEGER*) &tail);
	cout << "Normal Col: " << (tail - head) * 1000.0 / freq << "ms" << endl;

	QueryPerformanceCounter((LARGE_INTEGER*) &head);
	cache();
	QueryPerformanceCounter((LARGE_INTEGER*) &tail);
	cout << "Cache Col: " << (tail - head) * 1000.0 / freq << "ms" << endl;

	return 0;
}