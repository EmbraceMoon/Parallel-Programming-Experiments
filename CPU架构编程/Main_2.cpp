#include<iostream>
#include<Windows.h>
using namespace std;

const int N = 4096;
int a[N];
void init() {
	for (int i = 0;i < N;i++)
		a[i] = i * i + 2 * i;
}

int normal() {
	int sum = 0;
	for (int i = 0; i < N; i++)
		sum += a[i];
	return sum;
}

int muti_link() {
	int sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
	for (int i = 0;i < N;i += 4) {
		sum1 += a[i];
		sum2 += a[i + 1];
		sum3 += a[i + 2];
		sum4 += a[i + 3];
	}
	return sum1 + sum2 + sum3 + sum4;
}

int rec(int p, int q) {
	if (p == q) return a[p];
	return rec(p, (p + q) / 2) + rec(((p + q) / 2) + 1, q);
}

int recursion() {
	return rec(0, N - 1);
}

int main() {
	long long head, tail, freq;
	int M = 4000;
	init();

	QueryPerformanceFrequency((LARGE_INTEGER*) &freq);
	QueryPerformanceCounter((LARGE_INTEGER*) &head);
	for (int i = 0;i < M;i++) normal();
	QueryPerformanceCounter((LARGE_INTEGER*) &tail);
	cout << "Normal Col: " << (tail - head) * 1000.0 / freq << "ms" << endl;

	QueryPerformanceCounter((LARGE_INTEGER*) &head);
	for (int i = 0;i < M;i++) muti_link();
	QueryPerformanceCounter((LARGE_INTEGER*) &tail);
	cout << "Muti-link Col: " << (tail - head) * 1000.0 / freq << "ms" << endl;

	QueryPerformanceCounter((LARGE_INTEGER*) &head);
	for (int i = 0;i < M;i++) recursion();
	QueryPerformanceCounter((LARGE_INTEGER*) &tail);
	cout << "Recursion Col: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	return 0;
}