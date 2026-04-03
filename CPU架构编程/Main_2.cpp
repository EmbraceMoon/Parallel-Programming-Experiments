#include<iostream>
#include<Windows.h>
using namespace std;

const int N = 4096;
int a[N], b[N];
void init() {
	for (int i = 0;i < N;i++)
		a[i] = 2 * i;
}

int normal() {
	int sum = 0;
	for (int i = 0; i < N; i++)
		sum += a[i];
	return sum;
}

int multi_link() {
	int sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
	for (int i = 0;i < N;i += 4) {
		sum1 += a[i];
		sum2 += a[i + 1];
		sum3 += a[i + 2];
		sum4 += a[i + 3];
	}
	return sum1 + sum2 + sum3 + sum4;
}

int recursion() {
	for (int m = N; m > 1; m /= 2)
		for (int i = 0; i < m / 2; i++)
			b[i] = b[i * 2] + b[i * 2 + 1];
	return b[0];
}

int main() {
	long long head, tail, freq, count = 0;
	int M = 4500;
	init();

	QueryPerformanceFrequency((LARGE_INTEGER*) &freq);
	QueryPerformanceCounter((LARGE_INTEGER*) &head);
	for (int i = 0;i < M;i++) normal();
	QueryPerformanceCounter((LARGE_INTEGER*) &tail);
	cout << "Normal Col: " << (tail - head) * 1000.0 / freq << "ms" << endl;

	QueryPerformanceCounter((LARGE_INTEGER*) &head);
	for (int i = 0;i < M;i++) multi_link();
	QueryPerformanceCounter((LARGE_INTEGER*) &tail);
	cout << "Multi-link Col: " << (tail - head) * 1000.0 / freq << "ms" << endl;

	for (int i = 0;i < M;i++) {
		memcpy(b, a, sizeof(a));
		QueryPerformanceCounter((LARGE_INTEGER*) &head);
		recursion();
		QueryPerformanceCounter((LARGE_INTEGER*) &tail);
		count += tail - head;
	}
	cout << "Recursion Col: " << count * 1000.0 / freq << "ms" << endl;
	return 0;
}