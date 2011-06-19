#pragma once

//
inline void equation(float t, float* x, float* F){
	F[0] = t*2;
}

inline void sin_eq(double t, double* x, double* F)
{
	F[0] = 3*sin(4*t);
}

inline double sin_eq_analytic(double tm)
{
	return -3./4*cos(4*tm);
}
