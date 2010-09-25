#pragma  once
#include <time.h>
#include <windows.h>

class profile {
	clock_t T1,T2;
	char buf[32];
	LARGE_INTEGER li;
	unsigned __int64 norm; 
public:
	profile()
	{
		T1=clock();
	}
	clock_t get_time()
	{
		return clock()-T1;
	}
	const char* get_time_as_string()
	{
		T2=clock()-T1;
		sprintf(buf,"%ld",T2);
		return (const char*)buf;
	}
	void reset()
	{
		T1=clock();
	}
	void inline start_hi_res(unsigned __int64 n=1000)
	{
		QueryPerformanceCounter(&li);		
		norm=n;
	}
	unsigned int query_hi_res()
	{
		LARGE_INTEGER li2,freq;
		QueryPerformanceCounter(&li2);
		QueryPerformanceFrequency(&freq);
		double tpc = norm/(double)freq.QuadPart;
		unsigned int res = (int)((li2.QuadPart-li.QuadPart)*tpc);
		return res;
	}
};