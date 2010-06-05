#pragma once

#include <windows.h>
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"
#include <tbb/task_scheduler_init.h>
#include <tbb/cache_aligned_allocator.h>

#define NTHREADS /*43750*/87500 /*175000*/ /*350000*/

typedef tbb::blocked_range<int>  TbbRange;

template<class Float_t> struct Tbb_worker{

	typedef Float_t Float;
	
};

template<class Float_t> struct Unary_tbb_worker:public Tbb_worker<Float_t>
{
	Float * m_result;
	Float * m_a;

	Float m_c;

	Unary_tbb_worker(Float*a,Float*res):m_result(res),m_a(a){}
};

template<class Float_t> struct Binary_tbb_worker:public Tbb_worker<Float_t>
{

	Float * m_result;
	Float * m_a;
	Float * m_b;
	Float m_c;

	Binary_tbb_worker(Float*a,Float*b,Float*res):m_result(res),m_a(a),m_b(b){}
};

static tbb::cache_aligned_allocator<char> t_alloc;

inline void* aligned_malloc(size_t sz){
	return VirtualAlloc(0,sz,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
}
inline void aligned_free(void* p){
	VirtualFree(p,0,MEM_DECOMMIT|MEM_FREE);
}
