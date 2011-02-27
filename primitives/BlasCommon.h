#pragma  once

namespace BlasCommon{
	template<class RealT> struct CRS_matrix{
		RealT* a;
		unsigned int* colind;
		unsigned int* rowptr;
		unsigned int nrows;
		unsigned int nelements;
	};

	typedef float FloatType;
}

