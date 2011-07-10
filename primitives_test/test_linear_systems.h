#pragma once
#include <algorithm>

double myrand(){
	return double(rand()%100)/50.-1.0;
}

template<class RealT,class Matrix,class Vector>
void generate_linear_system(unsigned int dim, Matrix& A, Vector& b,Vector& roots, unsigned int seed){

	srand(seed);

	std::generate(&roots[0],&roots[dim],myrand);

	for (unsigned int i = 0;i<dim;++i)
	{
		std::vector<RealT> coeff(dim);	
		std::generate(coeff.begin(),coeff.end(),myrand);

		for (unsigned int j=0;j<dim;++j){
			A(i,j) = coeff[j];
		}

		//calculate b
		RealT sum = 0;
		for (unsigned int j=0;j<dim;++j)
		{
			sum += coeff[j]*roots[j];
		}
		b[i] = sum;
	}

}