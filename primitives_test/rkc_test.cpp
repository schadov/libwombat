#include <gtest/gtest.h>
#include "../primitives/types.h"
#include "../primitives/ode_solver.h"
#include "../primitives/RefBlas.h"
#include "../primitives/rkc.h"

#include "test_functions.h"

TEST(RKC3Test,SolvesTestEq1){
	test_equation1<RKCStep3>();
}

TEST(RKC3Test,SolvesTestEq2){
	test_equation2<RKCStep3>();
}


TEST(RKC4Test,SolvesTestEq1){
	test_equation1<RKCStep4>();
}

TEST(RKC4Test,SolvesTestEq2){
	test_equation2<RKCStep4>();
}


TEST(RKC5Test,SolvesTestEq1){
	test_equation1<RKCStep5>();
}

TEST(RKC5Test,SolvesTestEq2){
	test_equation2<RKCStep5>();
}


TEST(RKC6Test,SolvesTestEq1){
	test_equation1<RKCStep6>();
}

TEST(RKC6Test,SolvesTestEq2){
	test_equation2<RKCStep6>();
}

TEST(RKC7Test,SolvesTestEq1){
	test_equation1<RKCStep7>();
}

TEST(RKC7Test,SolvesTestEq2){
	test_equation2<RKCStep7>();
}


TEST(RKC8Test,SolvesTestEq1){
	test_equation1<RKCStep8>();
}

TEST(RKC8Test,SolvesTestEq2){
	test_equation2<RKCStep8>();
}

