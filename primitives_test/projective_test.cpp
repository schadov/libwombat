#include <gtest/gtest.h>
#include "../primitives/types.h"
#include "../primitives/ode_solver.h"
#include "../primitives/RefBlas.h"
#include "../primitives/projective.h"

#include "test_functions.h"

TEST(PFETest,SolvesTestEq1){
	test_equation1<PFEStep>();
}

TEST(PFETest,SolvesTestEq2){
	test_equation2<PFEStep>();
}

