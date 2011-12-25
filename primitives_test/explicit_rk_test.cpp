#include <gtest/gtest.h>
#include "../primitives/types.h"
#include "../primitives/ode_solver.h"
#include "../primitives/RefBlas.h"
#include "../primitives/dirk.h"

#include "test_functions.h"

TEST(BogackihSampineTest, SolvesTestEq1){
	test_equation1<BogackiShampineStep>();
}

TEST(BogackihSampineTest, SolvesTestEq2){
	test_equation2<BogackiShampineStep>();
}
