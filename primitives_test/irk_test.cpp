#include <gtest/gtest.h>
#include "../primitives/types.h"
#include "../primitives/ode_solver.h"
#include "../primitives/RefBlas.h"
#include "../primitives/irk.h"

#include "test_functions.h"

TEST(RadauIA3Test,SolvesTestEq1){
	test_equation1<IrkRadauIA3Step>();
}

TEST(RadauIA3Test,SolvesTestEq2){
	test_equation2<IrkRadauIA3Step>();
}

TEST(Gauss4Test,SolvesTestEq1){
	test_equation1<IrkGauss4Step>();
}

TEST(Gauss4Test,SolvesTestEq2){
	test_equation2<IrkGauss4Step>();
}


TEST(LobattoIIIC6Test,SolvesTestEq1){
	test_equation1<IrkLobattoIIIC6Step>();
}

TEST(LobattoIIIC6Test,SolvesTestEq2){
	test_equation2<IrkLobattoIIIC6Step>();
}
