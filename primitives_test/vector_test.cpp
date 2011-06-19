#include <gtest/gtest.h>
#include "../primitives/types.h"

TEST(VectorTest,AllocatesAndSetsCorrectSize){
	Vector<float> v;
	EXPECT_EQ((const float*)(0), v.data());
	v.set_dimension(32);
	EXPECT_EQ(32, v.size());
	EXPECT_NE((const float*)(0), v.data());
}

TEST(VectorTest,ConstructorsWorkFine){
	Vector<float> v0;
	EXPECT_EQ(0, v0.data());
	EXPECT_EQ(0, v0.size());
	
	Vector<float> v1(32);
	EXPECT_NE((const float*)0, v1.data());
	EXPECT_EQ(32, v1.size());

	float aaa[] = {1,2,3,4,5};
	Vector<float> v2(&aaa[0],5,false);
	EXPECT_EQ(&aaa[0], v2.data());
	EXPECT_EQ(5, v2.size());

}

