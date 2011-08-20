#include <gtest/gtest.h>
#include "../primitives/types.h"

#define SCL_SECURE_NO_WARNINGS 0

TEST(VectorDequeTest,AllocatesAndSetsCorrectSize){
	VectorDeque<float> v(5,7);
	EXPECT_EQ(v.get_N(),5);
	EXPECT_EQ(v.get_capacity(),7);
}

 template<class T>static bool cmp_array(const T* left, const T* right,unsigned int sz){
	return std::equal(left,left+sz,right);
}

TEST(VectorDequeTest,Push){
	VectorDeque<float> v(3,3);
	
	float test_data1[] = {1,1,1};
	float test_data2[] = {2,2,2};
	float test_data3[] = {3,3,3};
	float test_data4[] = {4,4,4};
	float test_data5[] = {5,5,5};

	v.push(test_data1);
	EXPECT_EQ(cmp_array(v.get_vector(0),test_data1,3),true);

	v.push(test_data2);
	EXPECT_EQ(cmp_array(v.get_vector(0),test_data1,3),true);
	EXPECT_EQ(cmp_array(v.get_vector(1),test_data2,3),true);

	v.push(test_data3);
	EXPECT_EQ(cmp_array(v.get_vector(0),test_data1,3),true);
	EXPECT_EQ(cmp_array(v.get_vector(1),test_data2,3),true);
	EXPECT_EQ(cmp_array(v.get_vector(2),test_data3,3),true);

	v.push(test_data4);
	EXPECT_EQ(cmp_array(v.get_vector(0),test_data2,3),true);
	EXPECT_EQ(cmp_array(v.get_vector(1),test_data3,3),true);
	EXPECT_EQ(cmp_array(v.get_vector(2),test_data4,3),true);

	v.push(test_data5);
	EXPECT_EQ(cmp_array(v.get_vector(0),test_data3,3),true);
	EXPECT_EQ(cmp_array(v.get_vector(1),test_data4,3),true);
	EXPECT_EQ(cmp_array(v.get_vector(2),test_data5,3),true);
}

