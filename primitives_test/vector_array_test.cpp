#include <gtest/gtest.h>
#include "../primitives/types.h"
#include "../primitives/RefBlas.h"

typedef RefBlas<float> MyBlas;
typedef VectorArray<MyBlas> MyVectorArray;
TEST(VectorArrayTest,AllocatesAndSetsCorrectSize){
	MyVectorArray v(5,7);
	EXPECT_EQ(v.occupied_items(),0);
	
}

template<class T>static bool cmp_array(const T* left, const T* right,unsigned int sz){
	return std::equal(left,left+sz,right);
}

TEST(VectorArrayTest,PushAndSet){
	MyVectorArray v(3,3);

	float test_data1[] = {1,1,1};
	float test_data2[] = {2,2,2};
	float test_data3[] = {3,3,3};
	float test_data4[] = {4,4,4};

	v.push(test_data1);
	EXPECT_EQ(cmp_array(v.get_vector(0),test_data1,3),true);

	v.push(test_data2);
	EXPECT_EQ(cmp_array(v.get_vector(0),test_data1,3),true);
	EXPECT_EQ(cmp_array(v.get_vector(1),test_data2,3),true);

	v.push(test_data3);
	EXPECT_EQ(cmp_array(v.get_vector(0),test_data1,3),true);
	EXPECT_EQ(cmp_array(v.get_vector(1),test_data2,3),true);
	EXPECT_EQ(cmp_array(v.get_vector(2),test_data3,3),true);

	v.set_at(1,test_data4);
	EXPECT_EQ(cmp_array(v.get_vector(0),test_data1,3),true);
	EXPECT_EQ(cmp_array(v.get_vector(1),test_data4,3),true);
	EXPECT_EQ(cmp_array(v.get_vector(2),test_data3,3),true);

	EXPECT_THROW(v.push(test_data3),std::exception);

}

TEST(VectorArrayTest,Clear){
	MyVectorArray v(3,3);

	float test_data1[] = {1,1,1};
	float test_data2[] = {2,2,1};
	
	v.push(test_data1);
	v.clear();
	EXPECT_EQ(v.occupied_items(),0);

	v.push(test_data2);
	EXPECT_EQ(cmp_array(v.get_vector(0),test_data2,3),true);
}



