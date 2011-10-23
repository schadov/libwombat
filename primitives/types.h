#pragma once
#include <deque>

template <class RealT> class Vector{
	RealT * pdata_;
	unsigned int size_;
	bool own_;

	void set_size(unsigned int new_sz){
		if(pdata_ && own_){
			destroy();
		}
		pdata_ = new RealT[new_sz];
		size_ = new_sz;
		own_ = true;
	}

	void destroy(){
		delete [] pdata_;
		pdata_ = 0;
		size_ = 0;
		own_ = false;
	}


public:
	Vector():pdata_(0),size_(0),own_(true){}
	Vector(RealT* data, unsigned int sz, bool own=false):pdata_(data),size_(sz),own_(own){}
	explicit Vector(unsigned int sz):pdata_(0),size_(0),own_(true){
		set_dimension(sz);
	}

	RealT& operator[](unsigned int i){
		return pdata_[i];
	}

	const RealT& operator[](unsigned int i) const{
		return pdata_[i];
	}

	RealT * data(){
		return pdata_;
	}

	const RealT * data() const{
		return pdata_;
	}

	/*operator const RealT*()const{
		return data();
	}*/


	unsigned int size()const{
		return size_;
	}

	void assign(RealT* data, unsigned int sz){
		set_size(sz);
		memcpy(pdata_,data,sizeof(RealT)*sz);
	}

	void set_dimension(unsigned int new_sz){
		assert(pdata_==0);
		set_size(new_sz);
	}

	~Vector(){
		if(pdata_ && own_){
			destroy();
		}
	}
};


template<class Blas> 
class BlasVector{
protected:
	typename Blas::FloatType* data_;
	unsigned int sz_;

	void allocate(unsigned int N){
		Blas::allocate(N,data_);
		sz_ = N;
	}

	void deallocate(){
		if(data_==0)
			return;
		Blas::deallocate(data_);
		sz_ = 0;
		data_ = 0;
	}

	BlasVector(const BlasVector<Blas> &); //private copy ctor

public:
	BlasVector(unsigned int N):sz_(0),data_(0){
		reset(N);
	}

	BlasVector():sz_(0),data_(0){
	}

	~BlasVector(){
		deallocate();
	}

	void reset(unsigned int N){
		if(sz_!=0 && data_!=0)
			deallocate();
		allocate(N);
	}

	operator typename Blas::FloatType*(){
		return data_;
	}

	operator const typename  Blas::FloatType*() const{
		return data_;
	}

};

template<class Blas> class BlasMatrix: public BlasVector<Blas>{
	unsigned int dim_;
public:
	BlasMatrix(unsigned int N):BlasVector<Blas>(N*N),dim_(N){}

	BlasMatrix():BlasVector<Blas>(),dim_(0){}

	void reset(unsigned int N){
		dim_ = N;
		BlasVector<Blas>::reset(N*N);
	}

	 typename Blas::FloatType& operator()(unsigned int i,unsigned int j){
		return  BlasVector<Blas>::data_[i*dim_+j];
	}

	const typename  Blas::FloatType& operator()(unsigned int i,unsigned int j) const{
		return BlasVector<Blas>::data_[i*dim_+j];
	}

	typename Blas::FloatType* as_vector(){
		return BlasVector<Blas>::data_;
	}

	const typename  Blas::FloatType* as_vector() const{
		return BlasVector<Blas>::data_;
	}

	unsigned int get_dim()const{
		return dim_;
	}
	
};

template<class Blas> class BlasMatrixTransponsed: public BlasVector<Blas>{
public:
	BlasMatrixTransponsed(unsigned int N):BlasVector<Blas>(N*N){}

	typename Blas::FloatType& operator()(unsigned int i,unsigned int j){
		return BlasVector<Blas>::data_[j*BlasVector<Blas>::N+i];
	}

};


template<class RealT,class Blas> class SimpleBlasDeque{
	std::deque<RealT*> data_;
	unsigned int sz_;
	unsigned int noccupied_;

	void clone_and_push(unsigned int N,  RealT* t){
		RealT * clone_data;
		Blas::allocate(N,clone_data);
		Blas::copy(N,t,clone_data);
		data_.push_back(clone_data);
	}

public:
	SimpleBlasDeque(unsigned int sz):sz_(0),noccupied_(0){
		//data_.resize(sz);
		sz_ = sz;
	}

	void push(unsigned int N,  RealT* t){
		if(noccupied_<sz_){
			clone_and_push(N,t);
			noccupied_ ++;
			return;
		}
		Blas::deallocate(data_.front());
		data_.pop_front();
		clone_and_push(N,t);
	}

	RealT* last()const{
		return data_[noccupied_-1];
	}

	RealT* operator [](unsigned int n){
		//assert(n<noccupied_);
		return data_[n];
	}

	~SimpleBlasDeque(){
		for (unsigned int i=0;i<data_.size();++i){
			Blas::deallocate(data_[i]);
		}
	}


};



template<class Blas> class VectorDeque{
	typedef typename Blas::FloatType RealT;
	BlasVector<Blas> data_;
	unsigned int vector_length_;
	unsigned int vector_number_;

	unsigned int begin_;
	unsigned int end_;

	unsigned int occupied_items_;

	unsigned int total_sz()const{
		return vector_number_ * vector_length_;
	}

	RealT* get_nth(unsigned int n){
		unsigned int pos = n * vector_length_;
		//assert(n<total_sz())
		return &data_[pos];
	} 

	void push_vector(unsigned int pos,RealT *v){
		//memcpy(&data_[pos],v,vector_length_*sizeof(RealT));
		const unsigned int sz = vector_length_;
		Blas::copy(sz,v,&data_[pos]);
	}

	unsigned int add_with_wrap(unsigned int n,unsigned int m) const{
		if(n+m>=total_sz()){
			return n+m - total_sz();
		} 
		else return n+m;
	}

	unsigned int add_with_wrap_e(unsigned int n,unsigned int m) const{
		if(n+m>total_sz()){
			return n+m - total_sz();
		} 
		else return n+m;
	}


public:
	VectorDeque(unsigned int vector_length, unsigned int max_sz):
	  vector_length_(vector_length),vector_number_(max_sz),occupied_items_(0)
	  {
		  data_.reset(total_sz());
		 // std::fill(( RealT*)data_,data_+total_sz(),RealT(0));
		  begin_ = 0;
		  end_ = 0;
	  }

	  void push(RealT* data){
		  if(occupied_items_*vector_length_<total_sz()){
			push_vector(end_,data);
			end_ = end_ + vector_length_;
			occupied_items_++;
		  }
		  else{
			  end_ = add_with_wrap_e(end_,vector_length_);
			  begin_ = add_with_wrap(begin_,vector_length_);
			  push_vector(end_-vector_length_,data);
		  }
	  }

	  unsigned int get_N()const{
		return vector_length_;
	  }

	  unsigned int get_capacity()const{
		  return vector_number_;
	  }

	  unsigned int occupied_items()const{
		  return occupied_items_;
	  }

	  const RealT* get_vector(unsigned int n)const{
		  return &data_[add_with_wrap(begin_,n*vector_length_)];
	  }

	  RealT* get_vector_pointer(unsigned int n){
		  return &data_[add_with_wrap(begin_,n*vector_length_)];
	  }

	  RealT* last() const{
		  return const_cast<RealT*>(&data_[end_-vector_length_]);
	  }

	  const RealT* last(unsigned int n) const{
		  return get_vector(get_capacity()-n);
	  }

};


template<class Blas> class VectorArray{
	typedef typename Blas::FloatType RealT;
	BlasVector<Blas> data_;

	unsigned int vector_number_,vector_length_,size_;
	unsigned int total_sz()const{
		return vector_number_ * vector_length_;
	}
	void push_vector(unsigned int pos, const RealT *v){
		const unsigned int sz = vector_length_;
		Blas::copy(sz,v,&data_[pos]);
	}
	unsigned int get_address_of_vector(unsigned int n) const{
		return vector_length_*n;
	}
public:
	VectorArray(unsigned int vector_len, unsigned int capacity=0):
	  vector_number_(capacity),vector_length_(vector_len)
	{
		reset(vector_len,capacity);	
	}

	VectorArray():vector_number_(0),vector_length_(0){

	}

	void reset(unsigned int vector_len, unsigned int capacity=0){
		vector_length_ = vector_len;
		vector_number_ = capacity;
		size_ = 0;
		data_.reset(total_sz());
	}

	RealT* get_vector_pointer(unsigned int n){
		const unsigned int p = get_address_of_vector(n);
		return &data_[p];
	}

	const RealT* get_vector(unsigned int n)const{
		const unsigned int p = get_address_of_vector(n);
		return &data_[p];
	}

	void push(const RealT* v){
		if(size_>=vector_number_)
			throw std::exception("overflow");
		const unsigned int p = get_address_of_vector(size_);
		push_vector(p,v);
		size_++;
	}

	void set_at(unsigned int j,const RealT* v){
		const unsigned int p = get_address_of_vector(j);
		push_vector(p,v);
	}

	void clear(){
		reset(vector_length_,vector_number_);
	}

	unsigned int occupied_items()const{
		return size_;
	}


};
