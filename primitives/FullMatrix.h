#pragma once

template<class RealT> class FullMatrix{
public:
	typedef std::map<unsigned int,RealT> Row;
	typedef std::vector<Row> MatrData;
private:
	MatrData data_;

	//unsigned int nrows_,ncolumns;
public:
	void set_value(unsigned int nrow,unsigned int ncolumn, RealT val){
		if(data_.size()<=nrow)
			data_.resize(nrow+1);
		/*if(data_[nrow].size()<=ncolumn)
			data_[nrow].resize(ncolumn+1);*/
		data_[nrow][ncolumn] = val;
	}

	RealT get_value(unsigned int x, unsigned int y){
		return data_[x].find(y)->second;
	}

	void reserve(unsigned int sz){
		data_.resize(sz);
		//std::for_each(data_.begin(),data_.end(),boost::bind(&Row::resize,_1,sz));
	}

	const Row& row(unsigned int idx) const {
		return data_[idx];
	}
	unsigned int rows()const{
		return data_.size();
	}
	
};

template<class RealT> class CNCLoader{

	template<class Fun> static bool parse_a(char* buf, Fun &cb){
		char* c = buf;
		char* cn = buf;
		cn = strchr(c,' ');
		if(cn==0) return false;
		*cn = 0;
		unsigned int m = atoi(c);
		c = cn + 1;

		cn = strchr(c,' ');
		if(cn==0) return false;
		*cn = 0;
		unsigned int n = atoi(c);
		c = cn + 1;

		RealT a = static_cast<RealT>(atof(c));
		cb(m,n,a);
		return true;
	}

	template<class Fun> static bool parse_b(char* buf, Fun &cb){
		char* c = buf;
		char* cn = buf;
		cn = strchr(c,' ');
		if(cn==0) return false;
		*cn = 0;
		unsigned int m = atoi(c);
		c = cn + 1;
		RealT a = static_cast<RealT>(atof(c));
		cb(m,a);
		return true;
	}

	template<class Funa,class Funb> static bool parse(char* buf, Funa &cba,Funb &cbb){
		char* c = buf;
		char* cn = buf;
		cn = strchr(c,' ');
		if(cn==0) return false;
		*cn = 0;
		if(strcmp(c,"a")==0){
			return parse_a(cn+1,cba);
		}
		if(strcmp(c,"b")==0){
			return parse_b(cn+1,cbb);
			//return true;
		}
		return false;
		
	}
public:
	template<class Fun,class Funb,class Fun2> static bool load(const wchar_t* file_name, Fun& cba,Funb& cbb,Fun2& cb2){
		FILE* f = _wfopen(file_name,L"r");
		if(!f) return false;
		unsigned int nline = 0;
		while (!feof(f))
		{
			char buf[128] = {};
			fgets(buf,128,f);
			if (nline>1){
				parse(buf,cba,cbb);
			}
			else if(nline==0) {
				cb2(atoi(buf));
			}
			nline++;
		}
		return true;
	}
};