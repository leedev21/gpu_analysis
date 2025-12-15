
#ifndef OP_TRACE_H_
#define OP_TRACE_H_

#include <string>
#include <vector>
#include <memory>
#include <iostream>

namespace OPTRACE {

enum class datatype {
  F32 = 0,
  F64,
  C64,
  C128,
  I64,
  U64,
  BOOL,
  F16,
  BF16,             // not support, map to U16
  F8E4M3,           // not support, map to U16
  F8E5M2,           // not support, map to U16
  U32,
  I32,
  U16,
  I16,
  U8,
  I8,
  B8,               // bool instead of pred
  STR,              // string
  CUSTOM_DATA_TYPE  // customer type
};

enum class valuetype {
  TENSOR = 0,
  SCALAR,
  STRUCT
};

class value {
public:
  value();
  virtual ~value();
  virtual std::unique_ptr<value> clone() {
    std::cout << "class value do not clone implementation" << std::endl;
    return nullptr;
  }
  virtual valuetype getValueType() = 0;
  virtual std::string getString() = 0;
};

class scalar : public value
{
public:
  scalar(int64_t tensorID, std::string name, datatype dtype, std::string data, std::string customerType = "");
  scalar(int64_t tensorID, datatype dtype, std::string customerType = ""); //tensor id + type
  scalar(std::string name, datatype dtype, std::string data, std::string customerType = ""); // name + type + value
  scalar(datatype dtype, std::string data, std::string customerType = ""); // type + value
  scalar(const scalar& rhs);
  scalar& operator=(const scalar& rhs);
  virtual ~scalar() {
  }
  std::unique_ptr<value> clone() override;
  valuetype getValueType() override;
  std::string getString() override;

  int64_t getTensorID();
  std::string getName();
  datatype getDataType();
  std::string getCustomerType();
  std::string getValue();

  float getDataF32();
  uint32_t getDataU32();
  int32_t getDataI32();
  uint16_t getDataU16();
  int16_t getDataI16();
  uint8_t getDataU8();
  int8_t getDataI8();
  bool getDataPred();

private:
  int64_t     tensorID_;  // optional, tensor ID >= 0, -1 for non 
  datatype    dtype_;
  std::string customerType_; //optional customer type
  std::string strVal_;    // optional
  std::string name_;      // optional

  //double      dVal_; // optional
  //int64_t     iVal_; // optional
};

class tensor : public value
{
public:
  tensor(int64_t tensorID, datatype dtype,
    std::vector<int64_t>& dims, std::vector<int64_t> strides = {},
    int64_t offset = 0, std::string name = "");
  tensor() {};
  tensor(const tensor& rhs);
  tensor& operator=(const tensor& rhs);
  virtual ~tensor() {
  }
  std::unique_ptr<value> clone() override;
  valuetype getValueType() override;
  std::string getString() override;

  std::string getName();
  std::string getParamName();
  int64_t getTensorID();
  int64_t getOffset();
  datatype getDataType();

  int64_t getRank();
  std::vector<int64_t> getDims();
  int64_t getDim(size_t index);
  std::vector<int64_t> getStrides();
  int64_t getStride(size_t index);

private:
  std::string           name_;
  int64_t               tensorID_;
  datatype              dtype_;
  std::vector<int64_t>  dims_;
  std::vector<int64_t>  strides_;
  int64_t               offset_;
};

class structure : public value
{
public:
  // construct for c++
  structure(int64_t tensorID, std::string name, std::unique_ptr<std::vector<std::unique_ptr<value>>> data);
  structure(std::string paramName, int64_t tensorID, std::string name, std::unique_ptr<std::vector<std::unique_ptr<value>>> data);
  structure(std::string tensorIDName, std::string name, std::unique_ptr<std::vector<std::unique_ptr<value>>> data);
  structure(std::string paramName, std::string tensorIDName, std::string name, std::unique_ptr<std::vector<std::unique_ptr<value>>> data);
  structure(const structure& rhs);
  structure& operator=(const structure& rhs);
  
  // construct for pybind
  structure();
  structure(int64_t tensorID, std::string name, std::vector<value*> data);
  structure(std::string paramName, int64_t tensorID, std::string name, std::vector<value*> data);
  structure(std::string tensorIDName, std::string name, std::vector<value*> data);
  structure(std::string paramName, std::string tensorIDName, std::string name, std::vector<value*> data);

  virtual ~structure() {
  }
  std::unique_ptr<value> clone() override;
  valuetype getValueType() override;
  std::string getString() override;

  const std::string& getTensorIDName();
  int64_t getTensorID();
  const std::string& getName();
  const std::string& getParamName();
  const std::vector<std::unique_ptr<value>>& getData() const;

private:
  std::string tensorIDName_;  // optional, recored the name of "embed_tuple" e.g. embed_tuple:tuple{a=1.0:f32, b=32:i16}
  int64_t tensorID_;          // optional, recored the tensor ID of "%6", e.g. %6:tuple{%7:<32x16xf16>, %8:<32x16xf16>, count=2:i32}
  std::string name_;          // recored the structure name of "tuple", reference sample of tensorID_
  std::string paramName_;     // python input parameter name of "generator", e.g. generator=%2:tuple{seed=123:i32, offset=None:NoneType}
  std::unique_ptr<std::vector<std::unique_ptr<value>>> data_;
};


class instruct
{
public:
  // construct for c++
  instruct(std::string& processID, int64_t rankID, int64_t streamID,
    std::string &opname, std::string &domain, std::string& version,
    std::unique_ptr<std::vector<std::unique_ptr<value>>> operands,
    std::unique_ptr<std::vector<std::unique_ptr<value>>> results);
  instruct(const instruct& rhs);
  instruct& operator=(const instruct& rhs);
  // construct for pybind
  instruct();
  instruct(std::string& processID, int64_t rankID, int64_t streamID,
    std::string& opname, std::string& domain, std::string& version,
    std::vector<value*> operands,
    std::vector<value*> results);
  std::string getString();
  ~instruct();

public:
  const std::string& getProcessID();
  int64_t getRankID();
  int64_t getStreamID();

  const std::string& getOpname();
  const std::string& getDomain();
  const std::string& getVersion();

  size_t getOperandSize();
  size_t getResultSize();

  const std::vector<std::unique_ptr<value>>& getOperands() const;
  const std::vector<std::unique_ptr<value>>& getResults() const;

  const value* getOperandItem(size_t index) const;
  const value* getResultItem(size_t index) const;

private:
  std::string processID_;
  int64_t rankID_;
  int64_t streamID_;
  std::string opname_;
  std::string domain_;
  std::string version_;
  std::unique_ptr<std::vector<std::unique_ptr<value>>> operands_;
  std::unique_ptr<std::vector<std::unique_ptr<value>>> results_;
};

class module
{
public:
  // constract for c++
  module(std::unique_ptr<std::vector<std::unique_ptr<instruct>>> instructs);
  module(const module& rhs);
  module& operator=(const module& rhs);
  

  // construct for pybind
  module();
  module(std::vector<instruct*> instructs);
  ~module();

  std::string getString();

  const std::vector<std::unique_ptr<instruct>>& getInstructs() const;
  friend std::ostream& operator<<(std::ostream& os, const module& m);

private:
  // data convert fo pybind
  static std::unique_ptr<std::vector<std::unique_ptr<instruct>>> convertToUniquePtrVector(const std::vector<instruct*>& data) {
    auto uniquePtrVector = std::make_unique<std::vector<std::unique_ptr<instruct>>>();
    for (auto& ptr : data) {
      uniquePtrVector->emplace_back(ptr);
    }
    return uniquePtrVector;
  }

private:
  std::unique_ptr<std::vector<std::unique_ptr<instruct>>> instructs_;
};


class optrace
{
public:
  optrace();
  // data read from file
  optrace(std::string filename);
  // data read from string
  optrace(const char* data, int size);
  ~optrace();
  // data read from istream
  friend std::istream& operator>>(std::istream& is, optrace& t);

  std::shared_ptr<module> getModule();
private:
  std::shared_ptr<module> module_;
  std::string             filename_;
  const char            * data_;
  int                     size_;
};

}

#endif
