#include "optrace.h"
#include <iostream>
#include <sstream>
#include <cassert>
#include <fstream>
#include <map>
#include <memory>
#include <cstring>
#include <iomanip>

namespace OPTRACE {

#define OPTCOUT std::cout << std::setw(5) << __LINE__ << ":" << std::setw(6) << lexer.getLine() << ":" << std::setw(4) << lexer.getCol()

/// Structure definition a location in a file.
struct Location {
  std::shared_ptr<std::string> file; ///< filename.
  int line;                          ///< line number.
  int col;                           ///< column number.
};

enum Token : int {
  tok_semicolon = ';',
  tok_comma = ',',
  tok_colon = ':',
  tok_cross = 'x',
  tok_star = '*',
  tok_equal = '=',
  tok_percent = '%',
  tok_parenthese_open = '(',
  tok_parenthese_close = ')',
  tok_bracket_open = '{',
  tok_bracket_close = '}',
  tok_sbracket_open = '[',
  tok_sbracket_close = ']',
  tok_abracket_open = '<',
  tok_abracket_close = '>',
  tok_underscore = '_',

  tok_newline = '\n',
  tok_eof = -1,
  // commands
  tok_return = -2,
  tok_var = -3,
  tok_def = -4,
  // primary
  tok_identifier = -5,
  tok_number = -6,
  tok_right_arrow = -7,
};


class Numeric {
public:
  Numeric() { ; }
  Numeric(const char* data)
    : data_(data) {}
  Numeric(const std::string data)
    : data_(data) {}
  Numeric(const Numeric& rhs) {
    this->data_ = rhs.data_;
    this->type_ = rhs.type_;
  }
  void settype(std::string type) { type_ = type; } //set customer type by string
  const std::string& gettype() { return type_; }

  void setdata(const char* data) { data_ = data; }
  void setdata(const std::string& data) { data_ = data; }
  void paddata(const char* data) { data_ += data; }
  void paddata(const std::string data) { data_ += data; }
  const std::string& getdata() { return data_; }

  Numeric& operator=(const Numeric& rhs) {
    this->data_ = rhs.data_;
    this->type_ = rhs.type_;
    return *this;
  }

  int toi() { return std::stoi(data_); }
  long tol() { return std::stol(data_); }
  long long toll() { return std::stoll(data_); }
  unsigned long toul() { return std::stoul(data_); }
  unsigned long long toull() { return std::stoull(data_); }
  float tof() { return std::stof(data_); }
  double tod() { return std::stod(data_); }
  bool tob() {
    if (data_ == "true")
      return true;
    else
      return false;
  }

  template<typename T>
  T tonum() {
    return 0;
  }

#if defined (_WIN32) || defined (_WIN64)
  template<>
  int8_t tonum<int8_t>() {
    return static_cast<int8_t>(tol());
  }
  template<>
  uint8_t tonum<uint8_t>() {
    return static_cast<uint8_t>(toul());
  }
  template<>
  int16_t tonum<int16_t>() {
    return static_cast<int16_t>(tol());
  }
  template<>
  uint16_t tonum<uint16_t>() {
    return static_cast<uint16_t>(toul());
  }
  template<>
  long tonum<long>() {
    return tol();
  }
  template<>
  unsigned long tonum<unsigned long>() {
    return toul();
  }
  template<>
  int tonum<int>() {
    return static_cast<int>(tol());
  }
  template<>
  unsigned int tonum<unsigned int>() {
    return static_cast<unsigned int>(toul());
  }
  template<>
  long long tonum<long long>() {
    return toll();
  }
  template<>
  unsigned long long tonum<unsigned long long>() {
    return toull();
  }
  template<>
  float tonum<float>() {
    return tof();
  }
  template<>
  double tonum<double>() {
    return tod();
  }
  template<>
  bool tonum<bool>() {
    return tob();
  }
#endif

private:
  std::string data_;
  std::string type_;
};

#if !(defined(_WIN32) || defined (_WIN64))
template<>
inline int8_t Numeric::tonum<int8_t>() {
  return static_cast<int8_t>(tol());
}
template<>
inline uint8_t Numeric::tonum<uint8_t>() {
  return static_cast<uint8_t>(toul());
}
template<>
inline int16_t Numeric::tonum<int16_t>() {
  return static_cast<int16_t>(tol());
}
template<>
inline uint16_t Numeric::tonum<uint16_t>() {
  return static_cast<uint16_t>(toul());
}
template<>
inline long Numeric::tonum<long>() {
  return tol();
}
template<>
inline unsigned long Numeric::tonum<unsigned long>() {
  return toul();
}
template<>
inline int Numeric::tonum<int>() {
  return static_cast<int>(tol());
}
template<>
inline unsigned int Numeric::tonum<unsigned int>() {
  return static_cast<unsigned int>(toul());
}
template<>
inline long long Numeric::tonum<long long>() {
  return toll();
}
template<>
inline unsigned long long Numeric::tonum<unsigned long long>() {
  return toull();
}
template<>
inline float Numeric::tonum<float>() {
  return tof();
}
template<>
inline double Numeric::tonum<double>() {
  return tod();
}
template<>
inline bool Numeric::tonum<bool>() {
  return tob();
}
#endif

std::map<std::string, datatype> string2datatype = {
  std::make_pair("f64", datatype::F64),
  std::make_pair("c64", datatype::C64),
  std::make_pair("c128", datatype::C128),
  std::make_pair("i64", datatype::I64),
  std::make_pair("u64", datatype::U64),
  std::make_pair("bool", datatype::BOOL),
  std::make_pair("f32", datatype::F32),
  std::make_pair("f16", datatype::F16),
  std::make_pair("bf16", datatype::BF16),
  std::make_pair("f8e4m3", datatype::F8E4M3),
  std::make_pair("f8e5m2", datatype::F8E5M2),
  std::make_pair("u32", datatype::U32),
  std::make_pair("i32", datatype::I32),
  std::make_pair("u16", datatype::U16),
  std::make_pair("i16", datatype::I16),
  std::make_pair("u8", datatype::U8),
  std::make_pair("i8", datatype::I8),
  std::make_pair("b8", datatype::B8),
  std::make_pair("str", datatype::STR),
  std::make_pair("CUSTOM_DATA_TYPE", datatype::CUSTOM_DATA_TYPE)
};

std::map<datatype, std::string> datatype2string = {
  std::make_pair(datatype::F64, "f64"),
  std::make_pair(datatype::C64, "c64"),
  std::make_pair(datatype::C128, "c128"),
  std::make_pair(datatype::I64, "i64"),
  std::make_pair(datatype::U64, "u64"),
  std::make_pair(datatype::BOOL, "bool"),
  std::make_pair(datatype::F32, "f32"),
  std::make_pair(datatype::F16, "f16"),
  std::make_pair(datatype::BF16, "bf16"),
  std::make_pair(datatype::F8E4M3,"f8e4m3"),
  std::make_pair(datatype::F8E5M2, "f8e5m2"),
  std::make_pair(datatype::U32, "u32"),
  std::make_pair(datatype::I32, "i32"),
  std::make_pair(datatype::U16, "u16"),
  std::make_pair(datatype::I16, "i16"),
  std::make_pair(datatype::U8, "u8"),
  std::make_pair(datatype::I8, "i8"),
  std::make_pair(datatype::B8, "b8"),
  std::make_pair(datatype::STR, "str"),
  std::make_pair(datatype::CUSTOM_DATA_TYPE, "CUSTOM_DATA_TYPE")
};

class Lexer {
public:
  Lexer(std::string filename)
    : lastLocation(
      { std::make_shared<std::string>(std::move(filename)), 0, 0 }) {}
  virtual ~Lexer() = default;

  /// get to next line when parse failure
  void gotoNextLine() {
    curLineBuffer = readNextLine();
    ++curLineNum;
    curCol = 0;
  }

  /// Look at the current token in the stream.
  Token getCurToken() { return curTok; }

  /// Move to the next token in the stream and return it.
  Token getNextToken() { return curTok = getTok(); }

  /// Move to the next token in the stream, asserting on the current token
  /// matching the expectation.
  void consume(Token tok) {
    assert(tok == curTok && "consume Token mismatch expectation");
    getNextToken();
  }

  /// Return the current identifier (prereq: getCurToken() == tok_identifier)
  const std::string& getId() {
    assert(curTok == tok_identifier);
    return identifierStr;
  }

  /// Return the current number (prereq: getCurToken() == tok_number)
  double getValue() {
    assert(curTok == tok_number);
    return numVal;
  }

  Numeric& getNumber() { return number_; }

  /// Return the location for the beginning of the current token.
  Location getLastLocation() { return lastLocation; }

  bool findColonAbracket() {
    int index = curCol;
    bool colon = false;
    while (true) {
      if (index == static_cast<int>(curLineBuffer.size()))
        return false;
      if (curLineBuffer[index] == ','
        || curLineBuffer[index] == '}'
        || curLineBuffer[index] == ')')
        return false;
      if (colon == false) {
        if (curLineBuffer[index] == ':')
          colon = true;
      }
      else {
        if (curLineBuffer[index] == '<')
          return true;
        if (curLineBuffer[index] != ' ')
          return false;
      }
      index++;
    }
  }

  bool findIdentifierBracket() {
    int index = curCol;
    //bool colon = false;
    bool identifier = false;
    while (true) {
      if (index == static_cast<int>(curLineBuffer.size()))
        return false;
      if (curLineBuffer[index] == ','
        || curLineBuffer[index] == '}'
        || curLineBuffer[index] == ')'
        || curLineBuffer[index] == '=')
        return false;
      
      //if (curLineBuffer[index] == ':' && colon == false) {
      //  colon = true;
      //  index++;
      //  continue;
      //}

      if (isalpha(curLineBuffer[index]) && identifier == false) {
        identifier = true;
        index++;
        continue;
      }

      if (curLineBuffer[index] == '{') {
        if (/*colon == true && */identifier == true)
          return true;
        else
          return false;
      }

      index++;
    }
  }

  // Return the current line in the file.
  int getLine() { return curLineNum; }

  // Return the current column in the file.
  int getCol() { return curCol; }

private:
  /// Delegate to a derived class fetching the next line. Returns an empty
  /// string to signal end of file (EOF). Lines are expected to always finish
  /// with "\n"
  virtual std::string readNextLine() = 0;

  int getNextChar() {
    if (curLineBuffer.empty())
      return EOF;
    ++curCol;
    auto nextchar = curLineBuffer[curCol-1];
    if (curCol == static_cast<int>(curLineBuffer.size())) {
      curLineBuffer = readNextLine();
      ++curLineNum;
      curCol = 0;
    }
    return nextchar;
  }

  ///  Return the next token from standard input.
  Token getTok() {
    // Skip any whitespace.
    while (isspace(lastChar))
      lastChar = Token(getNextChar());

    // Identifier: [_a-zA-Z][a-zA-Z0-9_]*
    if (isalpha(lastChar) || lastChar == '_') {
      if (lastChar == 'x') {
        // the first 'x'
        Token thisChar = Token(lastChar);
        lastChar = Token(getNextChar());
        return thisChar;
      }
      identifierStr = (char)lastChar;
      // stop the end of 'x'
      while ((isalnum((lastChar = Token(getNextChar()))) || lastChar == '_') && lastChar != 'x')
        identifierStr += (char)lastChar;
      return tok_identifier;
    }

    // Number: [0-9.]+
    if (isdigit(lastChar) || lastChar == '.') {
      std::string numStr;
      do {
        numStr += lastChar;
        lastChar = Token(getNextChar());
      } while (isdigit(lastChar) || lastChar == '.' || lastChar == 'e' || lastChar == 'E' || lastChar == '-' || lastChar == '+');

      numVal = strtod(numStr.c_str(), nullptr);
      this->number_.setdata(numStr);
      return tok_number;
    }

    if (lastChar == '-') {
      std::string idfStr;

      // negative Number: -[0-9.]+
      idfStr = (char)lastChar;
      lastChar = Token(getNextChar());

      while (isspace(lastChar))
        lastChar = Token(getNextChar());

      if (lastChar == '>') { // matching right arrow "->"
        lastChar = Token(getNextChar());
        return tok_right_arrow;
      }

      // Identifier: [a-zA-Z][a-zA-Z0-9_]*
      if (isalpha(lastChar)) {
        idfStr += (char)lastChar;
        while ((isalnum((lastChar = Token(getNextChar()))) || lastChar == '_'))
          idfStr += (char)lastChar;
        identifierStr = idfStr;
        return tok_identifier;
      }

      if (isdigit(lastChar)) {
        do {
          idfStr += lastChar;
          lastChar = Token(getNextChar());
        } while (isdigit(lastChar) || lastChar == '.' || lastChar == 'e' || lastChar == 'E' || lastChar == '-' || lastChar == '+');

        numVal = strtod(idfStr.c_str(), nullptr);
        this->number_.setdata(idfStr);
        return tok_number;
      }
    }

    if (lastChar == '#') {
      // Comment until end of line.
      do {
        lastChar = Token(getNextChar());
      } while (lastChar != EOF && lastChar != '\n' && lastChar != '\r');

      if (lastChar != EOF)
        return getTok();
    }

    // Check for end of file.  Don't eat the EOF.
    if (lastChar == EOF)
      return tok_eof;

    // Otherwise, just return the character as its ascii value.
    Token thisChar = Token(lastChar);
    lastChar = Token(getNextChar());
    return thisChar;
  }

  /// The last token read from the input.
  Token curTok = tok_eof;

  /// Location for `curTok`.
  Location lastLocation;

  /// If the current Token is an identifier, this string contains the value.
  std::string identifierStr;

  /// If the current Token is a number, this contains the value.
  double numVal = 0;
  Numeric number_;

  /// The last value returned by getNextChar(). We need to keep it around as we
  /// always need to read ahead one character to decide when to end a token and
  /// we can't put it back in the stream after reading from it.
  Token lastChar = Token(' ');

  /// Keep track of the current line number in the input stream
  int curLineNum = 0;

  /// Keep track of the current column number in the input stream
  int curCol = 0;

  /// Buffer supplied by the derived class on calls to `readNextLine()`
  std::string curLineBuffer = "\n";
};

/// A lexer implementation operating on a buffer in memory.
class LexerBuffer final : public Lexer {
public:
  LexerBuffer(const char* begin, const char* end, std::string filename)
    : Lexer(std::move(filename)), current(begin), end(end) {}

private:
  /// Provide one line at a time to the Lexer
  std::string readNextLine() override  {
    //const char* begin = current;
    std::string data;
    while (current <= end && *current && *current != '\n') {
      data.push_back(*current);
      current++;
    }
    if (current <= end && *current) {
      ++current;
    }
    return data;
  }
  const char* current, * end;
};

class LexerFile final : public Lexer {
public:
  LexerFile(std::string filename)
    : Lexer(filename) {
    file_.open(filename);
  }
  ~LexerFile() {
    if (file_.is_open())
      file_.close();
  }

private:
  std::string readNextLine() override {
    if (!file_.is_open() || file_.eof() || file_.fail()) {
      return "";
    }
    std::string line;
    std::getline(file_, line);
    return line;
  }

private:
  std::ifstream file_;
};

value::value() {
  //std::cout << std::hex << __FUNCTION__ << " addr= " << this << std::endl;
}

value::~value() {
  //std::cout << std::hex << __FUNCTION__ << " addr= " << this << std::endl;
}


scalar::scalar(int64_t tensorID, std::string name, datatype dtype, std::string val, std::string customerType) :
  tensorID_(tensorID), dtype_(dtype), strVal_(val), name_(name), customerType_(customerType) {
}
scalar::scalar(int64_t tensorID, datatype dtype, std::string customerType)
  : tensorID_(tensorID), dtype_(dtype), strVal_(""), name_(""), customerType_(customerType) {
}
scalar::scalar(std::string name, datatype dtype, std::string data, std::string customerType)
  : tensorID_(-1), dtype_(dtype), strVal_(data), name_(name), customerType_(customerType) {
}
scalar::scalar(datatype dtype, std::string data, std::string customerType)
  : tensorID_(-1), dtype_(dtype), strVal_(data), name_(""), customerType_(customerType) {
}

scalar::scalar(const scalar& rhs) {
  tensorID_ =rhs.tensorID_;
  dtype_ = rhs.dtype_;
  customerType_ = rhs.customerType_;
  strVal_ = rhs.strVal_;
  name_ = rhs.name_;
}
scalar& scalar::operator=(const scalar& rhs) {
  if (this == &rhs)
    return *this;
  tensorID_ = rhs.tensorID_;
  dtype_ = rhs.dtype_;
  customerType_ = rhs.customerType_;
  strVal_ = rhs.strVal_;
  name_ = rhs.name_;
  return *this;
}
std::unique_ptr<value> scalar::clone() {
  return std::make_unique<scalar>(tensorID_, name_, dtype_, strVal_, customerType_);
}
valuetype scalar::getValueType() {
  return valuetype::SCALAR;
}
std::string scalar::getString() {
  std::ostringstream oss;
  if (name_.size() > 0)
    oss << name_ << "=";
  if (tensorID_ >= 0)
    oss << "%" << tensorID_;
  if (strVal_.size() > 0)
    oss << strVal_;
  oss << ":";
  if (strVal_.size() == 0)
    oss << "<";
  if(dtype_ == datatype::CUSTOM_DATA_TYPE)
    oss << customerType_;
  else
    oss << datatype2string[dtype_];
  if (strVal_.size() == 0)
    oss << ">";
  return oss.str();
}

int64_t scalar::getTensorID() {
  return tensorID_;
}
std::string scalar::getName() {
  return name_;
}
datatype scalar::getDataType() {
  return dtype_;
}
std::string scalar::getCustomerType() {
  return customerType_;
}
std::string scalar::getValue() {
  return strVal_;
}

float scalar::getDataF32() {
  return std::stof(strVal_);
}
uint32_t scalar::getDataU32() {
  return static_cast<unsigned int>(std::stoul(strVal_));
}
int32_t scalar::getDataI32() {
  return std::stoi(strVal_);
}
uint16_t scalar::getDataU16() {
  return static_cast<uint16_t>(std::stol(strVal_));
}
int16_t scalar::getDataI16() {
  return static_cast<int16_t>(std::stol(strVal_));
}
uint8_t scalar::getDataU8() {
  return static_cast<uint8_t>(std::stol(strVal_));
}
int8_t scalar::getDataI8() {
  return static_cast<int8_t>(std::stol(strVal_));
}
bool scalar::getDataPred() {
  if (strVal_ == "0" || strVal_ == "false")
    return false;
  else
    return true;
}


tensor::tensor(int64_t tensorID, datatype dtype,
  std::vector<int64_t>& dims, std::vector<int64_t> strides,
  int64_t offset, std::string name) :
  tensorID_(tensorID), dtype_(dtype),
  dims_(std::move(dims)), strides_(std::move(strides)),
  offset_(offset), name_(name) {
}
tensor::tensor(const tensor& rhs) {
  name_ = name_;
  tensorID_ = rhs.tensorID_;
  dtype_ = rhs.dtype_;
  dims_ = rhs.dims_;
  strides_ = rhs.strides_;
  offset_ = rhs.offset_;
}
tensor& tensor::operator=(const tensor& rhs) {
  if (this == &rhs)
    return *this;
  name_ = name_;
  tensorID_ = rhs.tensorID_;
  dtype_ = rhs.dtype_;
  dims_ = rhs.dims_;
  strides_ = rhs.strides_;
  offset_ = rhs.offset_;
  return *this;
}
std::unique_ptr<value> tensor::clone() {
  std::unique_ptr<tensor> t = std::make_unique<tensor>();
  t->name_ = name_;
  t->tensorID_ = tensorID_;
  t->dtype_ = dtype_;
  t->dims_ = dims_;
  t->strides_ = strides_;
  t->offset_ = offset_;
  return std::move(t);
}
valuetype tensor::getValueType() {
  return valuetype::TENSOR;
}
std::string tensor::getString() {
  std::ostringstream oss;
  if (name_.size() > 0)
    oss << name_ << "=";
  oss << "%"
    << tensorID_ << ":"
    << "<" << dims_[0];
  for (size_t i = 1; i < dims_.size(); i++)
    oss << "x" << dims_[i];
  oss << "x" << datatype2string[dtype_] << ">";
  if (strides_.size() > 0) {
    oss << "{" << strides_[0];
    for (size_t i = 1; i < strides_.size(); i++)
      oss << ", " << strides_[i];
    oss << "}";
  }
  if (offset_ > 0)
    oss << "+" << offset_;

  return oss.str();
}
std::string tensor::getName() {
  return name_;
}
std::string tensor::getParamName() {
  return name_;
}
int64_t tensor::getTensorID() {
  return tensorID_;
}
int64_t tensor::getOffset() {
  return offset_;
}
datatype tensor::getDataType() {
  return dtype_;
}
int64_t tensor::getRank() {
  return static_cast<int64_t>(dims_.size());
}
std::vector<int64_t> tensor::getDims() {
  return dims_;
}
int64_t tensor::getDim(size_t index) {
  if (index < dims_.size())
    return dims_[index];
  else
    return -1;
}
std::vector<int64_t> tensor::getStrides() {
  return strides_;
}
int64_t tensor::getStride(size_t index) {
  if (index < strides_.size())
    return strides_[index];
  else
    return -1;
}

structure::structure(int64_t tensorID, std::string name, std::unique_ptr<std::vector<std::unique_ptr<value>>> data):
  tensorID_(tensorID), name_(std::move(name)), data_(std::move(data)) {
}
structure::structure(std::string paramName, int64_t tensorID, std::string name, std::unique_ptr<std::vector<std::unique_ptr<value>>> data) :
  paramName_(paramName), tensorID_(tensorID), name_(std::move(name)), data_(std::move(data)) {
}

structure::structure(std::string tensorIDName, std::string name, std::unique_ptr<std::vector<std::unique_ptr<value>>> data) :
  tensorIDName_(tensorIDName), tensorID_(-1), name_(std::move(name)), data_(std::move(data)) {
}
structure::structure(std::string paramName, std::string tensorIDName, std::string name, std::unique_ptr<std::vector<std::unique_ptr<value>>> data) :
  paramName_(paramName), tensorIDName_(tensorIDName), tensorID_(-1), name_(std::move(name)), data_(std::move(data)) {
}

structure::structure(const structure& rhs) {
  tensorIDName_ = rhs.tensorIDName_;
  tensorID_ = rhs.tensorID_;
  name_ = rhs.name_;
  paramName_ = rhs.paramName_;
  data_ = std::make_unique<std::vector<std::unique_ptr<value>>>();
  for (auto& iter : *(rhs.data_.get())) {
    data_->push_back(std::move(iter->clone()));
  }
}
structure& structure::operator=(const structure& rhs) {
  if (this == &rhs)
    return *this;
  tensorIDName_ = rhs.tensorIDName_;
  tensorID_ = rhs.tensorID_;
  name_ = rhs.name_;
  paramName_ = rhs.paramName_;
  data_ = std::make_unique<std::vector<std::unique_ptr<value>>>();
  for (auto& iter : *(rhs.data_.get())) {
    data_->push_back(std::move(iter->clone()));
  }
  return *this;
}

std::unique_ptr<value> structure::clone() {
  std::unique_ptr<structure> structure_new = std::make_unique<structure>();
  structure_new->tensorIDName_ = tensorIDName_;
  structure_new->tensorID_ = tensorID_;
  structure_new->paramName_ = paramName_;
  structure_new->name_ = name_;
  structure_new->data_ = std::make_unique<std::vector<std::unique_ptr<value>>>();
  for (auto& iter : *data_.get()) {
    structure_new->data_->push_back(std::move(iter->clone()));
  }
  return std::move(structure_new);
}

structure::structure() {
}
structure::structure(int64_t tensorID, std::string name, std::vector<value*> data):
  tensorID_(tensorID), name_(name) {
  data_ = std::make_unique<std::vector<std::unique_ptr<value>>>();
  for (auto& ptr : data) {
    data_->emplace_back(std::move(ptr->clone()));
  }
}
structure::structure(std::string paramName, int64_t tensorID, std::string name, std::vector<value*> data) :
  paramName_(paramName), tensorID_(tensorID), name_(name) {
  data_ = std::make_unique<std::vector<std::unique_ptr<value>>>();
  for (auto& ptr : data) {
    data_->emplace_back(std::move(ptr->clone()));
  }
}

structure::structure(std::string tensorIDName, std::string name, std::vector<value*> data):
  tensorIDName_(std::move(tensorIDName)), name_(name) {
  data_ = std::make_unique<std::vector<std::unique_ptr<value>>>();
  for (auto& ptr : data) {
    data_->emplace_back(std::move(ptr->clone()));
  }
}
structure::structure(std::string paramName, std::string tensorIDName, std::string name, std::vector<value*> data) :
  paramName_(paramName), tensorIDName_(std::move(tensorIDName)), name_(name) {
  data_ = std::make_unique<std::vector<std::unique_ptr<value>>>();
  for (auto& ptr : data) {
    data_->emplace_back(std::move(ptr->clone()));
  }
}

valuetype structure::getValueType() {
  return valuetype::STRUCT;
}
std::string structure::getString() {
  std::ostringstream oss;
  if (paramName_.size() > 0)
    oss << paramName_ << " = ";
  if (tensorID_ >= 0)
    oss << "%" << tensorID_ << ":";
  if (tensorIDName_.size() > 0)
    oss << tensorIDName_ << ":";
  if (name_.size() > 0)
    oss << name_;
  oss << "{";
  bool firstValue = true;
  for (auto& val : *data_) {
    if (firstValue == true)
      firstValue = false;
    else
      oss << ", ";
    oss << val->getString();
  }
  oss << "}";
  return oss.str();
}
const std::string& structure::getTensorIDName() {
  return tensorIDName_;
}
int64_t structure::getTensorID() {
  return tensorID_;
}
const std::string& structure::getName() {
  return name_;
}
const std::string& structure::getParamName() {
  return paramName_;
}
const std::vector<std::unique_ptr<value>>& structure::getData() const {
  return *(data_.get());
}

instruct::instruct(std::string& processID, int64_t rankID, int64_t streamID,
  std::string& opname, std::string& domain, std::string& version,
  std::unique_ptr<std::vector<std::unique_ptr<value>>> operands,
  std::unique_ptr<std::vector<std::unique_ptr<value>>> results):
  processID_(processID), rankID_(rankID), streamID_(streamID),
  opname_(opname), domain_(domain), version_(version),
  operands_(std::move(operands)), results_(std::move(results)) {
    //std::cout << std::hex << __FUNCTION__ << " new cpp addr= " << this << std::endl;
}
instruct::instruct() {
  //std::cout << std::hex << __FUNCTION__ << " new empty addr= " << this << std::endl;
}
instruct::instruct(std::string& processID, int64_t rankID, int64_t streamID,
  std::string& opname, std::string& domain, std::string& version,
  std::vector<value*> operands,
  std::vector<value*> results):
  processID_(processID), rankID_(rankID), streamID_(streamID),
  opname_(opname), domain_(domain), version_(version) {
  //std::cout << std::hex << __FUNCTION__ << " new python addr= " << this << std::endl;

  operands_ = std::make_unique<std::vector<std::unique_ptr<value>>>();
  for (auto& ptr : operands) {
    operands_->emplace_back(std::move(ptr->clone()));
  }

  results_ = std::make_unique<std::vector<std::unique_ptr<value>>>();
  for (auto& ptr : results) {
    results_->emplace_back(std::move(ptr->clone()));
  }
}
instruct::~instruct() {
  //std::cout << std::hex << __FUNCTION__ << " addr= " << this << std::endl;
}
instruct::instruct(const instruct& rhs) {
  //std::cout << std::hex << __FUNCTION__ << " copy addr= " << this << std::endl;
  processID_ = rhs.processID_;
  rankID_ = rhs.rankID_;
  streamID_ = rhs.streamID_;
  opname_ = rhs.opname_;
  domain_ = rhs.domain_;
  version_ = rhs.version_;
  operands_ = std::make_unique<std::vector<std::unique_ptr<value>>>();
  results_ = std::make_unique<std::vector<std::unique_ptr<value>>>();
  for (auto& iter : *rhs.operands_.get()) {
    operands_->push_back(std::move(iter->clone()));
  }
  for (auto& iter : *rhs.results_.get()) {
    results_->push_back(std::move(iter->clone()));
  }
}
instruct& instruct::operator=(const instruct& rhs) {
  if (this == &rhs)
    return *this;
  processID_ = rhs.processID_;
  rankID_ = rhs.rankID_;
  streamID_ = rhs.streamID_;
  opname_ = rhs.opname_;
  domain_ = rhs.domain_;
  version_ = rhs.version_;
  operands_ = std::make_unique<std::vector<std::unique_ptr<value>>>();
  results_ = std::make_unique<std::vector<std::unique_ptr<value>>>();
  for (auto& iter : *rhs.operands_.get()) {
    operands_->push_back(std::move(iter->clone()));
  }
  for (auto& iter : *rhs.results_.get()) {
    results_->push_back(std::move(iter->clone()));
  }
  return *this;
}

std::string instruct::getString() {
  std::ostringstream oss;
  if (processID_.size() > 0)
    oss << processID_ << ":";
  oss << rankID_ << ":" << streamID_;
  oss << " " << (*results_)[0]->getString();
  oss << " = ";
  if (domain_.size() > 0)
    oss << domain_ << ".";
  if (version_.size() > 0)
    oss << version_ << ".";
  oss << opname_;
  oss << "(";
  bool firstValue = true;
  for (auto& val : *operands_) {
    if (firstValue == true)
      firstValue = false;
    else
      oss << ", ";
    oss << val->getString();
  }
  oss << ")";
  return oss.str();
}

const std::string& instruct::getProcessID() {
  return processID_;
}
int64_t instruct::getRankID() {
  return rankID_;
}
int64_t instruct::getStreamID() {
  return streamID_;
}
const std::string& instruct::getOpname() {
  return opname_;
}
const std::string& instruct::getDomain() {
  return domain_;
}
const std::string& instruct::getVersion() {
  return version_;
}
size_t instruct::getOperandSize() {
  return operands_->size();
}
size_t instruct::getResultSize() {
  return results_->size();
}
const std::vector<std::unique_ptr<value>>& instruct::getOperands() const {
  return *(operands_.get());
}
const std::vector<std::unique_ptr<value>>& instruct::getResults() const {
  return *(results_.get());
}

const value* instruct::getOperandItem(size_t index) const {
  if (index < operands_->size())
    return (*operands_.get())[index].get();
  else
    return nullptr;
}
const value* instruct::getResultItem(size_t index) const {
  if (index < results_->size())
    return (*results_.get())[index].get();
  else
    return nullptr;
}


module::module(std::unique_ptr<std::vector<std::unique_ptr<instruct>>> instructs):
  instructs_(std::move(instructs)) {
}
module::module(const module& rhs) {
  instructs_ = std::make_unique<std::vector<std::unique_ptr<instruct>>>();
  for (auto& iter : *rhs.instructs_.get()) {
    instructs_->push_back(std::make_unique<instruct>(*iter.get()));
  }
}
module& module::operator=(const module& rhs) {
  if (this == &rhs)
    return *this;
  instructs_ = std::make_unique<std::vector<std::unique_ptr<instruct>>>();
  for (auto& iter : *rhs.instructs_.get()) {
    instructs_->push_back(std::make_unique<instruct>(*iter.get()));
  }
  return *this;
}

module::module() {
}
module::module(std::vector<instruct*> instructs):
  instructs_(convertToUniquePtrVector(instructs)) {
}

module::~module() {
  //std::cout << std::hex << __FUNCTION__ << " addr= " << this << std::endl;
}

std::string module::getString() {
  std::ostringstream oss;
  for (auto& ins : *instructs_)
    oss << ins->getString() << std::endl;
  return oss.str();
}

const std::vector<std::unique_ptr<instruct>>& module::getInstructs() const {
  return *(instructs_.get());
}

std::ostream& operator<<(std::ostream& os, const module& m) {
  (void)m;
  return os;
}

class Parser {
public:
  Parser(Lexer &lexer) : lexer(lexer) {}
  /// Parse a full module, a module is a list of instruct definitions.
  std::unique_ptr<module> getModule() {
    lexer.getNextToken();
    auto m = parseModule();
    if (m.get() != nullptr)
      return std::make_unique<module>(std::move(m));
    else {
      OPTCOUT << " failure parse module." << std::endl;
      return nullptr;
    }
  }
private:
  Lexer &lexer;

  bool parseDim(std::vector<int64_t>& dims, datatype &dtype) {
    Token t = lexer.getCurToken();
    while (t != tok_eof) {
      switch (t) {
      case '<':
        t = lexer.getNextToken();
        break;
      case '>':
        //t = lexer.getNextToken();
        return true;
        break;
      case tok_number:
      {
        dims.push_back(lexer.getNumber().tonum<int64_t>());
        t = lexer.getNextToken();
        break;
      }
      case tok_identifier:
      {
        // sample case <3x3xf32>, match "...xf32"
        auto iter = string2datatype.find(lexer.getId());
        if (iter != string2datatype.end())
          dtype = iter->second;
        else {
          OPTCOUT << " parseDim unknow identifier: " << lexer.getId() << std::endl;
          return false;
        }
        t = lexer.getNextToken();
        break;
      }
      case 'x':
      {
        t = lexer.getNextToken();
        break;
      }
      default:
        OPTCOUT << " parseDim invalid token of: " << char(t) << std::endl;
        return false;
        break;
      }
    }

    return false;
  }

  std::unique_ptr<value> parseTensor(int64_t tensorID = -1, std::string name = "") {
    datatype dtype = datatype::U16;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    int64_t offset = 0;

    Token t = lexer.getCurToken();
    while (t != tok_eof && t != '\n' && t != ';') {
      switch (t) {
      case '%':
      {
        t = lexer.getNextToken();
        if (t == tok_number) {
          tensorID = lexer.getNumber().tonum<int64_t>();
        }
        else {
          OPTCOUT << " failure parse tensor id, token = " << t << std::endl;
          return nullptr;
        }
        t = lexer.getNextToken();
        break;
      }
      case ':':
        t = lexer.getNextToken();
        break;
      case tok_identifier:
      {
        break;
      }
      case '<':
      {
        bool ret = parseDim(dims, dtype);
        if (ret != true) {
          OPTCOUT << " failure parse tensor dims" << std::endl;
          return nullptr;
        }
        t = lexer.getCurToken();
        break;
      }
      case '>':
      {
        t = lexer.getNextToken();
        if (!(t == '{' || t == '+')) {
          std::unique_ptr<value> v;
          if (dims.size() == 0
            || (dims.size() == 1 && dims[0] == 1)) {
            // scalar
            v = std::make_unique<scalar>(tensorID, "", dtype,  "");
          }
          else {
            // tensor
            v = std::make_unique<tensor>(tensorID, dtype, dims, strides, offset, name);
          }
          return std::move(v);
        }
        break;
      }
      case tok_number:
      {
        t = lexer.getNextToken();
        break;
      }
      case '+':
      {
        t = lexer.getNextToken();
        if (t == tok_number) {
          offset = lexer.getNumber().tonum<int64_t>();
          t = lexer.getNextToken();// move to the next token
          auto new_tensor = std::make_unique<tensor>(tensorID, dtype, dims, strides, offset, name);
          return std::move(new_tensor);
        }
        else {
          OPTCOUT << " failure parse 'offset' after +" << std::endl;
          return nullptr;
        }
        break;
      }
      case '{':
      {
        // parse strides
        t = lexer.getNextToken();
        bool parsing_stride = true;
        while (parsing_stride) {
          if (t == tok_number) {
            strides.push_back(lexer.getNumber().tonum<int64_t>());
          }
          else if (t == ',') {
            //
          }
          else if (t == '}') {
            parsing_stride = false;
            //t = lexer.getNextToken();
            //if (t != '+') {
            //  auto new_tensor = std::make_unique<tensor>(tensorID, dtype, dims, strides, offset, name);
            //  return std::move(new_tensor);
            //}
          }
          else {
            OPTCOUT << " failure stride, unknow token = " << t << std::endl;
            return nullptr;
          }
          Token last_t = t;
          t = lexer.getNextToken();
          if (last_t == '}' && t == '}') {
            auto v = std::make_unique<tensor>(tensorID, dtype, dims, strides, offset, name);
            return std::move(v);
          }
        }
        break;
      }
      case '}':
        t = lexer.getNextToken();
        break;
      case '=':
      {
        auto v = std::make_unique<tensor>(tensorID, dtype, dims, strides, offset, name);
        return std::move(v);
      }
      case ',':
      case ')':
      {
        t = lexer.getNextToken();
        auto new_tensor = std::make_unique<tensor>(tensorID, dtype, dims, strides, offset, name);
        return std::move(new_tensor);
        break;
      }
      default:
        OPTCOUT << " tensor parse read wrong token = " << t << std::endl;
        return nullptr;
      }
    }
    return nullptr;
  }

  std::unique_ptr<scalar> parseScalar(int64_t tensorID, std::string name) {
    datatype dtype;
    Numeric num;

    Token t = lexer.getCurToken();
    while (t != tok_eof && t != '\n' && t != ';') {
      switch (t) {
      case '=':
        t = lexer.getNextToken();
        break;
      case tok_number:
        num.paddata(lexer.getNumber().getdata());
        t = lexer.getNextToken();
        break;
      case ':':
      {
        t = lexer.getNextToken();
        if (t != tok_identifier) {
          OPTCOUT << " failure parse data type after ':', token = " << t << std::endl;
          return nullptr;
        }
        auto iter = string2datatype.find(lexer.getId());
        if (iter != string2datatype.end()) {
          dtype = iter->second;
        }
        else {
          dtype = datatype::CUSTOM_DATA_TYPE;
          num.settype(lexer.getId());
        }
        
        t = lexer.getNextToken(); // move the next token after data type
        auto new_scalar = std::make_unique<scalar>(tensorID, name, dtype, num.getdata(), num.gettype());
        return std::move(new_scalar);
        break;
      }
      case ',':
        t = lexer.getNextToken();
        break;
      case tok_identifier:
      {
        num.paddata(lexer.getId());
        t = lexer.getNextToken();
        break;
      }
      default:
      {
        OPTCOUT << " wrong token = " << t << std::endl;
        t = lexer.getNextToken();
        return nullptr;
      }
      }
    }
    return nullptr;
  }

  std::unique_ptr<structure> parseStructure(std::string tensorIDName, int64_t tensorID, std::string name, std::string paramName = "") {
    std::unique_ptr<std::vector<std::unique_ptr<value>>> data
      = std::make_unique<std::vector<std::unique_ptr<value>>>();

    Token t = lexer.getCurToken();
    while (t != tok_eof && t != '\n' && t != ';') {
      switch (t) {
      case '{':
        t = lexer.getNextToken();
        break;
      case '}':
      {
        t = lexer.getNextToken();
        if (tensorIDName.size() == 0) {
          auto output = std::make_unique<structure>(paramName, tensorID, name, std::move(data));
          return std::move(output);
        }
        else {
          auto output = std::make_unique<structure>(paramName, tensorIDName, name, std::move(data));
          return std::move(output);
        }
        break;
      }
      case '%':
      {
        t = lexer.getNextToken();
        if (t != tok_number) {
          OPTCOUT << " missing tensor id after %, token = " << t << std::endl;
          return nullptr;
        }
        int64_t new_tensorid = lexer.getNumber().tonum<int64_t>();

        // main structure, do not continue for sub-structure process
        if (name.size() == 0) {
          tensorID = new_tensorid;
          t = lexer.getNextToken();
          break;
        }

        // by pass ':'
        t = lexer.getNextToken();
        t = lexer.getNextToken();
        if (t == '<') {
          // value of tensor or scalar
          auto result_value = parseTensor(new_tensorid);
          t = lexer.getCurToken();
          if (result_value.get() != nullptr)
            data->push_back(std::move(result_value));
          else {
            OPTCOUT << " failure parse result value of tensor or scalar." << std::endl;
            return nullptr;
          }
        }
        else if (t == tok_identifier) {
          // structure
          std::string new_name = lexer.getId();
          t = lexer.getNextToken();
          if (t == '{') {
            // value of structure
            auto result_value = parseStructure(tensorIDName, new_tensorid, new_name);
            t = lexer.getCurToken();
            if (result_value.get() != nullptr)
              data->push_back(std::move(result_value));
            else {
              OPTCOUT << " failure parse result value of structure type." << std::endl;
              return nullptr;
            }
          }
          else if (t == '=') {
            // scalar
            auto result_value = parseScalar(new_tensorid, new_name);
            t = lexer.getCurToken();
            if (result_value.get() != nullptr)
              data->push_back(std::move(result_value));
            else {
              OPTCOUT << " failure parse result value of scalar type." << std::endl;
              return nullptr;
            }
          }
          else {
            OPTCOUT << " out of type of structure & scalar, token = " << t << std::endl;
            return nullptr;
          }
        }
        else {
          OPTCOUT << " out of type of structure & scalar, token = " << t << std::endl;
          return nullptr;
        }
        break;
      }
      case ':':
        t = lexer.getNextToken();
        break;
      case ',':
        t = lexer.getNextToken();
        break;
      case tok_identifier:
      {
        // structure or named scalar
        std::string new_name = lexer.getId();
        t = lexer.getNextToken();
        if (t == '{') {

          // main structure, do not continue for sub-structure process
          if (name.size() == 0) {
            name = new_name;
            t = lexer.getNextToken();
            break;
          }

          // value of structure
          auto result_value = parseStructure(tensorIDName, -1, new_name);
          t = lexer.getCurToken();
          if (result_value.get() != nullptr)
            data->push_back(std::move(result_value));
          else {
            OPTCOUT << " failure parse result value of structure type." << std::endl;
            return nullptr;
          }
        }
        else if (t == '=') {
          // find ":identifier{" in string, true - structure
          if (lexer.findIdentifierBracket() == true) {
            // named structure
            auto result_value = parseStructure("", -1, "", new_name);
            t = lexer.getCurToken();
            if (result_value.get() == nullptr) {
              OPTCOUT << " failure parse result value of structure type." << std::endl;
              return nullptr;
            }
            data->push_back(std::move(result_value));
          }
          // find ":<" in string, true - tensor, false - scalar
          else if (lexer.findColonAbracket() == true) {
            // named tensor
            t = lexer.getNextToken();
            auto result_value = parseTensor(-1, new_name);
            t = lexer.getCurToken();
            if (result_value.get() == nullptr) {
              OPTCOUT << " failure parse named tensor in operand." << std::endl;
              return nullptr;
            }
            data->push_back(std::move(result_value));
          }
          else {
            // named scalar
            auto result_value = parseScalar(-1, new_name);
            t = lexer.getCurToken();
            if (result_value.get() == nullptr) {
              OPTCOUT << " failure parse named scaler in operand." << std::endl;
              return nullptr;
            }
            data->push_back(std::move(result_value));
          }

        }
        else if (t == ':') {
          t = lexer.getNextToken();
          if (t == tok_identifier) {
            // structur
            std::string embed_new_name = lexer.getId();
            t = lexer.getNextToken();
            if (t == '{') {
              if (name.size() == 0) {
                name = embed_new_name;
                tensorIDName = new_name;
                continue;
              }

              auto result_value = parseStructure(new_name, -1, embed_new_name);
              t = lexer.getCurToken();
              if (result_value.get() != nullptr)
                data->push_back(std::move(result_value));
              else {
                OPTCOUT << " failure parse result value of structure type." << std::endl;
                return nullptr;
              }
            }
            else {
              // name:type scalar
              auto new_scalar = std::make_unique<scalar>(datatype::CUSTOM_DATA_TYPE, new_name, embed_new_name);
              data->push_back(std::move(new_scalar));
            }
          }
          else if (t == '<') {
            // tensor
            // keep open
          }
          else {
            OPTCOUT << " unknow token of embed, token = " << t << std::endl;
            return nullptr;
          }
        }
        else {
          OPTCOUT << " out of type of structure & scalar, token = " << t << std::endl;
          return nullptr;
        }
        break;
      }
      case tok_number:
      {
        // non-named scalar, e.g. 2.0:f32
        auto result_value = parseScalar(-1, "");
        t = lexer.getCurToken();
        if (result_value.get() != nullptr)
          data->push_back(std::move(result_value));
        else {
          OPTCOUT << " failure parse result value of non-named scalar type." << std::endl;
          return nullptr;
        }
        break;
      }
      case tok_equal:
      {
        t = lexer.getNextToken();
        break;
      }
      default:
         OPTCOUT << " wrong token = " << t << std::endl;
        return nullptr;
        break;
      }
    }
    return nullptr;
  }

  std::unique_ptr<std::vector<std::unique_ptr<value>>> parseOperands() {
    
    std::unique_ptr<std::vector<std::unique_ptr<value>>> operands
      = std::make_unique<std::vector<std::unique_ptr<value>>>();

    Token t = lexer.getCurToken();
    while (t != tok_eof && t != '\n' && t != ';') {
      switch (t) {
      case '(':
        t = lexer.getNextToken();
        break;
      case ')':
      {
        // the end of operands parse
        return std::move(operands);
        break;
      }
      case '{':
        t = lexer.getNextToken();
        break;
      case '}':
      {
        //auto output = std::make_unique<structure>(tensorID, name, std::move(data));
        //return std::move(output);
        t = lexer.getNextToken();
        break;
      }
      case '%':
      {
        t = lexer.getNextToken();
        if (t != tok_number) {
          OPTCOUT << " missing tensor id after %, token = " << t << std::endl;
          return nullptr;
        }
        int64_t new_tensorid = lexer.getNumber().tonum<int64_t>();

        // by pass ':'
        t = lexer.getNextToken();
        t = lexer.getNextToken();
        if (t == '<') {
          // value of tensor or scalar
          auto result_value = parseTensor(new_tensorid);
          t = lexer.getCurToken();
          if (result_value.get() == nullptr) {
            OPTCOUT << " failure parse result value of tensor or scalar." << std::endl;
            return nullptr;
          }
          operands->push_back(std::move(result_value));
        }
        else if (t == tok_identifier) {
          // structure
          std::string new_name = lexer.getId();
          t = lexer.getNextToken();
          if (t == '{') {
            // value of structure
            auto result_value = parseStructure("", new_tensorid, new_name);
            t = lexer.getCurToken();
            if (result_value.get() == nullptr) {
              OPTCOUT << " failure parse result value of structure type." << std::endl;
              return nullptr;
            }
            operands->push_back(std::move(result_value));
          }
          else if (t == '=') {
            // find ":<" in string, true - tensor, false - scalar
            if (lexer.findColonAbracket() == false) {
              // scalar
              auto result_value = parseScalar(new_tensorid, new_name);
              t = lexer.getCurToken();
              if (result_value.get() == nullptr) {
                OPTCOUT << " failure parse result value of scalar type." << std::endl;
                return nullptr;
              }
              operands->push_back(std::move(result_value));
            }
            else {
              // tensor
            }
          }
          else {
            OPTCOUT << " out of type of structure & scalar, token = " << t << std::endl;
            return nullptr;
          }
        }
        else {
          OPTCOUT << " out of type of structure & scalar, token = " << t << std::endl;
          return nullptr;
        }
        break;
      }
      case ',':
        t = lexer.getNextToken();
        break;
      case tok_identifier:
      {
        // structure or named scalar
        std::string new_name = lexer.getId();
        t = lexer.getNextToken();
        if (t == '{') {
          // value of structure
          auto result_value = parseStructure("", -1, new_name);
          t = lexer.getCurToken();
          if (result_value.get() == nullptr) {
            OPTCOUT << " failure parse result value of structure type." << std::endl;
            return nullptr;
          }
          operands->push_back(std::move(result_value));
        }
        else if (t == '=') {
          // find ":identifier{" in string, true - structure
          if (lexer.findIdentifierBracket() == true) {
            // named structure
            auto result_value = parseStructure("", -1, "", new_name);
            t = lexer.getCurToken();
            if (result_value.get() == nullptr) {
              OPTCOUT << " failure parse result value of structure type." << std::endl;
              return nullptr;
            }
            operands->push_back(std::move(result_value));
          }
          // find ":<" in string, true - tensor, false - scalar
          else if (lexer.findColonAbracket() == true) {
            // named tensor
            t = lexer.getNextToken();
            auto result_value = parseTensor(-1, new_name);
            t = lexer.getCurToken();
            if (result_value.get() == nullptr) {
              OPTCOUT << " failure parse named tensor in operand." << std::endl;
              return nullptr;
            }
            operands->push_back(std::move(result_value));
          }
          else {
            // named scalar
            auto result_value = parseScalar(-1, new_name);
            t = lexer.getCurToken();
            if (result_value.get() == nullptr) {
              OPTCOUT << " failure parse named scaler in operand." << std::endl;
              return nullptr;
            }
            operands->push_back(std::move(result_value));
          }
        }
        else if (t == ':') {
          t = lexer.getNextToken();

          if (t == '<') {
            // value of tensor or scalar
            // unsupported
          }
          else if (t == tok_identifier) {
            // structure
            std::string structure_type_name = lexer.getId();
            t = lexer.getNextToken();
            if (t == '{') {
              // value of structure
              // the new_name is object name "embed_tuple" of structure, e.g. embed_tuple:tuple{a=1.0:f32, b=32:i16}
              // the structure_type_name is name "tuple" of structure
              auto result_value = parseStructure(new_name, -1, structure_type_name);
              t = lexer.getCurToken();
              if (result_value.get() == nullptr) {
                OPTCOUT << " failure parse result value of structure type." << std::endl;
                return nullptr;
              }
              operands->push_back(std::move(result_value));
            }
            else if (t == '=') {
              // scalar
              // unsupported
            }
            else {
              // default for nameobj:Type
              auto new_scalar = std::make_unique<scalar>(datatype::CUSTOM_DATA_TYPE, new_name, structure_type_name);
              operands->push_back(std::move(new_scalar));
              t = lexer.getNextToken();
              //OPTCOUT << " out of type of structure & scalar, token = " << t << std::endl;
              //return nullptr;
            }
          }
          else {
            OPTCOUT << " out of type of structure & scalar, token = " << t << std::endl;
            return nullptr;
          }

        }
        else {
          OPTCOUT << " out of type of structure & scalar, token = " << t << std::endl;
          return nullptr;
        }
        break;
      }
      case tok_number:
      {
        // non-named scalar with const value
        auto result_value = parseScalar(-1, "");
        t = lexer.getCurToken();
        if (result_value.get() == nullptr) {
          OPTCOUT << " failure parse result value of scalar type." << std::endl;
          return nullptr;
        }
        operands->push_back(std::move(result_value));
        break;
      }
      default:
        OPTCOUT << " wrong token = " << t << std::endl;
        return nullptr;
        break;
      }
    }

    return std::move(operands);
  }

  std::unique_ptr<instruct> parseInstruct() {
    std::string process_id;
    int64_t rank_id   = 0;
    int64_t stream_id = 0;
    int64_t tensor_id = 0;
    std::string opname;
    std::string domain;
    std::string version;
    std::unique_ptr<std::vector<std::unique_ptr<value>>> operands
      = std::make_unique<std::vector<std::unique_ptr<value>>>();
    std::unique_ptr<std::vector<std::unique_ptr<value>>> output
       = std::make_unique<std::vector<std::unique_ptr<value>>>();
    
    int linenumber = lexer.getLine();

    Token t = lexer.getCurToken();
    while (t != tok_colon) {
      if (t == tok_identifier)
        process_id += lexer.getId();
      else if (t == tok_number)
        process_id += lexer.getNumber().getdata();
      t = lexer.getNextToken();
    }

    t = lexer.getNextToken(); // by pass the ':'
    if (t == tok_number)
      rank_id = lexer.getNumber().tonum<int64_t>();
    else if (t == tok_identifier)
      rank_id = -1;
    else {
      OPTCOUT << " failure parse 'rank id', token = " << t << std::endl;
      return nullptr;
    }

    t = lexer.getNextToken();
    if (t != tok_colon) {
      OPTCOUT << " failure parse ':' after rank id, token = " << t << std::endl;
      return nullptr;
    }

    t = lexer.getNextToken();
    if (t == tok_number)
      stream_id = lexer.getNumber().tonum<int64_t>();
    else {
      OPTCOUT << " failure parse 'stream id', token = " << t << std::endl;
      return nullptr;
    }

    t = lexer.getNextToken();
    if (t == tok_percent) {
      t = lexer.getNextToken();
      if(t == tok_number)
        tensor_id = lexer.getNumber().tonum<int64_t>();
      else {
        OPTCOUT << " failure parse 'tensor id', token = " << t << std::endl;
        return nullptr;
      }
    }
    else {
      OPTCOUT << " failure parse tensor id perfix of '%', token = " << t << std::endl;
      return nullptr;
    }

    
    t = lexer.getNextToken();
    if (t == tok_colon) {
      t = lexer.getNextToken();
      if (t == '<') {
        // value of tensor or scalar
        auto result_value = parseTensor(tensor_id);
        if (result_value.get() != nullptr)
          output->push_back(std::move(result_value));
        else {
          OPTCOUT << " failure parse result value of tensor or scalar." << std::endl;
          return nullptr;
        }
      }
      else if (t == tok_identifier) {
        std::string name = lexer.getId();
        t = lexer.getNextToken();
        if (t == '{') {
          // value of structure
          auto result_value = parseStructure("", tensor_id, name);
          if (result_value.get() != nullptr)
            output->push_back(std::move(result_value));
          else {
            OPTCOUT << " failure parse result value of structure type." << std::endl;
            return nullptr;
          }
        }
        else if (t == '=') {
          // scalar
          auto result_value = parseScalar(tensor_id, name);
          if (result_value.get() != nullptr)
            output->push_back(std::move(result_value));
          else {
            OPTCOUT << " failure parse result value of scalar type." << std::endl;
            return nullptr;
          }
        }
        else {
          OPTCOUT << " out of type of structure & scalar, token = " << t << std::endl;
          return nullptr;
        }
      }
    }
    else {
      std::cout << "line:" <<lexer.getLine() << " failure parse ':' define result tensor shape, token = " << t << std::endl;
      return nullptr;
    }

    t = lexer.getCurToken();
    while (t != '=' && t != tok_eof && t != '\n' && t != ';') {
      t = lexer.getNextToken();
    }
    
    // parse domain, e.g. torch
    t = lexer.getNextToken();
    if (t == tok_identifier) {
      domain = lexer.getId();
    }

    // parse verisonm, e.g 2_10
    std::vector<std::string> ver;
    t = lexer.getNextToken();
    while (t == tok_number || t == tok_underscore) {
      if (t == tok_number) {
        for (auto iter : lexer.getNumber().getdata())
          if (iter != '.')
            version.push_back(iter);
      }
      else if (t == tok_underscore)
        version.push_back('_');
      t = lexer.getNextToken();
    }

    // parse op name, e.g. aten::add
    while (t != '(') {
      if (t == tok_identifier)
        opname += lexer.getId();
      else if (t == ':')
        opname.push_back(':');
      t = lexer.getNextToken();
    }

    if (t == '(') {
      operands = parseOperands();
      if (operands.get() == nullptr) {
        OPTCOUT << " failure parse operand" << std::endl;
        return nullptr;
      }
      t = lexer.getCurToken();
      if(t == ')')
        t = lexer.getNextToken();
    }

    while (t != tok_eof && t != '\n' && t != ';' && linenumber == lexer.getLine()) {
      t = lexer.getNextToken();
    }
    while (t == ';' &&linenumber == lexer.getLine()) {
      t = lexer.getNextToken();
    }

    std::unique_ptr<instruct> ins =
      std::make_unique<instruct>(process_id, rank_id, stream_id,
        opname, domain, version,
        std::move(operands), std::move(output));

    return std::move(ins);
  }

  std::unique_ptr<std::vector<std::unique_ptr<instruct>>> parseModule() {
    auto exprList = std::make_unique<std::vector<std::unique_ptr<instruct>>>();
    while (lexer.getCurToken() != tok_eof) {
      auto expr = parseInstruct();
      if (expr.get() != nullptr) {
        exprList->push_back(std::move(expr));
      }
      else {
        OPTCOUT << "failure parse instruct, continue" << std::endl;
        lexer.gotoNextLine();
        continue;

        //OPTCOUT << " failure parse operand, exit process" << std::endl;
        //return nullptr;
      }
    }
    return std::move(exprList);
  }
};

optrace::optrace() {
}
optrace::optrace(std::string filename) :
filename_(filename), data_(nullptr), size_(0) {
  std::cout << "op trace file: " << filename_ << std::endl;
}
optrace::optrace(const char* data, int size):
  size_(size) {
  std::cout << "op trace buffer, size = " << size_ << std::endl;
  if(size > 0) {
    char *d = new char[size + 1];
    memset(d, 0, size + 1);
    memcpy(d, data, size);
    data_ = d;
  }
}

optrace::~optrace() {
  //std::cout << std::hex << __FUNCTION__ << " addr= " << this << std::endl;
  if(data_ != nullptr)
    delete[] data_;
}

std::istream& operator>>(std::istream& is, optrace& t) {
  (void)t;
  return is;
}

std::shared_ptr<module> optrace::getModule() {
  if (data_ != nullptr && size_ > 0) {
    LexerBuffer lexer(data_, data_ + size_, std::string("string-file"));
    Parser parser(lexer);
    module_ = std::move(parser.getModule());
  }
  else if (filename_.size() > 0) {
    LexerFile lexer(filename_);
    Parser parser(lexer);
    module_ = std::move(parser.getModule());
  }
  return module_;
}

}



