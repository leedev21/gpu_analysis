// test by pybind 2.13, python 3.7.0

#if defined (_WIN32) || defined (_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "optrace.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <sstream>

namespace opt = OPTRACE;
namespace py = pybind11;
using namespace pybind11::literals;

class pyValue : public opt::value {
public:
  using value::value;
  opt::valuetype getValueType() override {
    PYBIND11_OVERRIDE_PURE(
      opt::valuetype, // return
      opt::value,     // parent class
      getValueType    // name of function in c++
                      // argument(s)
    );
  }
  std::string getString() override {
    PYBIND11_OVERRIDE_PURE(
      std::string,
      opt::value,
      getString
      );
  }
};

PYBIND11_MODULE(optrace_py, m) {
  m.doc() = "optrace python interface";

  py::enum_<opt::datatype>(m, "datatype", py::arithmetic())
    .value("F64", opt::datatype::F64)
    .value("C64", opt::datatype::C64)
    .value("C128", opt::datatype::C128)
    .value("I64", opt::datatype::I64)
    .value("U64", opt::datatype::U64)
    .value("BOOL", opt::datatype::BOOL)
    .value("F32", opt::datatype::F32)
    .value("F16", opt::datatype::F16)
    .value("BF16", opt::datatype::BF16)
    .value("F8E4M3", opt::datatype::F8E4M3)
    .value("F8E5M2", opt::datatype::F8E5M2)
    .value("U32", opt::datatype::U32)
    .value("I32", opt::datatype::I32)
    .value("U16", opt::datatype::U16)
    .value("I16", opt::datatype::I16)
    .value("U8", opt::datatype::U8)
    .value("I8", opt::datatype::I8)
    .value("B8", opt::datatype::B8)
    .value("STR", opt::datatype::STR)
    .value("CUSTOM_DATA_TYPE", opt::datatype::CUSTOM_DATA_TYPE);

  py::enum_<opt::valuetype>(m, "valuetype", py::arithmetic())
    .value("TENSOR", opt::valuetype::TENSOR)
    .value("SCALAR", opt::valuetype::SCALAR)
    .value("STRUCT", opt::valuetype::STRUCT);

  py::class_<opt::value, pyValue>(m, "value")
    .def(py::init<>())
    .def("getValueType", &opt::value::getValueType)
    .def("getString", &opt::value::getString);

  py::class_<opt::scalar, opt::value>(m, "scalar")
    .def(py::init<int64_t, std::string, opt::datatype, std::string, std::string>(),
      py::arg("tensorID"), py::arg("name"), py::arg("dtype"), py::arg("data"), py::arg("customerType") = "")
    .def(py::init<int64_t, opt::datatype, std::string>(),
      py::arg("tensorID"), py::arg("dtype"), py::arg("customerType") = "")
    .def(py::init<std::string, opt::datatype, std::string, std::string>(),
      py::arg("name"), py::arg("dtype"), py::arg("data"), py::arg("customerType") = "")
    .def(py::init<opt::datatype, std::string, std::string>(),
      py::arg("dtype"), py::arg("data"), py::arg("customerType") = "")
    .def("getString", &opt::scalar::getString)
    .def("getTensorID", &opt::scalar::getTensorID)
    .def("getName", &opt::scalar::getName)
    .def("getDataType", &opt::scalar::getDataType)
    .def("getCustomerType", &opt::scalar::getCustomerType)
    .def("getValue", &opt::scalar::getValue)
    .def("getDataF32", &opt::scalar::getDataF32)
    .def("getDataU32", &opt::scalar::getDataU32)
    .def("getDataI32", &opt::scalar::getDataI32)
    .def("getDataU16", &opt::scalar::getDataU16)
    .def("getDataI16", &opt::scalar::getDataI16)
    .def("getDataU8", &opt::scalar::getDataU8)
    .def("getDataI8", &opt::scalar::getDataI8)
    .def("getDataPred", &opt::scalar::getDataPred);

  py::class_<opt::tensor, opt::value>(m, "tensor")
    .def(py::init<int64_t, opt::datatype, std::vector<int64_t>&, std::vector<int64_t>, int64_t, std::string>(),
      py::arg("tensorID"), py::arg("dtype"), py::arg("dims"),
      py::arg("strides") = std::vector<int64_t>(), py::arg("offset") = 0,
      py::arg("name") = "")
    .def("getString", &opt::tensor::getString)
    .def("getName", &opt::tensor::getName)
    .def("getParamName", &opt::tensor::getParamName)
    .def("getTensorID", &opt::tensor::getTensorID)
    .def("getOffset", &opt::tensor::getOffset)
    .def("getDataType", &opt::tensor::getDataType)
    .def("getRank", &opt::tensor::getRank)
    .def("getDims", &opt::tensor::getDims)
    .def("getDim", &opt::tensor::getDim)
    .def("getStrides", &opt::tensor::getStrides)
    .def("getStride", &opt::tensor::getStride);

  py::class_<opt::structure, opt::value>(m, "structure")
    .def(py::init<>())
    .def(py::init<int64_t, std::string, std::vector<opt::value*>>())
    .def(py::init<std::string, int64_t, std::string, std::vector<opt::value*>>())
    .def(py::init<std::string, std::string, std::vector<opt::value*>>())
    .def(py::init<std::string, std::string, std::string, std::vector<opt::value*>>())
    .def("getString", &opt::structure::getString)
    .def("getParamName", &opt::structure::getParamName)
    .def("getTensorIDName", &opt::structure::getTensorIDName)
    .def("getTensorID", &opt::structure::getTensorID)
    .def("getName", &opt::structure::getName)
    .def("getData", [](const opt::structure& s) {
      std::vector<opt::value*> data;
      for (const auto& ptr : s.getData()) {
        auto value_new = ptr->clone();
        data.push_back(value_new.get());
        value_new.release();
        //data.push_back(std::shared_ptr<opt::value>(const_cast<opt::value*>(ptr.get()), [](opt::value*) {}));
      }
      return data;
      });

  py::class_<opt::instruct>(m, "instruct")
    .def(py::init<>())
    .def(py::init<std::string&, int64_t, int64_t, std::string&, std::string&, std::string&, std::vector<opt::value*>, std::vector<opt::value*>>())
    .def("getString", &opt::instruct::getString)
    .def("getProcessID", &opt::instruct::getProcessID)
    .def("getRankID", &opt::instruct::getRankID)
    .def("getStreamID", &opt::instruct::getStreamID)
    .def("getOpname", &opt::instruct::getOpname)
    .def("getDomain", &opt::instruct::getDomain)
    .def("getVersion", &opt::instruct::getVersion)
    .def("getOperandSize", &opt::instruct::getOperandSize)
    .def("getResultSize", &opt::instruct::getResultSize)

    .def("getOperands", [](const opt::instruct& s) {
      std::vector<opt::value *> data;
      for (const auto& ptr : s.getOperands()) {
        auto value_new = ptr->clone();
        data.push_back(value_new.get());
        value_new.release();
        //data.push_back(std::shared_ptr<opt::value>(const_cast<opt::value*>(ptr.get()), [](opt::value*) {}));
      }
      return data;
      })
    .def("getResults", [](const opt::instruct& s) {
      std::vector<opt::value *> data;
      for (const auto& ptr : s.getResults()) {
        auto value_new = ptr->clone();
        data.push_back(value_new.get());
        value_new.release();
        //data.push_back(std::shared_ptr<opt::value>(const_cast<opt::value*>(ptr.get()), [](opt::value*) {}));
      }
      return data;
      })
    .def("getOperandItem", &opt::instruct::getOperandItem)
    .def("getResultItem", &opt::instruct::getResultItem);

  py::class_<opt::module>(m, "module")
    .def(py::init<>())
    .def(py::init<std::vector<opt::instruct*>>())
    .def("getString", &opt::module::getString)
    .def("getInstructs", [](const opt::module& s) {
      std::vector<opt::instruct> data;
      for (const auto& ptr : s.getInstructs()) {
        data.push_back(*ptr.get());
        //data.push_back(std::shared_ptr<opt::instruct>(const_cast<opt::instruct*>(ptr.get()), [](opt::instruct*) {}));
      }
      return data;
      })
    .def("__str__", [](const opt::module& mod) {
        std::ostringstream oss;
        oss << mod;
        return oss.str();
      });

    py::class_<opt::optrace>(m, "optrace")
      .def(py::init<>())
      .def(py::init<std::string>())
      .def(py::init<const char*, int>())
      //.def("getModule", &opt::optrace::getModule)
      .def("getModule", [](opt::optrace& s) {
        return *s.getModule().get();
        })
      ;

}
