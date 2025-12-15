
#include "optrace.h"
#include <iostream>
#include <memory>
#include <cstring>

using namespace OPTRACE;

//executing sample:
// 1. op-trace "A123:0:0 %3:<32x16xf16> = torch.2_10.aten::add(%1:<32x16xf16>{1, 32}+1024, %2:<f16>, 2.0:f32)"
// 2. op-trace optrace_sample.log
int main(int argc, char **argv)
{
  if (argc <= 1) {
    std::cout << "please provide log string or file" << std::endl;
    return 0;
  }

  size_t datalen = strlen(argv[1]);
  if (datalen <= 4) {
    std::cout << "bad input, data length less than 4 bytes, can't check .log postfix." << std::endl;
    return 0;
  }

  std::unique_ptr<optrace> optracer;

  if (argv[1][datalen - 1] == 'g' && argv[1][datalen - 2] == 'o'
    && argv[1][datalen - 3] == 'l' && argv[1][datalen - 4] == '.') {
    // filename
    optracer = std::make_unique<optrace>(argv[1]);
  }
  else {
    // log string
    optracer = std::make_unique<optrace>(argv[1], static_cast<int>(strlen(argv[1])));
  }

  auto m1 = optracer->getModule();
  if (m1.get() == nullptr) {
    std::cout << "failure get module." << std::endl;
    return -1;
  }

  // copy module test
  OPTRACE::module* m = new OPTRACE::module(*m1.get());

  // dump instruct by getString() interface.
  std::cout << "-------------------------------------------" << std::endl;
  std::cout << "getString by objects: " << std::endl;
  std::cout << m->getString();
  std::cout << std::endl;
  std::cout << "-------------------------------------------" << std::endl << std::endl;

  // access instruct object interfaces.
  int instructIndex = 0;
  // iterator all instructs
  for (auto& instructIter : m->getInstructs()) {
    std::cout << "instruct: " << instructIndex ++ << std::endl;
    std::cout << "\tprocessid: " << instructIter->getProcessID() << ", "
      << "rank:" << instructIter->getRankID() << ", "
      << "stream:" << instructIter->getStreamID() << std::endl;

    // iterator all output results
    std::cout << "\tresults:  \n";
    for (auto& resultIter : instructIter->getResults()) {
      auto type = resultIter->getValueType();
      // dynamic cast operand object to tensor object
      if (resultIter->getValueType() == valuetype::TENSOR) {
        tensor* t = dynamic_cast<tensor*>(resultIter.get());
        std::cout << "\t\ttensor: "
          << "tensorid:" << t->getTensorID() << ", "
          << "rank:" << t->getRank() << ", "
          << "type:" << static_cast<int>(t->getDataType()) << ", "
          << "dims:[";
        for (auto dimiter : t->getDims()) {
          std::cout << dimiter << ",";
        }
        std::cout << "], "
          << "strides:[";
        for (auto strideIter : t->getStrides()) {
          std::cout << strideIter << ",";
        }
        std::cout << "]" << std::endl;
      }
      // dynamic cast operand object to scalar object
      else if (resultIter->getValueType() == valuetype::SCALAR) {
        scalar* t = dynamic_cast<scalar*>(resultIter.get());
        std::cout << "\t\tscalar: "
          << "tensorid:" << t->getTensorID() << ", "
          << "name:" << t->getName() << ", "
          << "type:" << static_cast<int>(t->getDataType()) << ", "
          << "value:" << t->getValue() << std::endl;
      }
      // dynamic cast operand object to structure object
      else if (resultIter->getValueType() == valuetype::STRUCT) {
        structure* t = dynamic_cast<structure*>(resultIter.get());
        std::cout << "\t\tstructure: "
          << "tensorid name:" << t->getTensorIDName() << ", "
          << "tensorid: " << t->getTensorID() << ", "
          << "name: " << t->getName() << std::endl;
      }
    }

    // iterator all input operands
    for (auto& operandIter : instructIter->getOperands()) {
      // dynamic cast operand object to tensor object
      if (operandIter->getValueType() == valuetype::TENSOR) {
        tensor* t = dynamic_cast<tensor*>(operandIter.get());
        // same with result
      }
      // dynamic cast operand object to scalar object
      else if (operandIter->getValueType() == valuetype::SCALAR) {
        scalar *t = dynamic_cast<scalar*>(operandIter.get());
        // same with result
      }
      // dynamic cast operand object to structure object
      else if (operandIter->getValueType() == valuetype::STRUCT) {
        structure* t = dynamic_cast<structure*>(operandIter.get());
        // same with result
      }
    }
  }
}
