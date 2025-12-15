import sys
import torchtrace.optrace.optrace_py as optrace
import os

print("Current working directory:", os.getcwd())

expected_count = 2  # 期望的输入个数
input = ""
if len(sys.argv) == expected_count:
    input = sys.argv[1]
else:
    raise ValueError(f"Expected {expected_count} inputs, but got {len(sys.argv)}.")

isfile = False
if ".log" in input:
    isfile = True

opt = None
if isfile:
    opt = optrace.optrace(input)
else:
    opt = optrace.optrace(input, len(input))

mod = opt.getModule()

# string to object sample
print("\n\nstring to object sample")
instruct_list = mod.getInstructs()

for instr_dump in instruct_list:
    print(instr_dump.getString())

for instr in instruct_list:
    print("INSTRUCT processid: %s, rank: %d, stream: %d, op: %s" %(instr.getProcessID(), instr.getRankID(), instr.getStreamID(), instr.getOpname()))

    for val in instr.getResults():
        if isinstance(val, optrace.tensor):
            dimstr = ""
            for d in val.getDims():
                dimstr += str(d)
                dimstr += ", "
            print("    tensor: tensor id: %d, rank: %d, type:%d, dims:[%s]" % (val.getTensorID(), val.getRank(), val.getDataType(), dimstr))
        elif isinstance(val, optrace.scalar):
            print("    scalar: tensor id: %d, name: %s, type:%d, value:%s" % (val.getTensorID(), val.getName(), val.getDataType(), val.getValue()))
        elif isinstance(val, optrace.structure):
            print("    structure: tensor name: %s, tensor id: %d, name:%s" % (val.getTensorIDName(), val.getTensorID(), val.getName()))
        else:
            print("dynamic cast failure")


# object to string sample
print("\n\nobject to string sample")

# create customer type scalar:  name + data type + value + type string
scalar_0 = optrace.scalar("customertype", optrace.datatype.CUSTOM_DATA_TYPE, "none", "NoneType")
print(scalar_0.getString())

# create scalar: tensorID + data type
scalar_1 = optrace.scalar(1, optrace.datatype.I32)
print(scalar_1.getString())

# crate scalar: name + data type + value
scalar_2 = optrace.scalar(-1, "offset", optrace.datatype.I32, "1024")
print(scalar_2.getString())
# crate scalar: name + data type + value
scalar_3 = optrace.scalar("offset", optrace.datatype.I32, "1024")
print(scalar_3.getString())

#create tensor: tensorID + data type + dims + strides + offset
tensor_1 = optrace.tensor(2, optrace.datatype.I32, [2, 3, 4], [12, 4, 1], 512)
print(tensor_1.getString())
#create tensor: tensorID + data type + dims
tensor_2 = optrace.tensor(3, optrace.datatype.I32, [2, 3, 4])
print(tensor_2.getString())
#create tensor: tensorID + data type + dims + offset
tensor_3 = optrace.tensor(4, optrace.datatype.I32, [2, 3, 4], [], 512)
print(tensor_3.getString())

#create instruct
instruct_1 = optrace.instruct("A123", 0, 0, "aten::add", "torch", "2_11", [tensor_1, scalar_1, scalar_2], [tensor_2])
print(instruct_1.getString())

#create structure tensor + scalare
struct_1 = optrace.structure(5, "tuple", [scalar_2, tensor_1, scalar_1])
print(struct_1.getString())
#create structure tensor + scalar + structure
struct_2 = optrace.structure(6, "tuple", [struct_1, tensor_1, scalar_1])
print(struct_2.getString())
#create structure tensor + scalar + parameter named structure
struct_3 = optrace.structure("tuple07", 7, "tuple", [struct_1, tensor_1, scalar_1])
print(struct_3.getString())

#create instruct
instruct_2 = optrace.instruct("A123", 1, 3, "aten::sample_add", "torch", "2_11", [tensor_1, scalar_1, scalar_2, struct_3], [struct_2])
print(instruct_2.getString())
