import sys
import optrace_py as optrace
import os

print("Current working directory:", os.getcwd())

expected_count = 2  # 期望的输入个数
input_str = ""
if len(sys.argv) == expected_count:
    input_str = sys.argv[1]
else:
    raise ValueError(f"Expected {expected_count} inputs, but got {len(sys.argv)}.")

opt = optrace.optrace(input_str, len(input_str))

mod = opt.getModule()

# string to object sample
print("\n\nstring to object sample")
instruct_list = mod.getInstructs()

for instr_dump in instruct_list:
    print(instr_dump.getString())

for instr in instruct_list:
    print("INSTRUCT processid: %s, rank: %d, stream: %d, op: %s" %(instr.getProcessID(), instr.getRankID(), instr.getStreamID(), instr.getOpname()))
    
    """
    for val in instr_dump.getOperands():
        print(f"type(val)={type(val)}; val.getString()={val.getString()}")
        
        if isinstance(val, optrace.scalar):
            print(f"val.getTensorID()={val.getTensorID()}")
            print(f"val.getName()={val.getName()}")
            print(f"val.getDataType()={val.getDataType()}")
            print(f"val.getCustomerType()={val.getCustomerType()}")
            print(f"val.getValue()={val.getValue()}")
    """
