import sys
import optrace_py as optrace
import os


expected_count = 2  # 期望的输入个数
log_filename = ""
if len(sys.argv) == expected_count:
    log_filename = sys.argv[1]
else:
    raise ValueError(f"Expected {expected_count} inputs, but got {len(sys.argv)}.")

print(f"parsing {log_filename}:")
with open(log_filename, 'r') as file:
    line = file.readline()  # 读取第一行
    while line:
    
        print(f"----------------------------------", end='\n', flush=True)
        print(f"parsing line :{line}", end='\n', flush=True)
        try:
            # 可能引发异常的代码
            opt = optrace.optrace(line, len(line))
            mod = opt.getModule()
            instruct_list = mod.getInstructs()

            print("instruct_list:", end='', flush=True)
            for instr_dump in instruct_list:
                print(instr_dump.getString())    
            
        except Exception as e:
            # 捕获所有异常
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", end='\n', flush=True)
            print(f"!!!!!!发生了异常: {e}", end='', flush=True)
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", end='\n', flush=True)

        print(f"----------------------------------", end='\n', flush=True)
        print("", end='\n', flush=True)
        
        line = file.readline()  # 读取第一行




