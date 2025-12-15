PYTHONIOENCODING=utf-8 \
python -m vllm_utils.benchmark_vision_language \
--backend vllm \
--demo \
--model=qwen2.5 \
--prompt=口播：来，欢迎新进直播间的家人们，来看看咱家这款香辣牛肉酱。9.99包邮到家。9.99到手4瓶展开给您包邮到家还赠送运费险。,###结合图片和口播信息，先判断当前在售的商品是什么，然后提取当前在售商品的属性 \
--dtype=bfloat16 \
--max-output-len=256 \
--device=cuda \
--tensor-parallel-size 1 \
--max-model-len 8192 \
--trust-remote-code \
--block-size=64 \
--enforce-eager
