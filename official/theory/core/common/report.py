class Report(object):
    def __init__(self, name):
        self.name = name
        self.op = []
        self.module = []
        self.model = []
        self.total = {}
        self.target = {}
        self.filter = {
            'modules': ['Embedding', 'Attention', 'MLP', 'Post&Loss', 'Adam']
        }
        self.conf = {}
        self.decoding = False
        self.debug = False

        # Field name mapping dictionary
        self.field_mappings = {
            # Model level field mapping
            'model': {
                'MFU': 'MFU',
                'total_latency': 'total_time',
                'tps': 'tokens_per_sec_per_device',
                '2D_percent': 'percentage_2D',
                'comm_percent': 'percentage_COMM',
                'FA_percent': 'percentage_FA',
                'params': 'weight',
                'activation': 'forward_activation',
                'optim': 'optimizer',
                'total_mem': 'total_memory',
                'flops': 'device_tflops',
                'bubble_rate': 'bubble_rate',
                'io': 'io',
                '2D_flops': 'flops_2D',
                'HW': 'hardware',
                'default_2D_percent': 'default_percentage_2D',
                'decoding_2D_percent': 'decoding_percentage_2D',
                'total_2D_percent': 'total_percentage_2D',
                'default_FA_percent': 'default_percentage_FA',
                'decoding_FA_percent': 'decoding_percentage_FA',
                'total_FA_percent': 'total_percentage_FA',
                'default_comm_percent': 'default_percentage_COMM',
                'decoding_comm_percent': 'decoding_percentage_COMM',
                'total_comm_percent': 'total_percentage_COMM',
            },

            # Module level field mapping
            'module': {
                'io': 'io',
                'params': 'weight',
                'activation': 'forward_activation',
                'latency': 'total_time',
                'shape': 'input_shape',
                'num_layers': 'num_layers',
                'flops': 'device_tflops',
                '2D_flops': 'flops_2D',
            },

            # OP level field mapping
            'op': {
                'op_type': 'dtype',
                'shape': 'input_shape',
                'params': 'weight',
                'activation': 'forward_activation',
                'latency': 'duration',
            }
        }

    def get_mapped_name(self, category, field_name):
        """Get mapped field name"""
        if category in self.field_mappings and field_name in self.field_mappings[category]:
            return self.field_mappings[category][field_name]
        return field_name

    def apply_field_mapping(self, category, data_dict):
        """Apply field mapping to data dictionary"""
        if not isinstance(data_dict, dict):
            return data_dict

        mapped_dict = {}
        mapping_changes = []

        for key, value in data_dict.items():
            mapped_key = self.get_mapped_name(category, key)
            mapped_dict[mapped_key] = value

            if mapped_key != key:
                mapping_changes.append(f"  {key} → {mapped_key}")
            else:
                mapping_changes.append(f"  {key} (no change)")

        return mapped_dict


    def __call__(self, cls, item, data, stage=None, test_id=None):
        if isinstance(item, str) and item.startswith('layer::'):
            item = item.replace('layer::', '')
            data['shape'] = data['shape'][1]
        test_id = test_id if test_id else self.conf['device']
        if isinstance(item, str) and item in self.target and 'latency' in data:
            target = self.target[item]
            diff = round(abs(target - data['latency']) / target * 100, 2)
            pass_fail = "Fail" if diff and diff > 10 else "Pass"
            if self.debug:
                print('report:', cls, item, data, f"target={self.target[item]} diff={diff}% {pass_fail}")
        elif self.debug:
            print('report:', cls, item, data)
        if cls == 'model' and isinstance(item, dict):
            self.model.append((item, test_id, data))
        if cls == 'module':
            self.module.append((item, data, test_id, stage))
        # if cls == 'op' and item in ['te::FlashAttnVarlenFunc', 'te::FlashAttnVarlenFuncBackward']:
        if cls == 'op':
            # and item in ['te::FusedRoPEFunc', 'te::FusedRoPEFuncBackward', 'te::nvte_rmsnorm_fwd', 'te::nvte_rmsnorm_bwd']:
            if self.conf.get('iter') and self.conf['iter'] != 0 and self.conf['iter'] < self.conf['tokens']:
                return
            self.op.append((item, test_id, self.conf.get('iter'), data))

    def add_target(self, target):
        self.target = target

    def config(self, attr, value):
        if hasattr(self, attr):
            setattr(self, attr, value)
        else:
            self.conf[attr] = value

    def report(self):
        '''
        ============ Serving Benchmark Result ============
        Successful requests:                     8
        Benchmark duration (s):                  83.97600
        Total input tokens:                      16384
        Total generated tokens:                  2032
        Request throughput (req/s):              0.09527
        Output token throughput (tok/s):         24.19739
        Total Token throughput (tok/s):          219.30076
        ---------------Time to First Token----------------
        Mean TTFT (ms):                          15067.18187
        Median TTFT (ms):                        15067.23289
        P99 TTFT (ms):                           18184.18872
        -----Time per Output Token (excl. 1st token)------
        Mean TPOT (ms):                          272.12139
        Median TPOT (ms):                        273.10620
        P99 TPOT (ms):                           287.27438
        ---------------Inter-token Latency----------------
        Mean ITL (ms):                           272.09889
        Median ITL (ms):                         253.38049
        P99 ITL (ms):                            387.76819
        ==================================================
        f"{num_parameters_on_most_loaded_model_shard / 10**9:.4f}"
        num_bytes_per_parameter = (
            18 if not args.use_distributed_optimizer else 6 + (12 / args.data_parallel_size)
    )
        '''
        summary =[]

        def print_info(title, item, stage, key, base_hw, div=1):
            for item, test_id, data in self.model:
                if key not in data[stage]:
                    return
                if item['hw'] == base_hw:
                    base = data[stage][key]
                    break
            print('*'*50, title, '*'*50)
            try:
                print('base:', base)
            except:
                # print('Err create report:', item, stage, key, base_hw, div, self.model)
                base = self.model[0][2][stage][key]
            self.total[title] = []
            for item, test_id, data in self.model:
                if key in ['MFU', '2D_percent', 'comm_percent', 'FA_percent', 'bubble_rate']:
                    value = f'{round(data[stage][key]*100, 1)}%'
                elif key == 'flops':
                    tflops_value = data[stage][key]/1000000000000
                    value = f'{tflops_value:.2e} TFLOPS'
                elif key == '2D_flops':
                    value = f'{data[stage][key]:.2e} FLOPS'
                else:
                    _v = data[stage][key]/div
                    unit = '' if key == 'tps' else ' ms' if stage == 'decoding' and key == 'total_latency' and not self.conf['text_to_image'] else ' s'
                    if key in ['params', 'activation', 'optim', 'total_mem']:
                        unit =  ' GB'
                    value = f'{round(_v, 3)}{unit}'
                self.total[title].append({'Key': title, 'hardware': test_id, 'Value': value,
                                        f'HW vs {base_hw}': f'{round(data[stage][key]/base, 2)}X',
                                        f'{base_hw} vs HW': f'{round(base/data[stage][key], 2)}X'})
                print(f'{test_id} [ {value} ] {item["hw"]} vs {base_hw}: {round(data[stage][key]/base, 2)}X {base_hw} vs {item["hw"]}: {round(base/data[stage][key], 2)}X')

        if self.op:
            print('*'*50, 'op', '*'*50)
            for item, device, iter, data in self.op:
                try:
                    if 'shape' in data:
                        print(device, iter, item, data['shape'], f"{round(data['latency'], 3)}us")
                    else:
                        print(device, iter, item, data['B'], data['M'], data['K'], data['N'], f"{round(data['latency'], 3)}us")
                except:
                    print('Err op process:', device, iter, item, data)
                    exit()

        modules = []
        if self.module:
            print('*'*50, 'module', '*'*50)
            for item, data, device, stage in self.module:
                module_name = item.replace('layer::', '')
                if self.filter.get('modules') and module_name in self.filter['modules']:
                    print(device, item, stage, data.get('params'), data.get('activation'))
                    modules.append((module_name, data, device, stage))

        if self.model:
            # for item, test_id, data in self.model:
            #     if not self.decoding:
            #         print(item, data['default'])
            #     else:
            #         for stage in data:
            #             print(item, stage, data[stage])
            base_hw = 'H20SXM' if self.decoding else 'A100SXM'
            if base_hw not in []:
                base_hw = 'H20SXM'
            for key in ['MFU', 'total_latency', 'tps', '2D_percent', 'comm_percent', 'FA_percent']:
                if not self.decoding:
                    print_info(key, item, 'default', key, base_hw)
                else:
                    for stage in ['default', 'decoding', 'total']:
                        if stage == 'default' and key == 'total_latency' and not self.conf['text_to_image']:
                            print_info('ttft', item, stage, key, base_hw)
                        elif stage == 'decoding' and key == 'total_latency' and not self.conf['text_to_image']:
                            print_info('tpot', item, stage, key, base_hw, div=self.conf['tokens']/1000)
                        else:
                            title = f'{stage}_{key}' if key != 'total_latency' else 'total_latency'
                            print_info(title, item, stage, key, base_hw)
            for key in ['params', 'activation', 'optim', 'total_mem', 'bubble_rate']: # , 'bubble_rate', 'bubble']:
                print_info(key, item, 'total', key, base_hw)
            for key in ['flops', '2D_flops']:
                print_info(key, item, 'total', key, base_hw)

        # Apply field mapping to actual data
        # 1. Convert model data
        mapped_model = []
        for item, test_id, data in self.model:
            mapped_data = {}
            for stage, stage_data in data.items():
                mapped_data[stage] = self.apply_field_mapping('model', stage_data)
            mapped_model.append((item, test_id, mapped_data))

        # 2. Convert module data
        mapped_modules = []
        for module_name, data, device, stage in modules:
            mapped_data = self.apply_field_mapping('module', data)
            mapped_modules.append((module_name, mapped_data, device, stage))

        # 3. Convert op data
        mapped_op = []
        for item, test_id, iter_num, data in self.op:
            mapped_data = self.apply_field_mapping('op', data)
            mapped_op.append((item, test_id, iter_num, mapped_data))

        # 4. Convert total data key names
        mapped_total = {}
        for key, value in self.total.items():
            mapped_key = self.get_mapped_name('model', key)
            # Map the 'Key' field of each inner dictionary
            mapped_value = []
            for item in value:
                if isinstance(item, dict) and 'Key' in item:
                    item_copy = item.copy()
                    item_copy['Key'] = mapped_key  # Use mapped key name
                    mapped_value.append(item_copy)
                else:
                    mapped_value.append(item)
            mapped_total[mapped_key] = mapped_value


        return {
            'model': mapped_model,
            'module': mapped_modules,
            'op': mapped_op,
            'total': mapped_total
        }