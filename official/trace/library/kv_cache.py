import torch


class Cache():
    def __init__(self):
        self.cache = {}
        self.slot_mapping = {}
        self.state_mapping = {}
        self.state_id = {}
        self.bs = -1
        self.layers = -1
        self.fa_func_schema = {'k': 1,
                      'v': 2,
                      'cu_seqlens_q': 4,
                      'cu_seqlens_k': 5,
                      'max_seqlen_q': 10,
                      'max_seqlen_k': 11,}

    def config(self, bs, layers, mtp=False):
        self.bs = bs
        self.layers = layers
        if mtp:
            self.layers += 1

    def update(self, state, k, v, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
        kv_insert = []
        k_to_send = []
        v_to_send = []
        this_slot_mapping = [cu_seqlens_k[i] - cu_seqlens_k[i-1] if i > 0 else 0 for i in range(len(cu_seqlens_k))]
        if state not in self.slot_mapping:
            self.slot_mapping[state] = []
            self.state_id[state] = len(self.state_mapping) + 1
            self.state_mapping[self.state_id[state]] = state
        elif max_seqlen_q == max_seqlen_k:
            return None, kv_insert, None, None
        for i, insert in enumerate(this_slot_mapping[1:]):
            s = cu_seqlens_k[i]
            if len(self.slot_mapping[state]) <= i:
                self.slot_mapping[state].append(insert)
                kv_insert.append(insert)
                k_to_send.append(k[s:s+insert, :, :].cpu())
                v_to_send.append(v[s:s+insert, :, :].cpu())
            else:
                kv_insert.append(insert - self.slot_mapping[state][i])
                k_to_send.append(k[s+self.slot_mapping[state][i]:s+insert, :, :].cpu())
                v_to_send.append(v[s+self.slot_mapping[state][i]:s+insert, :, :].cpu())
                self.slot_mapping[state][i] = insert
        return self.state_id[state], kv_insert, k_to_send, v_to_send

    def add_rsv(self, kv_rsv):
        if isinstance(kv_rsv, (tuple, list)) and len(kv_rsv) == 4:
            state, kv_insert, k_rsv, v_rsv= kv_rsv
            rank = 0
        else:
            state = kv_rsv.step
            rank = kv_rsv.rank
            def get_args(k):
                if k in kv_rsv.kwargs:
                    return kv_rsv.kwargs[k]
                else:
                    return kv_rsv.args[self.fa_func_schema[k]]
            k_rsv = get_args('k')
            v_rsv = get_args('v')
        if rank not in self.cache:
            self.cache[rank] = {}
        if state not in self.cache[rank]:
            self.cache[rank][state] = {'k': [], 'v': []}
        for i, insert in enumerate(k_rsv):
            print('\tk:', i, insert.size(), insert.dtype)
            if insert is not None:
                if len(self.cache[rank][state]['k']) <= i:
                    self.cache[rank][state]['k'].append(insert)
                else:
                    self.cache[rank][state]['k'][i] = torch.cat((self.cache[rank][state]['k'][i], insert), 0)
        for i, insert in enumerate(v_rsv):
            print('\tv:', i, insert.size(), insert.dtype)
            if insert is not None:
                if len(self.cache[rank][state]['v']) <= i:
                    self.cache[rank][state]['v'].append(insert)
                else:
                    self.cache[rank][state]['v'][i] = torch.cat((self.cache[rank][state]['v'][i], insert), 0)

    def get_kv(self, rank, state):
        return torch.cat(self.cache[rank][state]['k'], 0), torch.cat(self.cache[rank][state]['v'], 0)

    def clear(self):
        self.cache.clear()
        self.slot_mapping.clear()


customer_kv_cache = Cache()
customer_kv_cache_prepare = Cache()


def get_input(k, v, args, kwargs):
    args = list(args)
    def set_args(key, data):
        if key in kwargs:
            kwargs[key] = data
        else:
            args[customer_kv_cache.fa_func_schema[key]] = data
    set_args('k', k)
    set_args('v', v)
    return args, kwargs


def prepare_kv(state, *args, **kwargs):
    kv_for_send = []
    def get_arg(k):
        if k in kwargs:
            return kwargs[k]
        else:
            return args[customer_kv_cache.fa_func_schema[k]]
    # for k, index in fa_func_schema.items():
    #     if k in ['k', 'v']:
    #         print(index, k, len(get_arg(k)))
    #     else:
    #         print(index, k, get_arg(k))
    kv_for_send = customer_kv_cache.update(state, get_arg('k'), get_arg('v'),
                        get_arg('cu_seqlens_k').tolist(), get_arg('max_seqlen_q'),
                        get_arg('max_seqlen_k'))
    # if kv_for_send[0]:
    #     print('kv_insert:', kv_for_send[1], kv_for_send[2][0][:, 0, 0], kv_for_send[3][0][:, 0, 0])
    return kv_for_send, get_arg('k'), get_arg('v')


def prepare_for_send(state, *args, **kwargs):
    kv_for_send, k, v = prepare_kv(state, *args, **kwargs)
    if not kv_for_send[0]:
        return 0, None, None
    args, kwargs = get_input(kv_for_send[2], kv_for_send[3], args, kwargs)
    return kv_for_send[0], args, kwargs