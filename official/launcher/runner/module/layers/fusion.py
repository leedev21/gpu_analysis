import torch
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.fused_moe import fused_experts, grouped_topk

from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE, DeepseekV2MLP

class FusedDeepseekV2MoE(DeepseekV2MoE):

    def __init__(
        self,
        config,
        quant_config=None,
        prefix="",
        layer_prior_expert_map=None,
        layer_log2phy=None,
    ):
        super().__init__(
                    config=config,
                    quant_config=quant_config,
                    prefix=f"mlp",)
                    # layer_prior_expert_map=layer_expert_map,
                    # layer_log2phy=layer_log2phy,)
        self.config = config
        self.quant_config = quant_config
        self.block_quant = quant_config is not None and quant_config.weight_block_size is not None
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = config.routed_scaling_factor

        if layer_log2phy is not None:
            self.layer_log2phy = layer_log2phy.to(torch.cuda.current_device())
        else:
            self.layer_log2phy = layer_log2phy

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )
        self.num_experts = config.n_routed_experts
        self.tp_size = 1
        self.check_op = False
        # if config.n_shared_experts is not None:
        #     intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        #     self.shared_experts = DeepseekV2MLP(
        #         hidden_size=config.hidden_size,
        #         intermediate_size=intermediate_size,
        #         hidden_act=config.hidden_act,
        #         quant_config=quant_config,
        #         reduce_results=False,
        #         prefix=f"{prefix}.shared_experts",
        #     )

        #     if quant_config is not None:



        #         def custom_backend(
        #             graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
        #         ):
        #             from torch._inductor import config
        #             from torch._inductor.compile_fx import compile_fx

        #             current_config = config.get_config_copy()
        #             current_config["post_grad_custom_post_pass"] = custom_pass
        #             current_config["enable_auto_functionalized_v2"] = False

        #             return compile_fx(
        #                 graph, example_inputs, config_patches=current_config
        #             )

        #         self.experts.shared_experts = torch.compile(
        #             backend=custom_backend, dynamic=True
        #         )(self.shared_experts)
        #     else:
        #         self.experts.shared_experts = self.shared_experts

            # self.experts.routed_scaling_factor = self.routed_scaling_factor
            # self.experts.log2phy = self.layer_log2phy
        # if use_ep:
        #     # Set TP size to 1 to adjust for EP and adjust EP size and rank
        #     # for DP attention.
        #     self.ep_rank = tp_rank + self.tp_size * self.dp_rank
        #     self.tp_rank = 0
        #     self.ep_size = self.tp_size * self.dp_size
        #     self.tp_size = 1

        #     self.local_num_experts, self.expert_map = get_expert_map(
        #                                                 self.ep_size,
        #                                                 self.ep_rank,
        #                                                 self.global_num_experts,
        #                                                 layer_prior_expert_map
        #                                               )
        # else:
        #     # Adjust TP size for DP attention
        #     self.tp_rank = tp_rank + self.tp_size * self.dp_rank
        #     self.ep_rank = 0
        #     self.tp_size = self.tp_size * self.dp_size
        #     self.ep_size = 1
        #     self.local_num_experts = self.global_num_experts
        #     self.expert_map = None


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = None
        config = self.config
        indices_type = None

        if self.check_op:
            res = [('hidden_states', hidden_states.clone())]
        router_logits, _ = self.gate(hidden_states)
        if self.check_op:
            res.append(('router_logits', router_logits.clone()))

        # layer_prior_expert_map = ?
        activation = 'sllu'
        assert config.topk_group is not None
        assert config.n_group is not None
        topk_weights, topk_ids = grouped_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias)
        if indices_type is not None:
            topk_ids = topk_ids.to(dtype=indices_type)

        if self.check_op:
            res.append(('topk_weights', topk_weights.clone()))
            res.append(('topk_ids', topk_ids.clone()))

        if self.quant_config and self.quant_config.is_checkpoint_fp8_serialized:
            final_hidden_states = fused_experts(
                    hidden_states=hidden_states,
                    w1=self.experts.w13_weight,
                    w2=self.experts.w2_weight,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    inplace=True,
                    activation=self.config.hidden_act,
                    # apply_router_weight_on_input=self.experts.apply_router_weight_on_input,
                    global_num_experts=self.experts.global_num_experts,
                    expert_map=self.experts.expert_map,
                    use_fp8_w8a8=True,
                    block_shape=self.quant_config.weight_block_size,
                    w1_scale=(self.experts.w13_weight_scale_inv
                            if self.block_quant else self.experts.w13_weight_scale),
                    w2_scale=(self.experts.w2_weight_scale_inv
                            if self.block_quant else self.experts.w2_weight_scale),
                    # a1_scale=self.experts.w13_input_scale,
                    # a2_scale=self.experts.w2_input_scale,
                )
        else:
            final_hidden_states = fused_experts(
                    hidden_states=hidden_states,
                    w1=self.experts.w13_weight.to(torch.bfloat16),
                    w2=self.experts.w2_weight.to(torch.bfloat16),
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    inplace=True,
                    activation=self.config.hidden_act,
                    # apply_router_weight_on_input=self.experts.apply_router_weight_on_input,
                    global_num_experts=self.experts.global_num_experts,
                    expert_map=self.experts.expert_map,
                )

        if shared_output is not None:
            final_hidden_states *= self.routed_scaling_factor
            final_hidden_states = final_hidden_states + shared_output
        if self.check_op:
            res.append(('final_hidden_states', final_hidden_states.view(num_tokens, hidden_dim).clone()))
            return res

        return final_hidden_states.view(num_tokens, hidden_dim)
