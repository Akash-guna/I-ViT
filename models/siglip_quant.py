from .quantization_utils import QuantLinear, QuantAct, QuantConv2d, IntGELU, IntLayerNorm


class SiglipMLP(nn.Module):
    def __init__(
            self,config):
        super().__init__()
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = QuantLinear(
            config.hidden_size,
            config.intermediate_size
        )
        self.act = IntGELU()
        self.qact1 = QuantAct()
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = QuantLinear(
            config.intermediate_size,
            config.hidden_size
        )
        self.qact2 = QuantAct(16)

        self.qact_gelu = QuantAct()

    def forward(self, x, act_scaling_factor):
        x, act_scaling_factor = self.fc1(x, act_scaling_factor)
        x, act_scaling_factor = self.qact_gelu(x, act_scaling_factor)
        x, act_scaling_factor = self.act(x, act_scaling_factor)
        x, act_scaling_factor = self.qact1(x, act_scaling_factor)
        x, act_scaling_factor = self.fc2(x, act_scaling_factor)
        x, act_scaling_factor = self.qact2(x, act_scaling_factor)
        return x, act_scaling_factor

class SiglipSdpaAttention(nn.Module):
    """Quantized Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.q_proj = QuantLinear(self.embed_dim, self.embed_dim)
        self.k_proj = QuantLinear(self.embed_dim, self.embed_dim)
        self.v_proj = QuantLinear(self.embed_dim, self.embed_dim)
        self.out_proj = QuantLinear(self.embed_dim, self.embed_dim)

        self.qact1 = QuantAct() 
        self.qact2 = QuantAct()
        self.qact3 = QuantAct(16)

       
        self.matmul_qk = QuantMatMul() 
        self.matmul_attn = QuantMatMul()

        self.int_softmax = IntSoftmax()

    def forward(
        self,
        hidden_states: torch.Tensor,
        act_scaling_factor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        ,  
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states, act_scaling_factor = self.q_proj(hidden_states, act_scaling_factor)
        key_states, act_scaling_factor = self.k_proj(hidden_states, act_scaling_factor)
        value_states, act_scaling_factor = self.v_proj(hidden_states, act_scaling_factor)

        query_states, act_scaling_factor = self.qact1(query_states, act_scaling_factor)
        key_states, act_scaling_factor = self.qact1(key_states, act_scaling_factor)
        value_states, act_scaling_factor = self.qact1(value_states, act_scaling_factor)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights, act_scaling_factor = self.matmul_qk(query_states, act_scaling_factor, key_states.transpose(2, 3), act_scaling_factor)
        attn_weights = attn_weights * self.scale

        if attention_mask is not None:
            attn_weights += attention_mask

        attn_weights, act_scaling_factor = self.int_softmax(attn_weights, act_scaling_factor)

        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output, act_scaling_factor = self.matmul_attn(attn_weights, act_scaling_factor, value_states, act_scaling_factor)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output, act_scaling_factor = self.out_proj(attn_output, act_scaling_factor)
        attn_output, act_scaling_factor = self.qact3(attn_output, act_scaling_factor)

        return attn_output, attn_weights

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        
        self.attention = SiglipSdpaAttention(config)
        self.mlp = SiglipMLP(config)

        self.norm1 = IntLayerNorm(config.hidden_size, eps=config.layer_norm_eps) 
        self.norm2 = IntLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.dropout)

        self.qact1 = QuantAct()
        self.qact2 = QuantAct(16)  

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, act_scaling_factor: Optional[float] = None):
        """
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (torch.Tensor, optional): Mask to apply to the attention scores.
            act_scaling_factor (float, optional): Scaling factor for quantized activations.
        
        Returns:
            torch.Tensor: Output tensor of the encoder layer.
        """
        residual = hidden_states

        hidden_states, act_scaling_factor = self.norm1(hidden_states, act_scaling_factor)
        hidden_states, act_scaling_factor = self.attention(hidden_states, attention_mask, act_scaling_factor)
        hidden_states, act_scaling_factor = self.qact1(hidden_states, act_scaling_factor)
        hidden_states = residual + hidden_states
        hidden_states = self.dropout(hidden_states)
        residual = hidden_states

        hidden_states, act_scaling_factor = self.norm2(hidden_states, act_scaling_factor)

        hidden_states, act_scaling_factor = self.mlp(hidden_states, act_scaling_factor)
        hidden_states, act_scaling_factor = self.qact2(hidden_states, act_scaling_factor)
        hidden_states = residual + hidden_states
        hidden_states = self.dropout(hidden_states)

        return hidden_states, act_scaling_factor

class SiglipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` quantized self-attention layers.
    Each layer is a [`QuantizedSiglipEncoderLayer`].

    Args:
        config: SiglipConfig
    """

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        # Assuming SiglipEncoderLayer has been quantized to support quantized outputs
        self.layers = nn.ModuleList([QuantizedSiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a `~utils.ModelOutput` instead of a plain tuple.
            act_scaling_factor (`float`, *optional*): 
                The activation scaling factor used for quantized activations.

        Returns:
            `BaseModelOutput` or `tuple`: Contains the final hidden states, hidden states of all layers (if requested),
            and the attention outputs (if requested).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
                act_scaling_factor=act_scaling_factor
            )

            # Update hidden states and scaling factor
            hidden_states, act_scaling_factor = layer_outputs[0], layer_outputs[1]

            # Collect attention outputs if requested
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=[hidden_states,act_scaling_factor], hidden_states=encoder_states, attentions=all_attentions
        )

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.qact_input = QuantAct()
        self.encoder = SiglipEncoder(config)
        
        self.post_layernorm = IntLayerNorm(embed_dim, eps=config.layer_norm_eps)
        
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

    @add_start_docstrings_to_model_forward(SIGLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=SiglipVisionConfig)
    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        act_scaling_factor: Optional[float] = None  # Added for quantized activations
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states= self.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding, act_scaling_factor=act_scaling_factor
        )
        hidden_state,act_scaling_factor = QuantAct()
        # Step 2: Pass through the quantized encoder
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            act_scaling_factor=act_scaling_factor,  
        )

        last_hidden_state, act_scaling_factor = encoder_outputs[0], encoder_outputs[1]

        last_hidden_state, act_scaling_factor = self.post_layernorm(last_hidden_state, act_scaling_factor)

        # Step 4: Apply pooling head if needed
        pooler_output = self.head(last_hidden_state, act_scaling_factor) if self.use_head else None

        # Return outputs as either a tuple or BaseModelOutputWithPooling
        if not return_dict:
            return (last_hidden_state, pooler_output) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=[last_hidden_state,act_scaling_factor],
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )