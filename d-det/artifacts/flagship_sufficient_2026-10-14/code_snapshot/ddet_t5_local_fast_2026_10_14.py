"""Encoder-only T5 attention with equivalent SDPA computation, plus local checks."""
import copy,inspect,json,time,types
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers.models.t5.modeling_t5 import T5Attention,T5EncoderModel,T5LayerSelfAttention


def attention(self,hidden_states,mask=None,key_value_states=None,position_bias=None,past_key_values=None,output_attentions=False,**kwargs):
    if self.is_decoder or key_value_states is not None or past_key_values is not None or output_attentions:
        return self._original_forward(hidden_states,mask=mask,key_value_states=key_value_states,position_bias=position_bias,past_key_values=past_key_values,output_attentions=output_attentions,**kwargs)
    batch,length=hidden_states.shape[:2]
    q=self.q(hidden_states).view(batch,length,self.n_heads,self.key_value_proj_dim).transpose(1,2)
    k=self.k(hidden_states).view(batch,length,self.n_heads,self.key_value_proj_dim).transpose(1,2)
    v=self.v(hidden_states).view(batch,length,self.n_heads,self.key_value_proj_dim).transpose(1,2)
    if position_bias is None:
        if self.has_relative_attention_bias:
            position_bias=self.compute_bias(length,length,device=q.device,past_seen_tokens=0)
        else:
            position_bias=torch.zeros((1,self.n_heads,length,length),device=q.device,dtype=q.dtype)
            if self.gradient_checkpointing and self.training:position_bias.requires_grad=True
        if mask is not None:position_bias=position_bias+mask[:,:,:,:length]
    # T5 intentionally uses unscaled QK; the default 1/sqrt(d) would change the model.
    output=F.scaled_dot_product_attention(q,k,v,attn_mask=position_bias.to(q.dtype),dropout_p=self.dropout if self.training else 0.,scale=1.)
    output=self.o(output.transpose(1,2).contiguous().view(batch,length,self.inner_dim))
    return (output,position_bias,None) if self._return_triple else (output,position_bias)


def install(model):
    triple='attention_output, position_bias, attn_weights = self.SelfAttention' in inspect.getsource(T5LayerSelfAttention.forward)
    # Modern T5 routes masks/bias through its native attention interface; retain it.
    # The legacy adapter must not be applied across that API boundary.
    if triple:return model
    for module in model.modules():
        if isinstance(module,T5Attention) and not hasattr(module,'_original_forward'):
            module._original_forward=module.forward
            module._return_triple=triple
            module.forward=types.MethodType(attention,module)
    return model


def main():
    root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);torch.manual_seed(20261014)
    original=T5EncoderModel.from_pretrained(root/'d-det/models/codet5-small',local_files_only=True).eval()
    fast=install(copy.deepcopy(original)).eval()
    ids=torch.randint(100,2000,(2,40));mask=torch.ones_like(ids);mask[1,29:]=0
    a=original(ids,attention_mask=mask).last_hidden_state;b=fast(ids,attention_mask=mask).last_hidden_state
    error=float((a-b).abs().max().detach());assert torch.allclose(a,b,atol=2e-5,rtol=2e-4),error
    (a.square().mean()).backward();(b.square().mean()).backward()
    ge=max(float((p.grad-q.grad).abs().max()) for p,q in zip(original.parameters(),fast.parameters()) if p.grad is not None)
    assert ge<2e-4,ge
    del original,fast,a,b
    result={'fp32_forward_max_abs_error':error,'fp32_gradient_max_abs_error':ge,'encoder_only_scale':1.0,'modern_native_attention':'attention_output, position_bias, attn_weights = self.SelfAttention' in inspect.getsource(T5LayerSelfAttention.forward),'benchmark':[]}
    for batch in [4,8,16]:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        model=install(T5EncoderModel.from_pretrained(root/'d-det/models/codet5-small',local_files_only=True)).cuda()
        model.gradient_checkpointing_enable();model.train();ids=torch.randint(100,2000,(batch,512),device='cuda');mask=torch.ones_like(ids)
        times=[]
        try:
            for step in range(4):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.float16):out=model(ids,attention_mask=mask).last_hidden_state;loss=out.float().square().mean()
                loss.backward();torch.cuda.synchronize();times.append(time.perf_counter()-start)
            result['benchmark'].append({'micro_batch':batch,'seconds_per_step':sum(times[1:])/3,'rows_per_second':batch/(sum(times[1:])/3),'peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576})
        except torch.cuda.OutOfMemoryError:
            result['benchmark'].append({'micro_batch':batch,'oom':True})
        del model,ids,mask
        if 'out' in locals():del out,loss
    path=root/'d-det/artifacts/flagship_sufficient_2026-10-14';path.mkdir(parents=True,exist_ok=True)
    (path/'attention_equivalence_and_benchmark.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
