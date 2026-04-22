"""
MuKV rerank retrieval module.
基于重排检索机制的 MuKV 主实验模块。
"""
import torch
from logzero import logger
from model.mukv_dualsignal_compression import MuKVDualSignalCompressionModel


class MuKVRerankModel(MuKVDualSignalCompressionModel):
    """MuKV 的重排检索模型。"""
    
    def __init__(
        self, 
        config, 
        processor, 
        n_frame_tokens, 
        init_prompt_ids, 
        n_local, 
        topk, 
        chunk_size,
        granularities, 
        granularity_topks,
        # 继承Token压缩和FFT参数
        enable_compression=True,
        keep_ratios=None,
        importance_method='attention_fft_weighted',
        dedup_threshold=0.90,
        dedup_enabled=True,
        fft_enabled=True,
        fft_method='diff',
        attention_weight=None,
        fft_weight=None,
        # 🆕 Attention层策略
        attention_layer_strategy='last',
        # 🆕 统一层策略
        use_unified_layer=False,
        # 🆕 Rerank参数
        enable_rerank=True,
        rerank_alpha=0.5,  # 中粒度的consistency权重
        rerank_beta=0.6,   # 细粒度的consistency权重
        rerank_top_n=5,    # 用于提取全局上下文的top-N数量
        rerank_reverse=False,  # 🔄 反向检索：选择最不相关的blocks
    ):
        # 调用父类初始化
        super().__init__(
            config=config,
            processor=processor,
            n_frame_tokens=n_frame_tokens,
            init_prompt_ids=init_prompt_ids,
            n_local=n_local,
            topk=topk,
            chunk_size=chunk_size,
            granularities=granularities,
            granularity_topks=granularity_topks,
            enable_compression=enable_compression,
            keep_ratios=keep_ratios,
            importance_method=importance_method,
            dedup_threshold=dedup_threshold,
            dedup_enabled=dedup_enabled,
            fft_enabled=fft_enabled,
            fft_method=fft_method,
            attention_weight=attention_weight,
            fft_weight=fft_weight,
            attention_layer_strategy=attention_layer_strategy,
            use_unified_layer=use_unified_layer,
        )
        
        # 🆕 Rerank相关参数
        self.enable_rerank = enable_rerank
        self.rerank_alpha = rerank_alpha
        self.rerank_beta = rerank_beta
        self.rerank_top_n = rerank_top_n
        self.rerank_reverse = rerank_reverse
        
        logger.info("MuKV rerank configuration:")
        logger.info(f"  Enable Rerank: {self.enable_rerank}")
        if self.enable_rerank:
            logger.info(f"  Alpha (medium grain): {self.rerank_alpha}")
            logger.info(f"  Beta (fine grain): {self.rerank_beta}")
            logger.info(f"  Top-N for context: {self.rerank_top_n}")
            logger.info(f"  🔄 Reverse Mode: {self.rerank_reverse}")
    
    @torch.inference_mode()
    def _rerank_with_context(
        self, 
        query_vector, 
        coarse_candidates, 
        medium_candidates, 
        fine_candidates
    ):
        """
        Context-Aware Rerank算法
        
        Args:
            query_vector: 查询向量
            coarse_candidates: [(repr_vec, score, kv_layers, idx), ...] 粗粒度候选
            medium_candidates: 中粒度候选
            fine_candidates: 细粒度候选
        
        Returns:
            重排后的最终候选列表 (每个粒度)
        """
        # Step 1: 提取全局上下文向量（从top-N粗粒度候选）
        sorted_coarse = sorted(coarse_candidates, key=lambda x: x[1], reverse=True)
        top_n_coarse = sorted_coarse[:min(self.rerank_top_n, len(sorted_coarse))]
        
        if len(top_n_coarse) > 0:
            coarse_vectors = torch.stack([c[0] for c in top_n_coarse])  # (N, dim)
            global_context_vector = coarse_vectors.mean(dim=0)  # (dim,)
        else:
            # 如果没有粗粒度候选，使用query_vector作为context
            global_context_vector = query_vector
        
        logger.debug(f"[Rerank] Global context from top-{len(top_n_coarse)} coarse blocks")
        
        # Step 2: 计算中粒度的一致性分数
        for i in range(len(medium_candidates)):
            repr_vec, orig_score, kv_layers, idx = medium_candidates[i]
            consistency = torch.cosine_similarity(
                repr_vec.unsqueeze(0), 
                global_context_vector.unsqueeze(0), 
                dim=1
            ).item()
            
            # 加权融合：original_score * (1-α) + consistency * α
            reranked_score = (1 - self.rerank_alpha) * orig_score + self.rerank_alpha * consistency
            medium_candidates[i] = (repr_vec, reranked_score, kv_layers, idx)
        
        # Step 3: 计算细粒度的一致性分数
        for i in range(len(fine_candidates)):
            repr_vec, orig_score, kv_layers, idx = fine_candidates[i]
            consistency = torch.cosine_similarity(
                repr_vec.unsqueeze(0), 
                global_context_vector.unsqueeze(0), 
                dim=1
            ).item()
            
            # 加权融合：original_score * (1-β) + consistency * β
            reranked_score = (1 - self.rerank_beta) * orig_score + self.rerank_beta * consistency
            fine_candidates[i] = (repr_vec, reranked_score, kv_layers, idx)
        
        # Step 4: 按新分数重排并返回
        coarse_sorted = sorted_coarse  # 粗粒度保持原始排序
        medium_sorted = sorted(medium_candidates, key=lambda x: x[1], reverse=True)
        fine_sorted = sorted(fine_candidates, key=lambda x: x[1], reverse=True)
        
        # 🔄 反向模式：选择最不相关的blocks
        if self.rerank_reverse:
            logger.warning(f"🔄 REVERSE MODE: Selecting LEAST relevant blocks!")
            # 反转排序，取最低分的
            medium_sorted = medium_sorted[::-1]  # 完全反转列表
            fine_sorted = fine_sorted[::-1]
            coarse_sorted = coarse_sorted[::-1]
        
        return coarse_sorted, medium_sorted, fine_sorted
    
    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        """
        带Rerank的多粒度问答
        
        流程：
        1. 正常检索（如果enable_rerank=True，检索2倍数量）
        2. Context-aware重排
        3. 选择最终的topk blocks
        4. 拼接KV并生成答案
        """
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]
        
        # 编码问题
        input_ids = self.processor.tokenizer(input_text['question']).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        
        with torch.no_grad():
            question_output = self.language_model(
                inputs_embeds=inputs_embeds,
                past_key_values=self.init_kv_cache,
                use_cache=True,
                return_dict=True
            )
        
        if question_output.past_key_values is not None and len(question_output.past_key_values) > 0:
            # 🆕 根据 use_unified_layer 选择层
            n_layers = len(question_output.past_key_values)
            if self.use_unified_layer:
                layer_idx = self._get_target_layer_idx(n_layers)
                selected_layer_kv = question_output.past_key_values[layer_idx]
            else:
                selected_layer_kv = question_output.past_key_values[-1]  # 默认最后一层
            
            selected_k = selected_layer_kv[0][:, :, -1, :]
            query_vector = selected_k.reshape(selected_k.size(0), -1).squeeze(0)
        else:
            raise RuntimeError("Failed to get past_key_values")
        
        print(f"\n{'='*60}")
        if self.enable_rerank:
            print(f"[🆕 Two-Stage Retrieval + Context-Aware Rerank]")
        else:
            print(f"[Multi-Granularity Retrieval]")
        print(f"Query vector shape: {query_vector.shape}")
        print(f"{'='*60}")
        
        # 决定检索数量
        if self.enable_rerank:
            # 第一阶段：over-retrieve (2倍数量)
            retrieval_topks = [tk * 2 for tk in self.granularity_topks]
            print(f"📥 Stage 1: Over-retrieval ({sum(retrieval_topks)} blocks)")
        else:
            # 不使用rerank，正常检索
            retrieval_topks = self.granularity_topks
        
        # 检索所有粒度的候选（带分数）
        all_candidates = {gran: [] for gran in self.granularities}
        
        for gran_idx, granularity in enumerate(self.granularities):
            topk_for_gran = retrieval_topks[gran_idx]
            
            # 使用父类的retrieve方法，但需要获取scores
            # 这里简化处理：直接调用store的方法
            kv_store_dict = self.multigran_kv_store.kv_stores[granularity]
            repr_vectors = kv_store_dict['repr_vectors']
            
            if len(repr_vectors) == 0:
                continue
            
            repr_tensors = torch.stack(repr_vectors).to(device)
            similarities = torch.cosine_similarity(
                query_vector.unsqueeze(0), 
                repr_tensors, 
                dim=1
            )
            
            topk_actual = min(topk_for_gran, len(similarities))
            scores, indices = torch.topk(similarities, topk_actual)
            
            for score, idx in zip(scores, indices):
                kv_layers = kv_store_dict['keys'][idx.item()]
                kv_layers_gpu = [(k.to(device), v.to(device)) for k, v in kv_layers]
                repr_vec = repr_vectors[idx.item()].to(device)
                
                all_candidates[granularity].append((repr_vec, score.item(), kv_layers_gpu, idx.item()))
            
            print(f"  Gran {granularity:3d}: {len(all_candidates[granularity]):2d} candidates")
        
        # 🆕 Context-Aware Rerank
        if self.enable_rerank and len(self.granularities) == 3:
            print(f"\n📊 Stage 2: Context-Aware Reranking...")
            
            # 假设粒度顺序是 [49, 196, 784] (fine, medium, coarse)
            coarse_gran = self.granularities[-1]  # 784
            medium_gran = self.granularities[1]   # 196  
            fine_gran = self.granularities[0]     # 49
            
            coarse_sorted, medium_sorted, fine_sorted = self._rerank_with_context(
                query_vector,
                all_candidates[coarse_gran],
                all_candidates[medium_gran],
                all_candidates[fine_gran]
            )
            
            # 选择最终数量
            final_candidates = {
                coarse_gran: coarse_sorted[:self.granularity_topks[2]],
                medium_gran: medium_sorted[:self.granularity_topks[1]],
                fine_gran: fine_sorted[:self.granularity_topks[0]]
            }
            
            print(f"✅ Final: {self.granularity_topks[0]}/{self.granularity_topks[1]}/{self.granularity_topks[2]} blocks selected")
        else:
            # 不使用rerank或粒度数量不是3，直接使用原始结果
            final_candidates = {}
            for gran_idx, granularity in enumerate(self.granularities):
                sorted_cands = sorted(all_candidates[granularity], key=lambda x: x[1], reverse=True)
                final_candidates[granularity] = sorted_cands[:self.granularity_topks[gran_idx]]
        
        print(f"{'='*60}\n")
        
        # 拼接所有选中的KV
        all_retrieved_kvs = []
        for granularity in self.granularities:
            for _, _, kv_layers, _ in final_candidates[granularity]:
                all_retrieved_kvs.append(kv_layers)
        
        # 组合KV-Cache
        if len(all_retrieved_kvs) == 0:
            combined_kv = self.init_kv_cache
        else:
            combined_kv = []
            n_layers = len(self.init_kv_cache)
            
            for layer_idx in range(n_layers):
                init_k, init_v = self.init_kv_cache[layer_idx]
                layer_keys = [init_k]
                layer_values = [init_v]
                
                for retrieved_kv_layers in all_retrieved_kvs:
                    if layer_idx < len(retrieved_kv_layers):
                        layer_k, layer_v = retrieved_kv_layers[layer_idx]
                        layer_keys.append(layer_k)
                        layer_values.append(layer_v)
                
                combined_k = torch.cat(layer_keys, dim=2)
                combined_v = torch.cat(layer_values, dim=2)
                combined_kv.append((combined_k, combined_v))
        
        # 生成答案（与父类相同）
        print(f"[Generation] Max {max_new_tokens} tokens...", flush=True)
        output_ids = []
        past_key_values = combined_kv
        
        for i in range(max_new_tokens):
            if i == 0:
                input_ids = self.processor.tokenizer(input_text['prompt']).input_ids
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(
                    inputs_embeds=inputs_embeds, 
                    use_cache=True, 
                    past_key_values=past_key_values
                )
                past_key_values = out.past_key_values
                logits = out.logits
            else:
                out = self.language_model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0, -1, :]
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
            token = tokens[0]

            output_ids.append(token)

            if token in stop_token_ids:
                break

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        
        print(f"✅ Generated {len(output_ids)} tokens\n")
        
        return output


def load_model(
    model_path='model_zoo/llava_onevisionqwen2_0.5b_ov_hf',
    n_init=None, 
    n_local=None, 
    topk=64, 
    chunk_size=1,
    granularities=[49, 196, 784], 
    granularity_topks=None,
    # Token压缩参数
    enable_compression=True,
    keep_ratio_49=0.70,
    keep_ratio_196=0.50,
    keep_ratio_784=0.30,
    importance_method='attention_fft_weighted',
    dedup_threshold=0.90,
    dedup_enabled=True,
    # FFT参数
    fft_enabled=True,
    fft_method='diff',
    attention_weight_49=0.5,
    attention_weight_196=0.7,
    attention_weight_784=0.8,
    # 🆕 Attention层策略
    attention_layer_strategy='last',
    # 🆕 统一层策略
    use_unified_layer=False,
    # 🆕 Rerank参数
    enable_rerank=True,
    rerank_alpha=0.5,
    rerank_beta=0.6,
    rerank_top_n=5,
    rerank_reverse=False,
):
    """加载带Context-Aware Rerank的FFT增强Token压缩多粒度MuKV模型"""
    from transformers import LlavaOnevisionProcessor
    
    device = 'cuda'
    n_frame_tokens = 196
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    
    init_prompt = '<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user '
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    
    if granularity_topks is None:
        weights = [1.0 / gran for gran in granularities]
        total_weight = sum(weights)
        granularity_topks = [int(topk * w / total_weight) for w in weights]
        diff = topk - sum(granularity_topks)
        granularity_topks[0] += diff
    
    assert sum(granularity_topks) == topk, \
        f"granularity_topks sum {sum(granularity_topks)} != topk {topk}"
    
    keep_ratios = {
        49: keep_ratio_49,
        196: keep_ratio_196,
        784: keep_ratio_784
    }
    
    attention_weight = {
        49: attention_weight_49,
        196: attention_weight_196,
        784: attention_weight_784
    }
    
    fft_weight = {
        49: 1.0 - attention_weight_49,
        196: 1.0 - attention_weight_196,
        784: 1.0 - attention_weight_784
    }
    
    logger.info("MuKV rerank configuration:")
    logger.info(f"  Granularities: {granularities}")
    logger.info(f"  Granularity TopKs: {granularity_topks}")
    logger.info(f"  Total TopK: {topk}")
    logger.info(f"  Token Compression: {enable_compression}")
    logger.info(f"  Keep Ratios: {keep_ratios}")
    logger.info(f"  Importance Method: {importance_method}")
    logger.info(f"  FFT Enabled: {fft_enabled}")
    logger.info(f"  FFT Method: {fft_method}")
    logger.info(f"  Attention Weights: {attention_weight}")
    logger.info(f"  FFT Weights: {fft_weight}")
    logger.info(f"  Attention Layer Strategy: {attention_layer_strategy}")
    logger.info(f"  Use Unified Layer: {use_unified_layer}")
    if enable_compression:
        logger.info(f"  Dedup: {dedup_enabled}, threshold={dedup_threshold}")
    logger.info(f"  🆕 Rerank Enabled: {enable_rerank}")
    if enable_rerank:
        logger.info(f"  🆕 Rerank Alpha: {rerank_alpha}")
        logger.info(f"  🆕 Rerank Beta: {rerank_beta}")
        logger.info(f"  🆕 Rerank Top-N: {rerank_top_n}")
        logger.info(f"  🔄 Rerank Reverse: {rerank_reverse}")
    
    model = MuKVRerankModel.from_pretrained(
        model_path, 
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16,
        processor=processor,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        n_local=n_local,
        topk=topk,
        chunk_size=chunk_size,
        granularities=granularities,
        granularity_topks=granularity_topks,
        enable_compression=enable_compression,
        keep_ratios=keep_ratios,
        importance_method=importance_method,
        dedup_threshold=dedup_threshold,
        dedup_enabled=dedup_enabled,
        fft_enabled=fft_enabled,
        fft_method=fft_method,
        attention_weight=attention_weight,
        fft_weight=fft_weight,
        attention_layer_strategy=attention_layer_strategy,
        use_unified_layer=use_unified_layer,
        enable_rerank=enable_rerank,
        rerank_alpha=rerank_alpha,
        rerank_beta=rerank_beta,
        rerank_top_n=rerank_top_n,
        rerank_reverse=rerank_reverse,
    )
    
    logger.info(f'MuKV rerank model loaded with {len(granularities)} granularities')
    model.eval()

    return model, processor
