"""
MuKV multi-grained KV cache module.
基于多粒度 KV-Cache 的 MuKV 基础模块。
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from logzero import logger
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration

from model.abstract_mukv import Abstract_MuKV


class MuKVMultiGrainKVCache:
    """MuKV 的多粒度 KV-Cache 存储器。"""
    
    def __init__(self, granularities: List[int], device: str = 'cuda'):
        """
        Args:
            granularities: 粒度列表，如[49, 196, 784]
        """
        self.granularities = granularities
        self.device = device
        
        # 为每个粒度创建独立的KV-Cache存储
        self.kv_stores = {
            gran: {
                'keys': [],           # List[List[(k, v)]] - 所有blocks的KV
                'repr_vectors': [],   # List[torch.Tensor] - 代表向量
                'token_indices': [],  # List[torch.Tensor] - 保留的token位置
                'compression_stats': []  # List[dict] - 压缩统计
            }
            for gran in granularities
        }
        
    def store_kv_cache(
        self, 
        granularity: int, 
        kv_layers: List[Tuple[torch.Tensor, torch.Tensor]], 
        repr_vector: torch.Tensor,
        token_indices: torch.Tensor,
        compression_info: dict
    ):
        """
        存储压缩后的KV-Cache
        
        Args:
            granularity: 粒度大小
            kv_layers: List[(k, v)] 所有层的压缩后KV
            repr_vector: (hidden_dim,) 代表向量
            token_indices: (n_kept,) 保留的token位置索引
            compression_info: 压缩统计信息
        """
        # 移到CPU节省GPU内存
        kv_layers_cpu = [(k.detach().cpu(), v.detach().cpu()) for k, v in kv_layers]
        repr_cpu = repr_vector.detach().cpu()
        indices_cpu = token_indices.detach().cpu()
        
        self.kv_stores[granularity]['keys'].append(kv_layers_cpu)
        self.kv_stores[granularity]['repr_vectors'].append(repr_cpu)
        self.kv_stores[granularity]['token_indices'].append(indices_cpu)
        self.kv_stores[granularity]['compression_stats'].append(compression_info)
        
    def retrieve_kv_cache(
        self, 
        granularity: int, 
        query_vector: torch.Tensor, 
        topk: int
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        从指定粒度检索最相关的KV-Cache
        
        Args:
            granularity: 粒度
            query_vector: (hidden_dim,) 查询向量
            topk: 检索数量
            
        Returns:
            selected_kv_layers: List[List[(k, v)]] 检索到的KV列表
        """
        store = self.kv_stores[granularity]
        if len(store['repr_vectors']) == 0:
            return []
            
        # 计算相似度
        repr_vectors = torch.stack(store['repr_vectors']).to(self.device)
        query_vector_gpu = query_vector.to(self.device)
        similarities = torch.cosine_similarity(
            query_vector_gpu.unsqueeze(0), repr_vectors, dim=1
        )
        
        # TopK选择
        actual_topk = min(topk, len(similarities))
        top_indices = similarities.topk(actual_topk).indices.cpu().tolist()
        
        # 获取对应的KV-Cache并移回GPU
        selected_kv_layers = []
        for i in top_indices:
            kv_layers_cpu = store['keys'][i]
            kv_layers_gpu = [(k.to(self.device), v.to(self.device)) for k, v in kv_layers_cpu]
            selected_kv_layers.append(kv_layers_gpu)
        
        logger.debug(f"Granularity {granularity}: retrieved {len(selected_kv_layers)} blocks, "
                    f"max_sim={similarities.max().item():.3f}")
        
        return selected_kv_layers
    
    def get_stats(self):
        """获取存储统计信息"""
        stats = {}
        for gran in self.granularities:
            n_blocks = len(self.kv_stores[gran]['keys'])
            if n_blocks > 0:
                avg_tokens = np.mean([
                    len(indices) for indices in self.kv_stores[gran]['token_indices']
                ])
                avg_ratio = avg_tokens / gran if gran > 0 else 0
            else:
                avg_tokens = 0
                avg_ratio = 0
                
            stats[gran] = {
                'n_blocks': n_blocks,
                'avg_tokens_per_block': avg_tokens,
                'avg_keep_ratio': avg_ratio
            }
        return stats


class MuKVMultiGrainKVCacheModel(LlavaOnevisionForConditionalGeneration, Abstract_MuKV):
    """MuKV 的多粒度 KV cache 模型。"""
    
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
        # Token压缩相关参数
        enable_compression: bool = True,
        keep_ratios: Optional[Dict[int, float]] = None,
        importance_method: str = 'last_layer',
        dedup_threshold: float = 0.90,
        dedup_enabled: bool = True
    ):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)
        Abstract_MuKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size)
        
        self.granularities = granularities
        self.granularity_topks = granularity_topks
        
        # Token压缩配置
        self.enable_compression = enable_compression
        self.importance_method = importance_method
        self.dedup_threshold = dedup_threshold
        self.dedup_enabled = dedup_enabled
        
        # 默认保留率：细粒度保留更多
        if keep_ratios is None:
            self.keep_ratios = {
                49: 0.70,
                196: 0.50,
                784: 0.30
            }
        else:
            self.keep_ratios = keep_ratios
        
        # 多粒度KV存储器
        self.multigran_kv_store = MuKVMultiGrainKVCache(granularities, self.device)
        
        # 初始提示的KV-Cache
        self.init_kv_cache = None
        
        logger.info(f"Token压缩配置: enabled={enable_compression}")
        logger.info(f"保留率: {self.keep_ratios}")
        logger.info(f"去冗余: enabled={dedup_enabled}, threshold={dedup_threshold}")
        
    def get_prompt(self, query, mc=False):
        prompt = f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt
    
    def clear_cache(self):
        """清空所有缓存"""
        self.multigran_kv_store = MuKVMultiGrainKVCache(self.granularities, self.device)
        self.init_kv_cache = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    def calc_kv_cache_memory(self) -> float:
        """
        计算KV Cache占用的CPU内存(字节)
        
        Returns:
            memory_bytes: KV Cache总大小(bytes)
        """
        total_bytes = 0
        
        for gran in self.granularities:
            store = self.multigran_kv_store.kv_stores[gran]
            
            # 遍历所有存储的KV blocks
            for kv_layers_cpu in store['keys']:
                for k, v in kv_layers_cpu:
                    total_bytes += k.numel() * k.element_size()  # K的字节数
                    total_bytes += v.numel() * v.element_size()  # V的字节数
            
            # 加上repr_vectors的内存
            for repr_vec in store['repr_vectors']:
                total_bytes += repr_vec.numel() * repr_vec.element_size()
        
        return total_bytes
    
    @torch.inference_mode()
    def encode_init_prompt(self):
        """编码初始提示"""
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor([self.init_prompt_ids], device=self.device)
        
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.init_kv_cache = output.past_key_values
        logger.debug("Initial prompt encoded")
    
    def _get_video_features(self, pixel_values_videos):
        """获取视频特征"""
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)
        video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
        selected_video_feature = video_features.hidden_states[self.config.vision_feature_layer]

        if self.config.vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature
        video_features = self.multi_modal_projector(selected_video_feature)

        video_features = self.apply_pooling(video_features)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)
        return video_features
    
    def _compute_token_importance(
        self, 
        attentions: Tuple[torch.Tensor], 
        method: str = 'last_layer'
    ) -> torch.Tensor:
        """
        思路1：计算token重要性（只计算新tokens，不包含init部分）
        
        Args:
            attentions: tuple of (n_heads, seq_len, seq_len)
                       seq_len = init_len + granularity
            method: 'last_layer' | 'multi_layer' | 'all_layers'
            
        Returns:
            importance_scores: (granularity,) 只有新tokens的重要性分数
        """
        # 获取init长度
        init_len = self.init_kv_cache[0][0].shape[2]
        
        if method == 'last_layer':
            # 最后一层的attention
            last_attn = attentions[-1]  # (n_heads, seq_len, seq_len)
            
            # 🔍 Step 6: 详细分析attention的结构（调试用，已验证正常）
            # logger.debug(f"[STEP6] _compute_token_importance:")
            # logger.debug(f"  n_layers: {len(attentions)}")
            # logger.debug(f"  last_attn shape: {last_attn.shape}")
            # logger.debug(f"  init_len: {init_len}")
            
            # 🔑 核心修复：正确计算每个key token被关注的总量
            # 1. 去batch维度：mean(dim=0) → (n_heads, query_len, key_len)
            # 2. 对heads平均：mean(dim=0) → (query_len, key_len)
            # 3. 对query求和：sum(dim=0) → (key_len,) ✅ 1维向量
            importance = last_attn.mean(dim=0).mean(dim=0).sum(dim=0)  # (key_len,)
            
            # logger.debug(f"  importance (full) shape: {importance.shape}")
            
            # 只返回新tokens的importance
            importance = importance[init_len:]  # (granularity,)
            
            # logger.debug(f"  importance (sliced [{init_len}:]) shape: {importance.shape}")
            
        elif method == 'multi_layer':
            # 最后5层加权平均
            weights = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3], device=attentions[0].device)
            importance = torch.zeros(attentions[0].shape[-1], device=attentions[0].device)
            n_layers = min(5, len(attentions))
            for i in range(n_layers):
                layer_attn = attentions[-(n_layers-i)]
                # 同样的修复：正确聚合attention
                importance += weights[i] * layer_attn.mean(dim=0).mean(dim=0).sum(dim=0)
            # 只返回新tokens的importance
            importance = importance[init_len:]  # (granularity,)
        
        elif method == 'all_layers':
            # 所有层累积
            importance = torch.zeros(attentions[0].shape[-1], device=attentions[0].device)
            for attn in attentions:
                # 同样的修复：正确聚合attention
                importance += attn.mean(dim=0).mean(dim=0).sum(dim=0)
            importance = importance / len(attentions)
            # 只返回新tokens的importance
            importance = importance[init_len:]  # (granularity,)
        
        return importance  # (granularity,) - 与full_kvs的长度对齐
    
    def _get_keep_ratio(self, granularity: int) -> float:
        """
        思路2：获取粒度特定的保留率
        
        Args:
            granularity: 粒度大小
            
        Returns:
            keep_ratio: 保留率 (0.0-1.0)
        """
        return self.keep_ratios.get(granularity, 0.5)
    
    def _select_tokens_with_dedup(
        self,
        full_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
        importance_scores: torch.Tensor,
        target_k: int,
        similarity_threshold: float
    ) -> torch.Tensor:
        """
        思路3：去冗余选择tokens
        
        Args:
            full_kvs: 所有层的完整KV
            importance_scores: (seq_len,) token重要性
            target_k: 目标保留数量
            similarity_threshold: 相似度阈值
            
        Returns:
            final_indices: 最终保留的token索引
        """
        seq_len = importance_scores.shape[0]
        
        # 先选1.2倍候选（为去重留余地）
        candidate_k = min(int(target_k * 1.2), seq_len)
        candidate_indices = importance_scores.topk(candidate_k).indices
        
        # 🆕 确保类型和内存连续性
        candidate_indices = candidate_indices.long().contiguous()
        
        if not self.dedup_enabled or candidate_k <= target_k:
            return candidate_indices[:target_k].contiguous()
        
        # 提取最后一层的K向量
        last_layer_k = full_kvs[-1][0]  # (batch, n_heads, seq_len, dim)
        last_layer_k = last_layer_k.mean(dim=1).squeeze(0)  # (seq_len, dim)
        candidate_k_vectors = last_layer_k[candidate_indices]  # (candidate_k, dim)
        
        # 计算相似度矩阵（手动计算，避免torch.cosine_similarity的维度问题）
        # 归一化向量
        candidate_k_norm = candidate_k_vectors / candidate_k_vectors.norm(dim=1, keepdim=True)
        # 矩阵乘法得到余弦相似度
        sim_matrix = torch.mm(candidate_k_norm, candidate_k_norm.t())  # (candidate_k, candidate_k)
        
        # 去除冗余tokens
        keep_mask = torch.ones(candidate_k, dtype=torch.bool, device=sim_matrix.device)
        
        for i in range(candidate_k):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, candidate_k):
                if sim_matrix[i, j] > similarity_threshold:
                    # 保留importance更高的
                    idx_i = candidate_indices[i]
                    idx_j = candidate_indices[j]
                    if importance_scores[idx_i] < importance_scores[idx_j]:
                        keep_mask[i] = False
                        break
                    else:
                        keep_mask[j] = False
        
        # 最终保留的indices
        kept_indices = candidate_indices[keep_mask]
        
        # 确保数量不超过target_k
        if len(kept_indices) > target_k:
            # 从kept中再选top-k（按importance）
            kept_importance = importance_scores[kept_indices]
            final_positions = kept_importance.topk(target_k).indices
            final_indices = kept_indices[final_positions]
        else:
            final_indices = kept_indices
        
        return final_indices
    
    def _encode_granularity_chunk_with_compression(
        self, 
        visual_features: torch.Tensor, 
        granularity: int
    ):
        """
        为指定粒度编码并压缩
        
        Args:
            visual_features: (batch, n_tokens, hidden_dim)
            granularity: 粒度大小
        """
        batch_size, n_tokens, hidden_dim = visual_features.shape
        
        # 按粒度分块
        n_blocks = n_tokens // granularity
        if n_blocks == 0:
            return
            
        # 重塑为blocks
        blocks = visual_features[:, :n_blocks * granularity].reshape(
            batch_size, n_blocks, granularity, hidden_dim
        )
        
        # 🔍 调查日志: 记录初始状态（调试用，已验证正常）
        # original_init_len = self.init_kv_cache[0][0].shape[2]
        # original_init_id = id(self.init_kv_cache[0][0])
        # logger.info(f"\n{'='*60}")
        # logger.info(f"[INVESTIGATION] 开始编码粒度 {granularity}, {n_blocks} blocks")
        # logger.info(f"[INVESTIGATION] Init KV: length={original_init_len}, id={original_init_id}")
        # logger.info(f"{'='*60}")
        
        # 为每个block生成压缩的KV-Cache
        for block_idx in range(n_blocks):
            block_features = blocks[:, block_idx]  # (batch, granularity, hidden_dim)
            
            # 🔍 Step 1: 检查forward前的init_kv状态（调试用，已验证正常）
            # if block_idx in [0, 1, 2, n_blocks//2, n_blocks-1]:
            #     current_init_len = self.init_kv_cache[0][0].shape[2]
            #     current_init_id = id(self.init_kv_cache[0][0])
            #     logger.info(f"\n[STEP1] Block {block_idx}/{n_blocks}, Gran={granularity}")
            #     logger.info(f"  Before forward:")
            #     logger.info(f"    init_kv length: {current_init_len} (expected: {original_init_len})")
            #     logger.info(f"    init_kv id: {current_init_id} (expected: {original_init_id})")
            #     logger.info(f"    init_kv changed: {current_init_len != original_init_len}")
            #     logger.info(f"    block_features shape: {block_features.shape}")
            
            # Forward获取KV和Attention
            output = self.language_model(
                inputs_embeds=block_features, 
                past_key_values=self.init_kv_cache,
                use_cache=True, 
                return_dict=True,
                output_attentions=True if self.enable_compression else False
            )
            
            # 🔍 Step 2 & 3: 检查output的past_kv和attention长度（调试用，已验证正常）
            # if block_idx in [0, 1, 2, n_blocks//2, n_blocks-1]:
            #     returned_kv_len = output.past_key_values[0][0].shape[2]
            #     returned_kv_id = id(output.past_key_values[0][0])
            #     
            #     logger.info(f"  After forward:")
            #     logger.info(f"    output.past_kv length: {returned_kv_len}")
            #     logger.info(f"    output.past_kv id: {returned_kv_id}")
            #     logger.info(f"    Expected KV length: {original_init_len + granularity}")
            #     logger.info(f"    KV length diff: {returned_kv_len - (original_init_len + granularity)}")
            #     
            #     if self.enable_compression:
            #         attn_seq_len = output.attentions[-1].shape[-1]
            #         logger.info(f"    attention[-1] shape: {output.attentions[-1].shape}")
            #         logger.info(f"    attention seq_len: {attn_seq_len}")
            #         logger.info(f"    Expected seq_len: {original_init_len + granularity}")
            #         logger.info(f"    Attention extra tokens: {attn_seq_len - (original_init_len + granularity)}")
            #         
            #         # 🔍 累积假设验证
            #         if block_idx > 0:
            #             predicted_cumulative = original_init_len + granularity * (block_idx + 1)
            #             logger.info(f"    Predicted (cumulative): {predicted_cumulative}")
            #             logger.info(f"    Matches cumulative: {abs(attn_seq_len - predicted_cumulative) < 10}")
            #     
            #     # 🔍 检查init_kv是否被language_model修改
            #     after_init_len = self.init_kv_cache[0][0].shape[2]
            #     after_init_id = id(self.init_kv_cache[0][0])
            #     logger.info(f"    init_kv after LM:")
            #     logger.info(f"      length: {after_init_len} (changed: {after_init_len != original_init_len})")
            #     logger.info(f"      id: {after_init_id} (changed: {after_init_id != original_init_id})")
            #     
            #     if after_init_len != original_init_len:
            #         logger.error(f"    ⚠️⚠️⚠️  FOUND IT! init_kv_cache被language_model修改了!")
            #         logger.error(f"    ⚠️⚠️⚠️  从 {original_init_len} → {after_init_len}")
            
            # 提取完整KV
            init_len = self.init_kv_cache[0][0].shape[2]
            full_kvs = []
            for k, v in output.past_key_values:
                new_k = k[:, :, init_len:, :]
                new_v = v[:, :, init_len:, :]
                full_kvs.append((new_k, new_v))
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Token级压缩
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if self.enable_compression:
                # 🔑 获取实际的KV序列长度（可能与granularity不同）
                actual_seq_len = full_kvs[0][0].shape[2]
                
                # 🔍 Step 4: 详细分析attention和KV的关系（调试用，已验证正常）
                # if block_idx in [0, 1, 2, n_blocks//2, n_blocks-1]:
                #     logger.info(f"\n[STEP4] Token压缩分析 - Block {block_idx}")
                #     logger.info(f"  full_kvs[0][0] shape: {full_kvs[0][0].shape}")
                #     logger.info(f"  actual_seq_len (from KV): {actual_seq_len}")
                #     logger.info(f"  Expected seq_len: {granularity}")
                #     logger.info(f"  KV seq_len diff: {actual_seq_len - granularity}")
                
                # 思路1: 计算重要性
                importance_scores = self._compute_token_importance(
                    output.attentions, method=self.importance_method
                )
                
                # 🔍 Step 5: 分析importance计算的详细信息（调试用，已验证正常）
                # if block_idx in [0, 1, 2, n_blocks//2, n_blocks-1]:
                #     logger.info(f"  importance_scores shape (raw): {importance_scores.shape}")
                #     logger.info(f"  actual_seq_len: {actual_seq_len}")
                #     logger.info(f"  Mismatch: {importance_scores.shape[0] != actual_seq_len}")
                #     if importance_scores.shape[0] != actual_seq_len:
                #         ratio = importance_scores.shape[0] / actual_seq_len
                #         logger.info(f"  Ratio (importance/actual): {ratio:.2f}")
                #         logger.info(f"  Ratio ≈ block_idx+1: {abs(ratio - (block_idx+1)) < 0.5}")
                
                # 🆕 确保importance是1维tensor
                if importance_scores.dim() > 1:
                    importance_scores = importance_scores.squeeze()
                    if importance_scores.dim() > 1:
                        importance_scores = importance_scores.flatten()
                
                # 🆕 确保importance长度与actual_seq_len一致
                if importance_scores.shape[0] != actual_seq_len:
                    # 🔍 Step 7: 分析mismatch的详细信息（调试用，已验证正常）
                    # if block_idx in [0, 1, 2, n_blocks//2, n_blocks-1]:
                    #     logger.info(f"\n[STEP7] Importance Mismatch 详细分析:")
                    #     logger.info(f"  Block {block_idx}, Gran={granularity}")
                    #     logger.info(f"  importance shape: {importance_scores.shape[0]}")
                    #     logger.info(f"  actual_seq_len: {actual_seq_len}")
                    #     logger.info(f"  Ratio: {importance_scores.shape[0] / actual_seq_len:.2f}")
                    #     logger.info(f"  Blocks encoded so far: {block_idx + 1}")
                    #     logger.info(f"  If cumulative: {original_init_len} + {granularity} * {block_idx + 1} = {original_init_len + granularity * (block_idx + 1)}")
                    #     logger.info(f"  Matches (init + cumulative): {abs(importance_scores.shape[0] - (original_init_len + granularity * (block_idx + 1))) < 10}")
                    
                    logger.warning(f"Importance length mismatch: {importance_scores.shape[0]} vs {actual_seq_len}, gran={granularity}")
                    if importance_scores.shape[0] > actual_seq_len:
                        # 🔑 从末尾截断，确保是当前block的importance
                        importance_scores = importance_scores[-actual_seq_len:]
                        # logger.debug(f"Truncated from end to align with current block tokens")
                    else:
                        # 填充（用最小值）
                        padding_len = actual_seq_len - importance_scores.shape[0]
                        min_importance = importance_scores.min().item()
                        # 🔑 确保padding是1维
                        padding = torch.full((padding_len,), min_importance, 
                                            device=importance_scores.device,
                                            dtype=importance_scores.dtype)
                        importance_scores = torch.cat([importance_scores, padding])
                
                # 思路2: 粒度特定保留率（基于actual_seq_len）
                keep_ratio = self._get_keep_ratio(granularity)
                target_k = int(actual_seq_len * keep_ratio)  # 🆕 使用actual_seq_len
                
                # 思路3: 去冗余选择
                final_indices = self._select_tokens_with_dedup(
                    full_kvs, importance_scores, target_k, self.dedup_threshold
                )
                
                # 确保indices在正确的设备上
                final_indices = final_indices.to(full_kvs[0][0].device)
                
                # 🆕 验证indices范围（防御性检查）
                seq_len = full_kvs[0][0].shape[2]
                max_idx = final_indices.max().item() if len(final_indices) > 0 else -1
                
                if max_idx >= seq_len:
                    logger.warning(f"Index overflow: max={max_idx}, seq_len={seq_len}, gran={granularity}")
                    # 过滤越界的indices
                    final_indices = final_indices[final_indices < seq_len]
                
                if len(final_indices) == 0:
                    logger.warning(f"No valid indices after filtering, using all tokens")
                    final_indices = torch.arange(seq_len, device=full_kvs[0][0].device, dtype=torch.long)
                
                # 排序indices以保持时序
                final_indices, _ = torch.sort(final_indices)
                
                # 🆕 逐层提取KV（每层单独验证）
                compressed_kvs = []
                for layer_idx, (k, v) in enumerate(full_kvs):
                    layer_seq_len = k.shape[2]
                    
                    # 确保indices不超过当前层的长度
                    layer_valid_indices = final_indices[final_indices < layer_seq_len]
                    
                    if len(layer_valid_indices) == 0:
                        logger.warning(f"Layer {layer_idx}: no valid indices, using all")
                        layer_valid_indices = torch.arange(layer_seq_len, device=k.device, dtype=torch.long)
                    
                    compressed_kvs.append((
                        k[:, :, layer_valid_indices, :],
                        v[:, :, layer_valid_indices, :]
                    ))
                
                # 压缩统计
                compression_info = {
                    'original_tokens': granularity,
                    'kept_tokens': len(final_indices),
                    'compression_ratio': len(final_indices) / granularity,
                    'importance_mean': importance_scores.mean().item(),
                    'importance_std': importance_scores.std().item()
                }
                
                # 用压缩后的KV计算代表向量
                last_layer_k = compressed_kvs[-1][0]
                repr_vector = last_layer_k.mean(dim=2)
                repr_vector = repr_vector.reshape(batch_size, -1).squeeze(0)
                
                # 存储压缩后的KV
                self.multigran_kv_store.store_kv_cache(
                    granularity=granularity,
                    kv_layers=compressed_kvs,
                    repr_vector=repr_vector,
                    token_indices=final_indices,
                    compression_info=compression_info
                )
                
                # 🔍 Step 8: 检查存储后init_kv的状态（调试用，已验证正常）
                # if block_idx in [0, 1, 2, n_blocks//2, n_blocks-1]:
                #     final_init_len = self.init_kv_cache[0][0].shape[2]
                #     final_init_id = id(self.init_kv_cache[0][0])
                #     logger.info(f"  After storage:")
                #     logger.info(f"    init_kv length: {final_init_len}")
                #     logger.info(f"    init_kv id: {final_init_id}")
                #     logger.info(f"    Changed in this iteration: {final_init_len != current_init_len}")
                #     logger.info(f"{'='*60}\n")
                
            else:
                # 不压缩，直接存储
                last_layer_k = full_kvs[-1][0]
                repr_vector = last_layer_k.mean(dim=2).reshape(batch_size, -1).squeeze(0)
                
                all_indices = torch.arange(granularity, device=block_features.device)
                compression_info = {
                    'original_tokens': granularity,
                    'kept_tokens': granularity,
                    'compression_ratio': 1.0
                }
                
                self.multigran_kv_store.store_kv_cache(
                    granularity=granularity,
                    kv_layers=full_kvs,
                    repr_vector=repr_vector,
                    token_indices=all_indices,
                    compression_info=compression_info
                )
        
        # 🔍 Step 9: 粒度编码完成后的最终检查（调试用，已验证正常）
        # final_init_len = self.init_kv_cache[0][0].shape[2]
        # final_init_id = id(self.init_kv_cache[0][0])
        # logger.info(f"\n{'='*60}")
        # logger.info(f"[INVESTIGATION] 粒度 {granularity} 编码完成")
        # logger.info(f"[INVESTIGATION] 最终 Init KV: length={final_init_len}, id={final_init_id}")
        # logger.info(f"[INVESTIGATION] Init KV 总变化: length {original_init_len}→{final_init_len}, id_changed={final_init_id != original_init_id}")
        # if final_init_len != original_init_len:
        #     logger.error(f"[BUG CONFIRMED] init_kv_cache在编码{n_blocks}个blocks后从{original_init_len}增长到{final_init_len}!")
        #     logger.error(f"[BUG CONFIRMED] 增加了{final_init_len - original_init_len}个tokens!")
        # logger.info(f"{'='*60}\n")
    
    @torch.inference_mode() 
    def encode_video(self, video, encode_chunk_size=64):
        """
        多粒度视频编码（带Token压缩）
        
        Args:
            video: (n_frames, H, W, 3)
        """
        num_frames = video.shape[0]
        num_chunks = num_frames // encode_chunk_size
        
        print(f"\n{'='*60}")
        print(f"[Video Encoding with Token Compression]")
        print(f"Frames: {num_frames}, Chunks: {num_chunks}")
        print(f"Granularities: {self.granularities}")
        print(f"Compression: {self.enable_compression}")
        print(f"{'='*60}")
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]
            
            # 获取视觉特征
            pixel_values = self.processor.video_processor(chunk_video, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)
            visual_features = self._get_video_features(pixel_values)
            
            # 为每个粒度独立编码（带压缩）
            print(f"Chunk {chunk_idx+1}/{num_chunks}: ", end="", flush=True)
            
            # 🔍 Step 10: 检查粒度间的相互影响（调试用，已验证正常）
            # init_kv_before_granularities = self.init_kv_cache[0][0].shape[2]
            # logger.info(f"\n[STEP10] Chunk {chunk_idx+1}, 开始编码3个粒度")
            # logger.info(f"[STEP10] Init KV length at start: {init_kv_before_granularities}")
            
            for granularity in self.granularities:
                # init_before_this_gran = self.init_kv_cache[0][0].shape[2]
                # logger.info(f"\n[STEP10] 粒度{gran_idx+1}/3: {granularity}")
                # logger.info(f"[STEP10] Init KV before this granularity: {init_before_this_gran}")
                
                self._encode_granularity_chunk_with_compression(visual_features, granularity)
                
                # init_after_this_gran = self.init_kv_cache[0][0].shape[2]
                # logger.info(f"[STEP10] Init KV after this granularity: {init_after_this_gran}")
                # logger.info(f"[STEP10] Changed by granularity {granularity}: {init_after_this_gran - init_before_this_gran}")
                
                # if init_after_this_gran != init_before_this_gran:
                #     logger.error(f"[CROSS-GRAN BUG] 粒度{granularity}修改了init_kv!")
                #     logger.error(f"[CROSS-GRAN BUG] {init_before_this_gran} → {init_after_this_gran}")
            
            # init_kv_after_granularities = self.init_kv_cache[0][0].shape[2]
            # logger.info(f"\n[STEP10] Chunk {chunk_idx+1} 完成3个粒度")
            # logger.info(f"[STEP10] Init KV: {init_kv_before_granularities} → {init_kv_after_granularities}")
            # logger.info(f"[STEP10] Total change: {init_kv_after_granularities - init_kv_before_granularities}\n")
            
            stats = self.multigran_kv_store.get_stats()
            print(f"Blocks: P={stats[49]['n_blocks']}, F={stats[196]['n_blocks']}, S={stats[784]['n_blocks']}")
        
        # 处理剩余帧
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            remaining_video = video[start_idx:start_idx + remaining_frames]
            
            pixel_values = self.processor.video_processor(remaining_video, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)
            visual_features = self._get_video_features(pixel_values)
            
            for granularity in self.granularities:
                self._encode_granularity_chunk_with_compression(visual_features, granularity)
        
        # 最终统计
        final_stats = self.multigran_kv_store.get_stats()
        print(f"{'='*60}")
        print(f"✅ Encoding Done!")
        for gran in self.granularities:
            s = final_stats[gran]
            print(f"  Gran {gran:3d}: {s['n_blocks']:4d} blocks, "
                  f"avg {s['avg_tokens_per_block']:.1f} tokens/block "
                  f"(ratio: {s['avg_keep_ratio']:.2f})")
        print(f"{'='*60}\n")
    
    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        """
        多粒度问答（使用压缩的KV-Cache）
        """
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]
        
        # 1. 编码问题获取query向量
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
        
        # 提取query向量
        if question_output.past_key_values is not None and len(question_output.past_key_values) > 0:
            last_layer_kv = question_output.past_key_values[-1]
            last_k = last_layer_kv[0][:, :, -1, :]
            query_vector = last_k.reshape(last_k.size(0), -1).squeeze(0)
        else:
            raise RuntimeError("Failed to get past_key_values")
        
        # 2. 从每个粒度检索
        print(f"\n{'='*60}")
        print(f"[Multi-Granularity Retrieval]")
        print(f"Query vector shape: {query_vector.shape}")
        print(f"{'='*60}")
        
        all_retrieved_kvs = []
        retrieval_stats = {}
        
        for gran_idx, granularity in enumerate(self.granularities):
            topk_for_gran = self.granularity_topks[gran_idx]
            
            selected_kv_layers = self.multigran_kv_store.retrieve_kv_cache(
                granularity=granularity,
                query_vector=query_vector,
                topk=topk_for_gran
            )
            
            for kv_layers in selected_kv_layers:
                all_retrieved_kvs.append(kv_layers)
            
            retrieval_stats[granularity] = len(selected_kv_layers)
            print(f"  Gran {granularity:3d}: {len(selected_kv_layers):2d} blocks")
        
        print(f"✅ Total: {sum(retrieval_stats.values())} blocks retrieved")
        print(f"{'='*60}\n")
        
        # 3. 拼接KV-Cache
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
        
        # 4. 生成答案
        print(f"[Generation] Max {max_new_tokens} tokens...", flush=True)
        output_ids = []
        past_key_values = combined_kv
        
        for i in range(max_new_tokens):
            if i == 0:  # prefill
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
            else:  # decoding
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
    importance_method='last_layer',
    dedup_threshold=0.90,
    dedup_enabled=True
):
    """
    加载带Token压缩的多粒度MuKV模型
    
    Args:
        granularities: 粒度列表
        granularity_topks: 每个粒度检索的数量
        enable_compression: 是否启用Token压缩
        keep_ratio_*: 各粒度的保留率
        importance_method: 重要性计算方法
        dedup_threshold: 去冗余阈值
        dedup_enabled: 是否启用去冗余
    """
    device = 'cuda'
    n_frame_tokens = 196
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    
    init_prompt = '<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user '
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    
    # 默认的粒度检索数量分配
    if granularity_topks is None:
        weights = [1.0 / gran for gran in granularities]
        total_weight = sum(weights)
        granularity_topks = [int(topk * w / total_weight) for w in weights]
        diff = topk - sum(granularity_topks)
        granularity_topks[0] += diff
    
    assert sum(granularity_topks) == topk, \
        f"granularity_topks sum {sum(granularity_topks)} != topk {topk}"
    
    # 保留率配置
    keep_ratios = {
        49: keep_ratio_49,
        196: keep_ratio_196,
        784: keep_ratio_784
    }
    
    logger.info("MuKV multi-grained KV cache configuration:")
    logger.info(f"  Granularities: {granularities}")
    logger.info(f"  Granularity TopKs: {granularity_topks}")
    logger.info(f"  Total TopK: {topk}")
    logger.info(f"  Token Compression: {enable_compression}")
    if enable_compression:
        logger.info(f"  Keep Ratios: {keep_ratios}")
        logger.info(f"  Importance Method: {importance_method}")
        logger.info(f"  Dedup: {dedup_enabled}, threshold={dedup_threshold}")
    
    model = MuKVMultiGrainKVCacheModel.from_pretrained(
        model_path, 
        device_map={"": "cuda:0"},  # 强制单GPU，避免multi-GPU设备不匹配
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
    )
    
    logger.info(f'MuKV multi-grained KV cache model loaded with {len(granularities)} granularities')
    model.eval()

    return model, processor
