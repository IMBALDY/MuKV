"""
MuKV dual-signal compression module.
基于注意力和频域信号的 MuKV 双信号压缩模块。
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from logzero import logger
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration

from model.abstract_mukv import Abstract_MuKV
from model.mukv_multigrain_kvcache import MuKVMultiGrainKVCache


class MuKVDualSignalCompressionModel(LlavaOnevisionForConditionalGeneration, Abstract_MuKV):
    """MuKV 的双信号压缩模型。"""
    
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
        importance_method: str = 'attention_fft_weighted',
        dedup_threshold: float = 0.90,
        dedup_enabled: bool = True,
        # 🆕 FFT相关参数
        fft_enabled: bool = True,
        fft_method: str = 'diff',  # 'diff' or 'fft' or 'spectral_entropy'
        attention_weight: Optional[Dict[int, float]] = None,  # 粒度特定的attention权重
        fft_weight: Optional[Dict[int, float]] = None,  # 粒度特定的fft权重
        attention_layer_strategy: str = 'last',  # 'last', 'second_last', 'last_5', 'middle'
        use_unified_layer: bool = False,  # 🆕 是否让FFT和Query使用与Attention相同的层
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
        
        # 🆕 FFT配置
        self.fft_enabled = fft_enabled
        self.fft_method = fft_method
        
        # 默认保留率
        if keep_ratios is None:
            self.keep_ratios = {
                49: 0.70,
                196: 0.50,
                784: 0.30
            }
        else:
            self.keep_ratios = keep_ratios
        
        # 🆕 粒度自适应权重
        if attention_weight is None:
            self.attention_weight = {
                49:  0.5,  # 细粒度: 动态和语义同等重要
                196: 0.7,  # 中粒度: 语义为主
                784: 0.8,  # 粗粒度: 强调语义连贯性
            }
        else:
            self.attention_weight = attention_weight
        
        if fft_weight is None:
            self.fft_weight = {
                49:  0.5,
                196: 0.3,
                784: 0.2,
            }
        else:
            self.fft_weight = fft_weight
        
        # 多粒度KV存储器
        self.multigran_kv_store = MuKVMultiGrainKVCache(granularities, self.device)
        self.init_kv_cache = None
        
        # 🆕 Attention层策略
        self.attention_layer_strategy = attention_layer_strategy
        
        # 🆕 统一层策略：让FFT和Query使用与Attention相同的层
        self.use_unified_layer = use_unified_layer
        
        # 🆕 保存最后的attention和FFT scores（用于可视化）
        self.last_attention_scores = {}
        self.last_fft_scores = {}
        self.save_scores_for_viz = False  # 默认关闭，需要时开启
        
        logger.info("MuKV dual-signal compression configuration:")
        logger.info(f"  Compression: {enable_compression}")
        logger.info(f"  Keep Ratios: {self.keep_ratios}")
        logger.info(f"  FFT Enabled: {fft_enabled}")
        logger.info(f"  FFT Method: {fft_method}")
        logger.info(f"  Attention Weights: {self.attention_weight}")
        logger.info(f"  FFT Weights: {self.fft_weight}")
        logger.info(f"  Attention Layer Strategy: {attention_layer_strategy}")
        logger.info(f"  Use Unified Layer: {use_unified_layer}")
        logger.info(f"  Dedup: {dedup_enabled}, threshold={dedup_threshold}")
        
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
        """计算KV Cache占用的CPU内存(字节)"""
        total_bytes = 0
        
        for gran in self.granularities:
            store = self.multigran_kv_store.kv_stores[gran]
            
            for kv_layers_cpu in store['keys']:
                for k, v in kv_layers_cpu:
                    total_bytes += k.numel() * k.element_size()
                    total_bytes += v.numel() * v.element_size()
            
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
    
    def _get_target_layer_idx(self, n_layers: int) -> int:
        """
        🆕 根据attention_layer_strategy计算目标层索引
        
        Args:
            n_layers: 总层数
            
        Returns:
            layer_idx: 目标层索引（正数索引）
        """
        if self.attention_layer_strategy == 'last':
            return n_layers - 1
        elif self.attention_layer_strategy == 'second_last':
            return n_layers - 2 if n_layers >= 2 else n_layers - 1
        elif self.attention_layer_strategy == 'middle':
            return n_layers // 2
        elif self.attention_layer_strategy == 'last_5':
            # 对于多层策略，FFT/Query使用最后一层
            return n_layers - 1
        else:
            return n_layers - 1
    
    def _compute_fft_importance(
        self, 
        kv_layers: List[Tuple[torch.Tensor, torch.Tensor]],
        init_len: int,
        method: str = 'diff',
        layer_idx: Optional[int] = None  # 🆕 可选：指定使用哪一层
    ) -> torch.Tensor:
        """
        🆕 基于FFT计算token的时序变化重要性
        
        Args:
            kv_layers: List[(k, v)] - 所有层的KV
            init_len: init KV的长度
            method: 'diff' | 'fft' | 'spectral_entropy'
            layer_idx: 可选，指定使用哪一层的K向量（默认最后一层）
            
        Returns:
            fft_importance: (actual_seq_len,) - 时序变化重要性分数
        """
        # 🆕 根据layer_idx选择层，默认使用最后一层
        if layer_idx is None:
            target_k = kv_layers[-1][0]  # (batch, n_heads, seq_len, head_dim)
        else:
            target_k = kv_layers[layer_idx][0]
        
        # ✅ 修复：输入的kv_layers已经在调用前移除了init部分，不需要再次切片
        # 原代码：new_k = target_k[:, :, init_len:, :]  # ❌ 重复切片导致丢失前13个token
        new_k = target_k  # ✅ 直接使用，避免重复切片
        
        # 聚合batch和heads
        k_avg = new_k.mean(dim=(0, 1))  # (actual_seq_len, head_dim)
        
        if method == 'diff':
            # 方法1: Token间差分 (推荐 - 快速且有效)
            # 计算相邻tokens的向量差异，差异大 = 变化大 = 高频 = 重要
            diff = torch.diff(k_avg, dim=0, prepend=k_avg[0:1])  # (actual_seq_len, head_dim)
            change_magnitude = diff.norm(dim=-1)  # (actual_seq_len,)
            fft_importance = change_magnitude
            
        elif method == 'fft':
            # 方法2: 真正的FFT分析
            # 在时序维度做FFT，分析每个特征维度的频率分布
            k_transposed = k_avg.T  # (head_dim, actual_seq_len)
            
            # 🔧 FIX: cuFFT在FP16下要求长度为2的幂次方，转换为FP32进行FFT
            original_dtype = k_transposed.dtype
            if k_transposed.dtype in [torch.float16, torch.bfloat16]:
                k_transposed = k_transposed.float()
            
            # 1D FFT
            fft_result = torch.fft.rfft(k_transposed, dim=-1)  # (head_dim, freq_bins)
            magnitude = torch.abs(fft_result)  # 频谱幅度
            
            # 计算高频能量 (上半部分频率)
            freq_bins = magnitude.shape[-1]
            high_freq_start = max(1, freq_bins // 2)  # 跳过DC分量
            
            # 高频mask
            high_freq_mask = torch.zeros_like(magnitude)
            high_freq_mask[:, high_freq_start:] = magnitude[:, high_freq_start:]
            
            # 逆FFT回到时域，得到每个token的高频能量贡献
            high_freq_component = torch.fft.irfft(
                high_freq_mask * torch.exp(1j * torch.angle(fft_result)), 
                n=k_avg.shape[0], 
                dim=-1
            )
            
            # 聚合所有特征维度
            fft_importance = torch.abs(high_freq_component).mean(dim=0)  # (actual_seq_len,)
            
            # 🔧 转回原始dtype
            if fft_importance.dtype != original_dtype:
                fft_importance = fft_importance.to(original_dtype)
            
        elif method == 'spectral_entropy':
            # 方法3: 频谱熵
            # 熵高 = 频率分布均匀 = 复杂信号 = 重要
            k_transposed = k_avg.T
            
            # 🔧 FIX: cuFFT在FP16下要求长度为2的幂次方，转换为FP32进行FFT
            original_dtype = k_transposed.dtype
            if k_transposed.dtype in [torch.float16, torch.bfloat16]:
                k_transposed = k_transposed.float()
            
            fft_result = torch.fft.rfft(k_transposed, dim=-1)
            power_spectrum = torch.abs(fft_result) ** 2
            
            # 归一化为概率分布
            prob = power_spectrum / (power_spectrum.sum(dim=-1, keepdim=True) + 1e-10)
            
            # 计算熵
            entropy = -(prob * torch.log(prob + 1e-10)).sum(dim=-1)  # (head_dim,)
            
            # 简化: 使用整体熵作为权重
            fft_importance = entropy.mean().expand(k_avg.shape[0])
            
            # 🔧 转回原始dtype
            if fft_importance.dtype != original_dtype:
                fft_importance = fft_importance.to(original_dtype)
        
        else:
            raise ValueError(f"Unknown FFT method: {method}")
        
        # 归一化到[0, 1]
        fft_importance = (fft_importance - fft_importance.min()) / \
                        (fft_importance.max() - fft_importance.min() + 1e-8)
        
        return fft_importance
    
    def _compute_token_importance(
        self, 
        attentions: Tuple[torch.Tensor],
        kv_layers: List[Tuple[torch.Tensor, torch.Tensor]],
        granularity: int,
        method: str = 'attention_fft_weighted',
        return_components: bool = False
    ) -> torch.Tensor:
        """
        🆕 Attention + FFT 加权融合计算token重要性
        
        Args:
            attentions: tuple of (batch, n_heads, seq_len, seq_len)
            kv_layers: List[(k, v)] - 所有层的KV
            granularity: 当前粒度
            method: 'attention_fft_weighted' | 'attention_only' | 'fft_only'
            return_components: 是否返回attention和fft的分量（用于可视化）
            
        Returns:
            importance_scores: (actual_seq_len,) - 融合后的重要性分数
            或者 (importance_scores, attn_norm, fft_norm) 如果return_components=True
        """
        init_len = self.init_kv_cache[0][0].shape[2]
        n_layers = len(attentions)
        
        # 1️⃣ 根据策略选择Attention层
        if self.attention_layer_strategy == 'last':
            selected_attns = [attentions[-1]]
        elif self.attention_layer_strategy == 'second_last':
            selected_attns = [attentions[-2]] if n_layers >= 2 else [attentions[-1]]
        elif self.attention_layer_strategy == 'last_5':
            selected_attns = list(attentions[-5:]) if n_layers >= 5 else list(attentions)
        elif self.attention_layer_strategy == 'middle':
            mid_idx = n_layers // 2
            selected_attns = [attentions[mid_idx]]
        else:
            selected_attns = [attentions[-1]]
        
        # 2️⃣ 计算Attention重要性
        if len(selected_attns) == 1:
            attn_importance = selected_attns[0].mean(dim=0).mean(dim=0).sum(dim=0)
        else:
            # last_5: 加权平均
            weights = [0.1, 0.15, 0.2, 0.25, 0.3][-len(selected_attns):]
            attn_importance = torch.zeros(attentions[0].shape[-1], device=attentions[0].device, dtype=attentions[0].dtype)
            for w, attn in zip(weights, selected_attns):
                attn_importance += w * attn.mean(dim=0).mean(dim=0).sum(dim=0)
        
        attn_importance = attn_importance[init_len:]  # 只取新tokens
        
        if method == 'attention_only' or not self.fft_enabled:
            if return_components:
                attn_norm = (attn_importance - attn_importance.min()) / \
                           (attn_importance.max() - attn_importance.min() + 1e-8)
                return attn_importance, attn_norm, None
            return attn_importance
        
        # 2️⃣ 计算FFT重要性
        # 🆕 如果启用统一层策略，FFT使用与Attention相同的层
        if self.use_unified_layer:
            fft_layer_idx = self._get_target_layer_idx(n_layers)
            fft_importance = self._compute_fft_importance(kv_layers, init_len, self.fft_method, layer_idx=fft_layer_idx)
        else:
            fft_importance = self._compute_fft_importance(kv_layers, init_len, self.fft_method)
        
        if method == 'fft_only':
            if return_components:
                fft_norm = fft_importance  # 已经归一化
                return fft_importance, None, fft_norm
            return fft_importance
        
        # 3️⃣ 🆕 粒度自适应加权融合
        alpha = self.attention_weight.get(granularity, 0.7)
        beta = self.fft_weight.get(granularity, 0.3)
        
        # 确保两者长度一致
        if attn_importance.shape[0] != fft_importance.shape[0]:
            logger.warning(f"Length mismatch: attn={attn_importance.shape[0]}, fft={fft_importance.shape[0]}")
            min_len = min(attn_importance.shape[0], fft_importance.shape[0])
            attn_importance = attn_importance[:min_len]
            fft_importance = fft_importance[:min_len]
        
        # 归一化后融合
        attn_norm = (attn_importance - attn_importance.min()) / \
                   (attn_importance.max() - attn_importance.min() + 1e-8)
        fft_norm = fft_importance  # 已经归一化
        
        # 加权融合
        combined_importance = alpha * attn_norm + beta * fft_norm
        
        logger.debug(f"[Gran {granularity}] α={alpha:.2f}, β={beta:.2f}, "
                    f"attn_range=[{attn_norm.min():.3f}, {attn_norm.max():.3f}], "
                    f"fft_range=[{fft_norm.min():.3f}, {fft_norm.max():.3f}]")
        
        if return_components:
            return combined_importance, attn_norm, fft_norm
        return combined_importance
    
    def _get_keep_ratio(self, granularity: int) -> float:
        """获取粒度特定的保留率"""
        return self.keep_ratios.get(granularity, 0.5)
    
    def _select_tokens_with_dedup(
        self,
        full_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
        importance_scores: torch.Tensor,
        target_k: int,
        similarity_threshold: float
    ) -> torch.Tensor:
        """去冗余选择tokens (与原版相同)"""
        seq_len = importance_scores.shape[0]
        
        candidate_k = min(int(target_k * 1.2), seq_len)
        candidate_indices = importance_scores.topk(candidate_k).indices
        candidate_indices = candidate_indices.long().contiguous()
        
        if not self.dedup_enabled or candidate_k <= target_k:
            return candidate_indices[:target_k].contiguous()
        
        last_layer_k = full_kvs[-1][0]
        last_layer_k = last_layer_k.mean(dim=1).squeeze(0)
        candidate_k_vectors = last_layer_k[candidate_indices]
        
        candidate_k_norm = candidate_k_vectors / candidate_k_vectors.norm(dim=1, keepdim=True)
        sim_matrix = torch.mm(candidate_k_norm, candidate_k_norm.t())
        
        keep_mask = torch.ones(candidate_k, dtype=torch.bool, device=sim_matrix.device)
        
        for i in range(candidate_k):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, candidate_k):
                if sim_matrix[i, j] > similarity_threshold:
                    idx_i = candidate_indices[i]
                    idx_j = candidate_indices[j]
                    if importance_scores[idx_i] < importance_scores[idx_j]:
                        keep_mask[i] = False
                        break
                    else:
                        keep_mask[j] = False
        
        kept_indices = candidate_indices[keep_mask]
        
        if len(kept_indices) > target_k:
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
        """为指定粒度编码并压缩 (FFT增强版)"""
        batch_size, n_tokens, hidden_dim = visual_features.shape
        
        n_blocks = n_tokens // granularity
        if n_blocks == 0:
            return
            
        blocks = visual_features[:, :n_blocks * granularity].reshape(
            batch_size, n_blocks, granularity, hidden_dim
        )
        
        for block_idx in range(n_blocks):
            block_features = blocks[:, block_idx]
            
            output = self.language_model(
                inputs_embeds=block_features, 
                past_key_values=self.init_kv_cache,
                use_cache=True, 
                return_dict=True,
                output_attentions=True if self.enable_compression else False
            )
            
            init_len = self.init_kv_cache[0][0].shape[2]
            full_kvs = []
            for k, v in output.past_key_values:
                new_k = k[:, :, init_len:, :]
                new_v = v[:, :, init_len:, :]
                full_kvs.append((new_k, new_v))
            
            if self.enable_compression:
                actual_seq_len = full_kvs[0][0].shape[2]
                
                # 🆕 使用Attention + FFT融合计算重要性
                importance_scores = self._compute_token_importance(
                    output.attentions, 
                    full_kvs,
                    granularity,
                    method=self.importance_method
                )
                
                if importance_scores.dim() > 1:
                    importance_scores = importance_scores.squeeze()
                    if importance_scores.dim() > 1:
                        importance_scores = importance_scores.flatten()
                
                if importance_scores.shape[0] != actual_seq_len:
                    logger.warning(f"Importance length mismatch: {importance_scores.shape[0]} vs {actual_seq_len}, gran={granularity}")
                    if importance_scores.shape[0] > actual_seq_len:
                        importance_scores = importance_scores[-actual_seq_len:]
                    else:
                        padding_len = actual_seq_len - importance_scores.shape[0]
                        min_importance = importance_scores.min().item()
                        padding = torch.full((padding_len,), min_importance, 
                                            device=importance_scores.device,
                                            dtype=importance_scores.dtype)
                        importance_scores = torch.cat([importance_scores, padding])
                
                keep_ratio = self._get_keep_ratio(granularity)
                target_k = int(actual_seq_len * keep_ratio)
                
                final_indices = self._select_tokens_with_dedup(
                    full_kvs, importance_scores, target_k, self.dedup_threshold
                )
                
                final_indices = final_indices.to(full_kvs[0][0].device)
                
                seq_len = full_kvs[0][0].shape[2]
                max_idx = final_indices.max().item() if len(final_indices) > 0 else -1
                
                if max_idx >= seq_len:
                    logger.warning(f"Index overflow: max={max_idx}, seq_len={seq_len}, gran={granularity}")
                    final_indices = final_indices[final_indices < seq_len]
                
                if len(final_indices) == 0:
                    logger.warning(f"No valid indices after filtering, using all tokens")
                    final_indices = torch.arange(seq_len, device=full_kvs[0][0].device, dtype=torch.long)
                
                final_indices, _ = torch.sort(final_indices)
                
                compressed_kvs = []
                for layer_idx, (k, v) in enumerate(full_kvs):
                    layer_seq_len = k.shape[2]
                    layer_valid_indices = final_indices[final_indices < layer_seq_len]
                    
                    if len(layer_valid_indices) == 0:
                        logger.warning(f"Layer {layer_idx}: no valid indices, using all")
                        layer_valid_indices = torch.arange(layer_seq_len, device=k.device, dtype=torch.long)
                    
                    compressed_kvs.append((
                        k[:, :, layer_valid_indices, :],
                        v[:, :, layer_valid_indices, :]
                    ))
                
                compression_info = {
                    'original_tokens': granularity,
                    'kept_tokens': len(final_indices),
                    'compression_ratio': len(final_indices) / granularity,
                    'importance_mean': importance_scores.mean().item(),
                    'importance_std': importance_scores.std().item()
                }
                
                last_layer_k = compressed_kvs[-1][0]
                repr_vector = last_layer_k.mean(dim=2)
                repr_vector = repr_vector.reshape(batch_size, -1).squeeze(0)
                
                self.multigran_kv_store.store_kv_cache(
                    granularity=granularity,
                    kv_layers=compressed_kvs,
                    repr_vector=repr_vector,
                    token_indices=final_indices,
                    compression_info=compression_info
                )
                
            else:
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
    
    @torch.inference_mode() 
    def encode_video(self, video, encode_chunk_size=64):
        """多粒度视频编码（FFT增强的Token压缩）"""
        num_frames = video.shape[0]
        num_chunks = num_frames // encode_chunk_size
        
        print(f"\n{'='*60}")
        print(f"[FFT-Enhanced Video Encoding with Token Compression]")
        print(f"Frames: {num_frames}, Chunks: {num_chunks}")
        print(f"Granularities: {self.granularities}")
        print(f"Method: Attention + FFT (α={self.attention_weight}, β={self.fft_weight})")
        print(f"FFT Method: {self.fft_method}")
        print(f"{'='*60}")
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]
            
            pixel_values = self.processor.video_processor(chunk_video, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)
            visual_features = self._get_video_features(pixel_values)
            
            print(f"Chunk {chunk_idx+1}/{num_chunks}: ", end="", flush=True)
            
            for granularity in self.granularities:
                self._encode_granularity_chunk_with_compression(visual_features, granularity)
            
            stats = self.multigran_kv_store.get_stats()
            print(f"Blocks: P={stats[49]['n_blocks']}, F={stats[196]['n_blocks']}, S={stats[784]['n_blocks']}")
        
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            remaining_video = video[start_idx:start_idx + remaining_frames]
            
            pixel_values = self.processor.video_processor(remaining_video, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)
            visual_features = self._get_video_features(pixel_values)
            
            for granularity in self.granularities:
                self._encode_granularity_chunk_with_compression(visual_features, granularity)
        
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
        """多粒度问答（使用压缩的KV-Cache）"""
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]
        
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
            last_layer_kv = question_output.past_key_values[-1]
            last_k = last_layer_kv[0][:, :, -1, :]
            query_vector = last_k.reshape(last_k.size(0), -1).squeeze(0)
        else:
            raise RuntimeError("Failed to get past_key_values")
        
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
    # 🆕 FFT参数
    fft_enabled=True,
    fft_method='diff',
    attention_weight_49=0.5,
    attention_weight_196=0.7,
    attention_weight_784=0.8,
    # 🆕 Attention层策略
    attention_layer_strategy='last',
    # 🆕 统一层策略
    use_unified_layer=False,
):
    """加载FFT增强的Token压缩多粒度MuKV模型"""
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
    
    # 🆕 FFT权重配置
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
    
    logger.info("MuKV dual-signal compression configuration:")
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
    
    model = MuKVDualSignalCompressionModel.from_pretrained(
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
    )
    
    logger.info(f'MuKV dual-signal compression model loaded with {len(granularities)} granularities')
    model.eval()

    return model, processor
