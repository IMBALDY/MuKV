"""
MuKV on RVS-Ego.
"""
import os
import json
import argparse
import torch
import random
import numpy as np
import time
import csv
from tqdm import tqdm
from logzero import logger
import logzero
from transformers import logging as transformers_logging
from decord import VideoReader, cpu

from video_qa.base import BaseVQA
from model import mukv_rerank

import sys
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# Allows resolving imports for model and video_qa when running from scripts/ run_mukv_xxx.py
sys.path.append(os.path.dirname(PROJECT_DIR))


class MuKVRVSEgoRunner(BaseVQA):
    """MuKV RVS-Ego 入口。"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_memory_snapshots = []
    
    def load_video(self, video_path):
        """Load video"""
        if video_path.endswith('.npy'):
            video = np.load(video_path)
            num_frames = len(video)
            frame_idx = np.linspace(0, num_frames-1, int(num_frames*self.sample_fps), dtype=int).tolist()
            video = video[frame_idx]
        else:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            fps = round(vr.get_avg_fps())
            frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
            video = vr.get_batch(frame_idx).asnumpy()
        logger.debug(f'video shape: {video.shape}')
        return video
    
    def video_open_qa(self, question, max_new_tokens=1024):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }
        return self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens)

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        """Streaming模式: 增量编码到t时刻"""
        video_id = video_sample['video_id']
        video_path = video_sample['video_path']
        questions = video_sample['conversations']
        video_start_idx = video_end_idx = 0  # 跟踪编码进度
        
        logger.info(f"{'='*80}")
        logger.info(f"Processing video: {video_id} (Streaming Mode)")
        logger.info(f"Video path: {video_path}")
        logger.info(f"Questions: {len(questions)}")
        logger.info(f"{'='*80}")
        
        # 加载视频(但不立即编码)
        start_time = time.time()
        video = self.load_video(video_path)
        video_tensor = torch.from_numpy(video) if not isinstance(video, torch.Tensor) else video
        load_time = time.time() - start_time
        logger.info(f"Video loaded: {video.shape}, time: {load_time:.2f}s")
        
        if len(video_tensor) < 15:
            logger.warning(f"Skipping video {video_id}: insufficient frames")
            return []
        
        # 清空缓存
        self.qa_model.clear_cache()
        
        # 编码初始提示
        self.qa_model.encode_init_prompt()
        
        # 内存监控初始化
        torch.cuda.reset_peak_memory_stats()  # 重置峰值统计
        video_start_time = time.time()
        initial_gpu_memory = torch.cuda.memory_allocated() / (1024**2)
        initial_kv_memory = 0.0
        
        # 问答
        results = []
        inference_times = []
        total_encode_time = 0.0
        
        for q_idx, qa_pair in enumerate(questions):
            question_text = qa_pair['question']
            
            logger.info(f"\n--- Question {q_idx+1}/{len(questions)} ---")
            logger.info(f"Q: {question_text}")
            
            # 处理时间戳
            if 'start_time' in qa_pair and 'end_time' in qa_pair:
                temporal_windows = torch.tensor([qa_pair['start_time'], qa_pair['end_time']]) * self.sample_fps
            else:
                question_time = qa_pair.get('start_time', 0) * self.sample_fps
                temporal_windows = torch.tensor([0, question_time])
            
            temporal_windows = temporal_windows.tolist()
            
            # 只编码到问题时间点 (Streaming核心逻辑)
            if temporal_windows[-1] > video_end_idx:
                video_end_idx = temporal_windows[-1]
                encode_start_idx = int(video_start_idx)
                encode_end_idx = int(video_end_idx)
                
                logger.info(f"Encoding frames {encode_start_idx}-{encode_end_idx}...")
                encode_start = time.time()
                self.qa_model.encode_video(video_tensor[encode_start_idx:encode_end_idx], encode_chunk_size=64)
                encode_time = time.time() - encode_start
                total_encode_time += encode_time
                
                video_start_idx = video_end_idx
                logger.info(f"Encoded in {encode_time:.2f}s")
            
            # 问答
            qa_start = time.time()
            answer = self.video_open_qa(question_text, max_new_tokens=256)
            qa_time = time.time() - qa_start
            inference_times.append(qa_time)
            
            logger.info(f"A: {answer}")
            logger.info(f"Time: {qa_time:.2f}s")
            
            results.append({
                'video_id': video_id,
                'question_id': qa_pair.get('question_id', f'{video_id}_q{q_idx}'),
                'question': question_text,
                'answer': qa_pair.get('answer', ''),
                'pred_answer': answer,
                'inference_time': qa_time
            })
        
        # 内存统计
        elapsed_time = time.time() - video_start_time
        current_gpu_memory = torch.cuda.memory_allocated() / (1024**2)
        peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)  # 峰值GPU显存
        current_kv_memory = self.qa_model.calc_kv_cache_memory() / (1024**2)
        
        # 保存内存快照
        snapshot = {
            'video_id': video_id,
            'timestamp': round(elapsed_time, 2),
            'gpu_memory_mb': round(current_gpu_memory, 2),
            'peak_gpu_memory_mb': round(peak_gpu_memory, 2),
            'kv_cache_mb': round(current_kv_memory, 2),
            'n_blocks_stored': sum(stat['n_blocks'] for stat in self.qa_model.multigran_kv_store.get_stats().values()),
            'n_questions': len(questions),
            'avg_inference_time': round(np.mean(inference_times), 2),
            'encode_time': round(total_encode_time, 2),
            'total_time': round(total_encode_time + sum(inference_times), 2)
        }
        self.all_memory_snapshots.append(snapshot)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Video {video_id} completed!")
        logger.info(f"Total encode time: {total_encode_time:.2f}s")
        logger.info(f"Avg QA time: {np.mean(inference_times):.2f}s")
        logger.info(f"Total time: {snapshot['total_time']:.2f}s")
        logger.info(f"{'='*80}\n")
        
        return results
    
    def save_memory_stats(self):
        """保存内存统计"""
        if not self.all_memory_snapshots:
            logger.warning("No memory snapshots to save")
            return
        
        memory_stats_path = f'{self.save_dir}/memory_stats.csv'
        
        with open(memory_stats_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'video_id', 'timestamp', 'gpu_memory_mb', 'peak_gpu_memory_mb', 'kv_cache_mb',
                'n_blocks_stored', 'n_questions', 'avg_inference_time',
                'encode_time', 'total_time'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.all_memory_snapshots)
        
        logger.info(f"Memory stats saved to: {memory_stats_path}")
        logger.info(f"Total snapshots: {len(self.all_memory_snapshots)}")


def main():
    transformers_logging.set_verbosity_error()
    
    parser = argparse.ArgumentParser(description='MuKV on RVS-Ego')
    
    # 基础参数
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    parser.add_argument("--anno_path", type=str, default="data/rvs/ego/ego4d_oe.json")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--retrieve_chunk_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="results/mukv/rvs_ego")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_videos", type=int, default=None)
    
    # 多粒度参数
    parser.add_argument("--granularities", nargs='+', type=int, default=[49, 196, 784])
    parser.add_argument("--granularity_topks", nargs='+', type=int, default=None)
    
    # Token压缩参数
    parser.add_argument("--enable_compression", type=str, default="true")
    parser.add_argument("--keep_ratio_49", type=float, default=0.70)
    parser.add_argument("--keep_ratio_196", type=float, default=0.50)
    parser.add_argument("--keep_ratio_784", type=float, default=0.30)
    parser.add_argument("--importance_method", type=str, default="attention_fft_weighted")
    parser.add_argument("--dedup_threshold", type=float, default=0.90)
    parser.add_argument("--dedup_enabled", type=str, default="false")
    
    # FFT参数
    parser.add_argument("--fft_enabled", type=str, default="true")
    parser.add_argument("--fft_method", type=str, default="diff", 
                       choices=['diff', 'fft', 'spectral_entropy'])
    parser.add_argument("--attention_weight_49", type=float, default=0.5)
    parser.add_argument("--attention_weight_196", type=float, default=0.7)
    parser.add_argument("--attention_weight_784", type=float, default=0.8)
    parser.add_argument("--attention_layer_strategy", type=str, default="last",
                       choices=['last', 'second_last', 'last_5', 'middle'],
                       help="Which attention layer(s) to use for importance calculation")
    parser.add_argument("--use_unified_layer", type=str, default="false",
                       help="Use same layer for Attention/FFT/Query (default: false)")
    
    # 🆕 Rerank参数
    parser.add_argument("--enable_rerank", type=str, default="true")
    parser.add_argument("--rerank_alpha", type=float, default=0.5)
    parser.add_argument("--rerank_beta", type=float, default=0.6)
    parser.add_argument("--rerank_top_n", type=int, default=5)
    parser.add_argument("--rerank_reverse", type=str, default="false",
                       help="🔄 Reverse mode: select LEAST relevant blocks (for ablation study)")
    
    # Video format (for Ego dataset only)
    parser.add_argument("--video_format", type=str, default="mp4",
                       choices=['mp4', 'npy'],
                       help="Video format: mp4 (original) or npy (preprocessed)")
    
    args = parser.parse_args()
    
    # 转换bool参数
    args.enable_compression = args.enable_compression.lower() == "true"
    args.dedup_enabled = args.dedup_enabled.lower() == "true"
    args.fft_enabled = args.fft_enabled.lower() == "true"
    args.enable_rerank = args.enable_rerank.lower() == "true"
    args.rerank_reverse = args.rerank_reverse.lower() == "true"
    args.use_unified_layer = args.use_unified_layer.lower() == "true"
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置日志
    if args.debug:
        logzero.loglevel(10)
    else:
        logzero.loglevel(20)
    
    log_file = f'{args.save_dir}/inference.log'
    logzero.logfile(log_file)
    
    logger.info("="*80)
    logger.info("MuKV inference on RVS-Ego")
    logger.info("="*80)
    logger.info(f"Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("="*80)
    
    # 设置随机种子
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    # 加载模型
    logger.info("Loading MuKV model...")
    model, processor = mukv_rerank.load_model(
        model_path=args.model_path,
        n_local=args.n_local,
        topk=args.retrieve_size,
        chunk_size=args.retrieve_chunk_size,
        granularities=args.granularities,
        granularity_topks=args.granularity_topks,
        enable_compression=args.enable_compression,
        keep_ratio_49=args.keep_ratio_49,
        keep_ratio_196=args.keep_ratio_196,
        keep_ratio_784=args.keep_ratio_784,
        importance_method=args.importance_method,
        dedup_threshold=args.dedup_threshold,
        dedup_enabled=args.dedup_enabled,
        fft_enabled=args.fft_enabled,
        fft_method=args.fft_method,
        attention_weight_49=args.attention_weight_49,
        attention_weight_196=args.attention_weight_196,
        attention_weight_784=args.attention_weight_784,
        attention_layer_strategy=args.attention_layer_strategy,
        use_unified_layer=args.use_unified_layer,
        enable_rerank=args.enable_rerank,
        rerank_alpha=args.rerank_alpha,
        rerank_beta=args.rerank_beta,
        rerank_top_n=args.rerank_top_n,
        rerank_reverse=args.rerank_reverse,
    )
    logger.info("Model loaded successfully!")
    
    # 加载数据
    logger.info(f"Loading annotations from {args.anno_path}")
    with open(args.anno_path, 'r') as f:
        annotations = json.load(f)
    logger.info(f"Total videos: {len(annotations)}")
    
    # Convert video paths if needed
    if args.video_format == "npy":
        logger.info(f"Using NPY format: {annotations[0].get('video_path', '')}")
    else:
        logger.info(f"Using MP4 format: {annotations[0].get('video_path', '')}")
    
    # 分chunk处理
    if args.max_videos:
        annotations = annotations[:args.max_videos]
    
    # 创建VQA对象
    vqa_system = MuKVRVSEgoRunner(
        anno=annotations,
        qa_model=model,
        sample_fps=args.sample_fps,
        save_dir=args.save_dir,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        retrieve_size=args.retrieve_size,
        chunk_size=args.retrieve_chunk_size
    )
    
    logger.info(f"Processing chunk {args.chunk_idx+1}/{args.num_chunks}")
    logger.info(f"Videos to process: {len(vqa_system.anno)}")
    
    # 处理视频
    all_results = []
    for video_sample in tqdm(vqa_system.anno, desc="Processing videos"):
        try:
            results = vqa_system.analyze_a_video(video_sample)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Error processing video {video_sample['video_id']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果
    output_file = f'{args.save_dir}/results.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['video_id', 'question_id', 'question', 'answer', 'pred_answer', 'inference_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    logger.info(f"Results saved to: {output_file}")
    
    # 保存内存统计
    vqa_system.save_memory_stats()
    
    logger.info("="*80)
    logger.info("Inference completed!")
    logger.info(f"Total results: {len(all_results)}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
