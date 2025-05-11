import json
import os
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_gen import ModelGenerator
import jsonlines
from pathlib import Path
import sys

class BaichuanGenerator(ModelGenerator):
    """
    Generator class for Baichuan models (including EduChat based on Baichuan)
    """
    def __init__(self, task_path, model_path, model_name, device="0", is_few_shot=False,
                 few_shot_path=None, is_vllm=False, tensor_parallel_size=1,
                 gpu_memory_utilization=0.9, output_file=None):
        """
        Initialize Baichuan generator
        """
        # Convert task_path to absolute path if it's not already
        if not os.path.isabs(task_path):
            task_path = os.path.abspath(task_path)
        
        # Ensure the file exists
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"Task file not found: {task_path}")
            
        # Determine task type based on filename
        filename = os.path.basename(task_path)
        self.task_type = "unknown" # Default task type

        if "junior" in filename or "primary" in filename or "senior" in filename:
            self.task_type = "multiple_choice"
        elif "4_logiqa_500.jsonl" in filename:
             self.task_type = "multiple_choice"
        elif "5_writing_50.jsonl" in filename:
            self.task_type = "essay"
        elif "2_yuedu_100.jsonl" in filename or "2_shige_100.jsonl" in filename:
            self.task_type = "short_answer"
        elif "3_zuowen_100.jsonl" in filename:
            self.task_type = "essay_grading"
        elif "5_qg" in filename:
            self.task_type = "question_generation"

        # Debug print to confirm task type
        print(f"Determined task type for {filename}: {self.task_type}")
        if self.task_type == "unknown":
            print(f"Warning: Could not determine task type for file {filename}. Please check filename conventions.")

        super().__init__(
            task_path=task_path,
            model_path=model_path,
            model_name=model_name,
            device=device,
            is_few_shot=is_few_shot,
            few_shot_path=few_shot_path,
            is_vllm=is_vllm,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            output_file=output_file
        )
    
    def model_init(self):
        """
        Initialize Baichuan model and tokenizer
        """
        if self.is_vllm:
            # Use vLLM for faster inference
            return self.model_init_vllm()
        
        try:
            # 使用最简单的方法，直接从config.json加载模型信息
            from transformers import AutoConfig, PreTrainedTokenizer
            
            # 获取模型配置
            config_path = os.path.join(self.model_path, "config.json")
            if os.path.exists(config_path):
                print(f"Loading config from {config_path}")
                
                # 直接加载模型
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                # 先加载模型，确保模型正确加载
                print(f"Loading model from {self.model_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map=f"cuda:{self.device}",
                    trust_remote_code=True,
                    local_files_only=True
                )
                
                # 然后手动加载tokenizer
                print(f"Loading tokenizer manually")
                
                # Monkey patch: 手动导入tokenization_baichuan.py并创建tokenizer
                model_full_path = os.path.abspath(self.model_path)
                sys.path.insert(0, model_full_path)
                
                # 创建一个简单的SentencePiece tokenizer
                import sentencepiece as spm
                
                # 加载SP模型
                sp_model_path = os.path.join(model_full_path, "tokenizer.model")
                if os.path.exists(sp_model_path):
                    sp_model = spm.SentencePieceProcessor()
                    sp_model.Load(sp_model_path)
                    print(f"Loaded SentencePiece model with {sp_model.GetPieceSize()} tokens")
                    
                    # 创建简单的封装tokenizer
                    class SimpleTokenizer:
                        def __init__(self, sp_model):
                            self.sp_model = sp_model
                            # 设置常用属性
                            self.vocab_size = sp_model.GetPieceSize()
                            self.bos_token_id = 1  # <s>
                            self.eos_token_id = 2  # </s>
                            self.pad_token_id = 0  # <pad>
                            
                        def encode(self, text, add_special_tokens=True):
                            ids = self.sp_model.EncodeAsIds(text)
                            if add_special_tokens:
                                ids = [self.bos_token_id] + ids + [self.eos_token_id]
                            return ids
                            
                        def decode(self, token_ids):
                            # 过滤掉特殊token
                            filtered_ids = [id for id in token_ids if id < self.vocab_size]
                            return self.sp_model.DecodeIds(filtered_ids)
                        
                        def __call__(self, text, return_tensors=None, add_special_tokens=True):
                            if isinstance(text, list):
                                # 批处理
                                encodings = [self.encode(t, add_special_tokens) for t in text]
                                # 找到最长的序列
                                max_len = max(len(enc) for enc in encodings)
                                # 填充较短的序列
                                padded = []
                                for enc in encodings:
                                    if len(enc) < max_len:
                                        enc = enc + [self.pad_token_id] * (max_len - len(enc))
                                    padded.append(enc)
                                
                                if return_tensors == "pt":
                                    import torch
                                    return {
                                        "input_ids": torch.tensor(padded),
                                        "attention_mask": torch.tensor([[1 if id != self.pad_token_id else 0 for id in seq] for seq in padded])
                                    }
                                return {
                                    "input_ids": padded,
                                    "attention_mask": [[1 if id != self.pad_token_id else 0 for id in seq] for seq in padded]
                                }
                            else:
                                # 单个文本
                                encoded = self.encode(text, add_special_tokens)
                                if return_tensors == "pt":
                                    import torch
                                    return {
                                        "input_ids": torch.tensor([encoded]),
                                        "attention_mask": torch.tensor([[1] * len(encoded)])
                                    }
                                return {
                                    "input_ids": [encoded],
                                    "attention_mask": [[1] * len(encoded)]
                                }
                    
                    # 创建tokenizer实例
                    tokenizer = SimpleTokenizer(sp_model)
                    return model, tokenizer
                else:
                    print(f"SentencePiece model not found at {sp_model_path}, falling back to AutoTokenizer")
            else:
                print(f"Config not found at {config_path}")
                
            # 如果上面的方法失败，尝试使用AutoTokenizer
            print("Falling back to AutoTokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            # 强制指定vocab_size
            if not hasattr(tokenizer, "vocab_size"):
                print("Setting vocab_size attribute")
                if hasattr(tokenizer, "sp_model"):
                    tokenizer.vocab_size = tokenizer.sp_model.get_piece_size()
                else:
                    vocab = tokenizer.get_vocab()
                    tokenizer.vocab_size = len(vocab)
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=f"cuda:{self.device}",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            model.eval()
            
        except Exception as e:
            print(f"Error loading model/tokenizer: {str(e)}")
            raise e
            
        return model, tokenizer
    
    # 以下是任务特定的后处理函数
    def extract_choice_answer(self, text):
        """
        提取选择题答案，只保留ABCD选项
        """
        cleaned_text = text.strip()
        if "正确选项" in cleaned_text:
            cleaned_text = cleaned_text.split("正确选项")[1].strip()
            cleaned_text = cleaned_text.strip("：").strip(":").strip()
        pattern_end = r'([A-D])\s*$'
        match_end = re.search(pattern_end, cleaned_text)
        if match_end: return match_end.group(1).upper()
        pattern1 = r'\b([A-D])[\.。\s]'
        matches = re.findall(pattern1, cleaned_text)
        if matches: return matches[-1].upper()
        pattern2 = r'(?:答案|选择|选项|应选)(?:是|为)?[^A-Da-d]*([A-Da-d])'
        match = re.search(pattern2, cleaned_text)
        if match: return match.group(1).upper()
        pattern3 = r'[A-D]'
        matches = re.findall(pattern3, cleaned_text)
        if matches: return matches[-1].upper()
        pattern4 = r'[a-d]'
        matches = re.findall(pattern4, cleaned_text)
        if matches: return matches[-1].upper()
        pattern5 = r'[A-Da-d]'
        matches = re.findall(pattern5, text)
        if matches: return matches[-1].upper()
        return ""
    
    def cleanup_essay_output(self, text):
        """
        清理作文输出，删除无关信息
        """
        # 删除提示句末尾的回复格式指导
        patterns = [
            r"请针对这个题目写一篇作文[：:]?",
            r"请根据上述要求写一篇作文[：:]?",
            r"下面是我写的作文[：:]?",
            r"以下是我的作文[：:]?",
            r"以下是我针对这个题目写的作文[：:]?"
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text)
        return text.strip()
    
    def extract_score(self, text):
        """
        从文本中提取作文评分
        """
        # 定义可能的分数模式
        score_patterns = [
            r'总分[:：\s]*?(\d+)[\s分]',
            r'得分[:：\s]*?(\d+)[\s分]',
            r'分数[:：\s]*?(\d+)[\s分]',
            r'评分[:：\s]*?(\d+)[\s分]',
            r'(\d+)[\s]*?[/／][\s]*?100',
            r'(\d+)[\s]*?分'
        ]
        
        # 尝试匹配各种模式
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # 如果没有找到明确的分数表示，查找数字打分
        digits = re.findall(r'\b(\d{2,3})\b', text)
        # 过滤出可能的分数 (60-100)
        possible_scores = [int(d) for d in digits if 60 <= int(d) <= 100]
        if possible_scores:
            return str(possible_scores[0])
            
        return "75"  # 默认分数 

    def generate_output(self, tokenizer, model, batch_size=1, max_items=None, offset=0):
        """
        为Baichuan模型定制的输出生成方法
        
        Args:
            tokenizer: 分词器
            model: 加载的模型
            batch_size: 批处理大小
            max_items: 最多处理的项目数
            offset: 处理的起始偏移量
        """
        print(f"Preparing prompts for {self.task_type} task...")
        prompts, answers, questions, subjects = self.prepare_prompts(max_items, offset)
        
        if not prompts:
            print("No prompts to process. Exiting.")
            return
            
        print(f"Processing {len(prompts)} prompts with batch size {batch_size}")
        outputs = []
        
        # 对于SimpleTokenizer，进行特殊处理
        is_simple_tokenizer = hasattr(tokenizer, 'sp_model') and not hasattr(tokenizer, 'convert_tokens_to_string')
        
        # 保存结果到文件
        processed_items = []
        
        for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
            try:
                # 创建模型输入
                if is_simple_tokenizer:
                    # 使用SimpleTokenizer的特殊处理
                    inputs = tokenizer(prompt, return_tensors="pt")
                    inputs = {k: v.to(f"cuda:{self.device}") for k, v in inputs.items()}
                else:
                    # 使用标准tokenizer
                    inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{self.device}")
                
                # 设置生成参数
                gen_kwargs = {
                    "max_new_tokens": 2048,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True,
                }
                
                # 执行生成
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)
                
                # 解码生成的文本
                if is_simple_tokenizer:
                    # 对于SimpleTokenizer，直接使用decode方法
                    output_text = tokenizer.decode(outputs[0].tolist())
                    # 去除提示词部分
                    prompt_ids = tokenizer.encode(prompt)
                    if len(prompt_ids) < len(outputs[0]):
                        output_text = tokenizer.decode(outputs[0][len(prompt_ids):].tolist())
                else:
                    # 对于标准tokenizer，使用batch_decode
                    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                    # 移除输入提示
                    if prompt in output_text:
                        output_text = output_text[len(prompt):]
                
                # 后处理输出文本
                processed_text = self.post_process_output(output_text, self.task_type)
                processed_items.append(processed_text)
                
                # 如果有输出文件，则实时保存
                if len(processed_items) % (batch_size * 5) == 0 or i == len(prompts) - 1:
                    self.save_results(
                        questions[:len(processed_items)],
                        processed_items,
                        answers[:len(processed_items)],
                        subjects[:len(processed_items)] if subjects else None
                    )
                    
            except Exception as e:
                print(f"Error processing prompt {i}: {str(e)}")
                # 添加一个空输出，以保持索引一致
                processed_items.append("")
        
        # 最终保存
        self.save_results(questions, processed_items, answers, subjects)
        
        # 返回处理结果
        if subjects:
            return processed_items, answers, questions, subjects
        else:
            return processed_items, answers, questions 