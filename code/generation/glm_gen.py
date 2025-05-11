import json
import os
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_gen import ModelGenerator
import jsonlines
from pathlib import Path

class GLMGenerator(ModelGenerator):
    """
    Generator class for GLM models
    """
    def __init__(self, task_path, model_path, model_name, device="0", is_few_shot=False,
                 few_shot_path=None, is_vllm=False, tensor_parallel_size=1,
                 gpu_memory_utilization=0.9, output_file=None):
        """
        Initialize GLM generator
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
        Initialize GLM model and tokenizer
        """
        if self.is_vllm:
            # Use vLLM for faster inference
            return self.model_init_vllm()
        
        try:
            # Check if the path is a local directory or HuggingFace model ID
            is_local_dir = os.path.isdir(self.model_path)
            
            print(f"Loading tokenizer from {self.model_path} (local_dir={is_local_dir})")
            tokenizer_kwargs = {
                "trust_remote_code": True,
            }
            
            # Add local_files_only=True for local paths
            if is_local_dir:
                tokenizer_kwargs["local_files_only"] = True
                
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                **tokenizer_kwargs
            )
            
            print(f"Loading model from {self.model_path}")
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": f"cuda:{self.device}",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,  # 降低CPU内存使用
                "max_memory": {0: "16GiB"},   # <-- Increase memory limit
                # "attn_implementation": "eager" # <-- Remove/Comment out
            }
            
            # Add local_files_only=True for local paths
            if is_local_dir:
                model_kwargs["local_files_only"] = True
                
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            model.eval()
            
        except Exception as e:
            print(f"Error loading model/tokenizer from {self.model_path}: {e}")
            print("Trying to load with low_cpu_mem_usage=True")
            # Try again with low_cpu_mem_usage
            is_local_dir = os.path.isdir(self.model_path)
            tokenizer_kwargs = {
                "trust_remote_code": True,
            }
            
            if is_local_dir:
                tokenizer_kwargs["local_files_only"] = True
                
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                **tokenizer_kwargs
            )
            
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": f"cuda:{self.device}",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "max_memory": {0: "8GiB"},   # 限制GPU内存使用
                "attn_implementation": "eager"
            }
            
            if is_local_dir:
                model_kwargs["local_files_only"] = True
                
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            model.eval()
            
        return model, tokenizer
    
    def extract_choice_answer(self, text):
        """
        提取选择题答案，只保留ABCD选项
        """
        # 首先尝试清除指令和思考过程，只保留答案部分
        cleaned_text = text.strip()
        
        # 移除常见的提示词和思考过程
        if "正确选项" in cleaned_text:
            cleaned_text = cleaned_text.split("正确选项")[1].strip()
            cleaned_text = cleaned_text.strip("：").strip(":").strip()
        
        # 检查是否有纯字母答案出现在结尾
        pattern_end = r'([A-D])\s*$'
        match_end = re.search(pattern_end, cleaned_text)
        if match_end:
            return match_end.group(1).upper()
        
        # 检查文本中所有单独的选项字母
        pattern1 = r'\b([A-D])[\.。\s]'
        matches = re.findall(pattern1, cleaned_text)
        if matches:
            # 返回最后一个匹配项，通常是最终答案
            return matches[-1].upper()
        
        # 匹配"答案是X"或"选X"模式
        pattern2 = r'(?:答案|选择|选项|应选)(?:是|为)?[^A-Da-d]*([A-Da-d])'
        match = re.search(pattern2, cleaned_text)
        if match:
            return match.group(1).upper()
        
        # 匹配任何ABCD选项
        pattern3 = r'[A-D]'
        matches = re.findall(pattern3, cleaned_text)
        if matches:
            # 优先选择末尾的选项，因为更可能是最终答案
            return matches[-1].upper()
        
        # 没有找到任何选项，尝试小写选项
        pattern4 = r'[a-d]'
        matches = re.findall(pattern4, cleaned_text)
        if matches:
            return matches[-1].upper()
        
        # 如果实在找不到，尝试从整个原始文本中找
        pattern5 = r'[A-Da-d]'
        matches = re.findall(pattern5, text)
        if matches:
            return matches[-1].upper()
        
        # 没有找到任何选项
        return ""
    
    def cleanup_essay_output(self, text):
        """
        清理作文输出，移除思考过程和系统提示内容
        """
        # 移除提示词和思考过程的模式
        text = text.strip()
        
        # 移除包含作文提示的部分
        prefixes_to_remove = [
            "请根据以下题目写一篇作文。",
            "题目:", 
            "作文:",
            "下面是",
            "### ",
            "好的，"
        ]
        
        for prefix in prefixes_to_remove:
            if prefix in text:
                parts = text.split(prefix, 1)
                if len(parts) > 1:
                    text = parts[1].strip()
        
        # 清理其他工件和后缀
        unwanted_suffixes = ["\\n", "###", "我希望", "总结：", "总之，"]
        for suffix in unwanted_suffixes:
            if text.endswith(suffix):
                text = text[:text.rfind(suffix)].strip()
        
        return text.strip()
    
    def extract_score(self, text):
        """
        从文本中提取分数，确保只返回1-100的数字
        """
        # 打印原始输出以便调试
        print(f"提取分数的原始文本: {text[:200]}...")
        
        # 先尝试查找"评分：X分"这种明确的格式
        eval_pattern = r'(?:评分|得分|分数)[：:]\s*(\d+)(?:分)?'
        match = re.search(eval_pattern, text)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 100:
                print(f"找到明确评分格式，分数为: {score}")
                return str(score)
        
        # 再尝试查找"分数是X"、"打X分"等表达
        score_patterns = [
            r'(?:分数|评分|得分|成绩)(?:为|是|应该是|应为|给|打)[^\d]*?(\d+)(?:分)?',
            r'(?:给予|打|评|得|获得)[^\d]*?(\d+)(?:分)?',
            r'(?:\d+)[^\d]*?分[^\d]*?(?:数|值|数值)',
            r'(?:总分|满分|总评分|最终评分)[^\d]*?(\d+)(?:分)?'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    score = int(match.group(1))
                    if 1 <= score <= 100:
                        print(f"找到评分表达，分数为: {score}")
                        return str(score)
                except:
                    pass
        
        # 查找任何可能的70-89分这样的范围，取平均值
        range_pattern = r'(\d+)\s*(?:-|~|到|至)\s*(\d+)(?:分)?'
        matches = re.findall(range_pattern, text)
        for match in matches:
            try:
                low, high = int(match[0]), int(match[1])
                if 1 <= low <= 100 and 1 <= high <= 100:
                    avg_score = (low + high) // 2
                    print(f"找到分数范围 {low}-{high}，取平均值: {avg_score}")
                    return str(avg_score)
            except:
                pass
        
        # 查找任何可能的数字（优先选择"分"附近的数字）
        all_scores = []
        
        # 先匹配"分"附近的数字
        near_score_pattern = r'(\d+)[^\d]{0,5}分'
        matches = re.findall(near_score_pattern, text)
        for match in matches:
            try:
                score = int(match)
                if 1 <= score <= 100:
                    all_scores.append((score, 0))  # 0表示优先级最高
            except:
                pass
        
        # 然后匹配所有1-100范围内的数字
        all_numbers_pattern = r'\b(\d+)\b'
        matches = re.findall(all_numbers_pattern, text)
        for match in matches:
            try:
                score = int(match)
                if 1 <= score <= 100:
                    # 优先选择60-100之间的合理分数
                    priority = 1 if 60 <= score <= 100 else 2
                    all_scores.append((score, priority))
            except:
                pass
        
        # 如果找到了分数，按优先级排序后返回
        if all_scores:
            # 按优先级排序
            all_scores.sort(key=lambda x: x[1])
            best_score = all_scores[0][0]
            print(f"从所有匹配中选择最佳分数: {best_score}")
            return str(best_score)
        
        # 如果依然无法提取分数，返回默认分数75分（中等偏上）
        print("无法提取有效分数，使用默认分数: 75")
        return "75"
    
    def post_process_output(self, output_text, task_type):
        """
        根据任务类型处理模型输出
        """
        if task_type == "multiple_choice":
            return self.extract_choice_answer(output_text)
        elif task_type == "essay":
            return self.cleanup_essay_output(output_text)
        elif task_type == "essay_grading":
            return self.extract_score(output_text)
        else:
            return output_text.strip()  # 短答题等其他类型保持原样，但去掉前后空白
    
    def generate_output(self, tokenizer, model, batch_size=1, max_items=100, offset=0):
        """
        Generate model outputs for educational tasks
        
        Args:
            tokenizer: Tokenizer for the model
            model: The language model
            batch_size: Batch size for generating outputs
            max_items: Maximum number of items to process in a batch
            offset: Starting offset for data processing (for batch processing)
            
        Returns:
            outputs: List of model outputs
            answers: List of reference answers
            questions: List of questions
            subjects: List of subjects
        """
        # Prepare prompts
        prompts, answers, questions, subjects = self.prepare_prompts(max_items=max_items, offset=offset)
        
        if not prompts:
            print("No prompts were prepared, check if the task file is empty or invalid.")
            return [], [], [], []
            
        print(f"Generating outputs for {len(prompts)} prompts, batch size {batch_size}")
        
        # Apply Chat Template if necessary
        chat_formatted_prompts = []
        for prompt in prompts:
            if hasattr(tokenizer, "apply_chat_template"):
                # For newer GLM models that support chat templates
                chat_formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], # Pass the raw prompt directly
                    tokenize=False
                )
                chat_formatted_prompts.append(chat_formatted)
            else:
                # For older models
                chat_formatted_prompts.append(prompt)
        
        outputs = []
        raw_prompts_map = {chat_prompt: raw_prompt for chat_prompt, raw_prompt in zip(chat_formatted_prompts, prompts)} # Store mapping

        # Process each prompt
        for i, chat_prompt in enumerate(tqdm(chat_formatted_prompts)):
            original_user_prompt = raw_prompts_map[chat_prompt] # Get original prompt
            try:
                # Truncate prompt if necessary
                processed_prompt = self.truncate_prompt(chat_prompt, tokenizer)

                # Generate with small batches to avoid OOM
                inputs = tokenizer(processed_prompt, return_tensors="pt").to(f"cuda:{self.device}")
                
                # Generate with temperature=0 for consistent outputs
                with torch.no_grad():
                    outputs_tensor = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=0.01,
                        top_p=0.9,
                        repetition_penalty=1.1
                    )
                    
                # Decode output tokens
                output_text = tokenizer.decode(outputs_tensor[0], skip_special_tokens=True)

                # --- Improved Output Cleaning --- 
                # 1. Remove the chat template formatting artifacts if possible
                # This depends heavily on the specific template used by the tokenizer
                # A common pattern is removing everything up to the first assistant message marker
                assistant_marker = "assistant\n"
                if assistant_marker in output_text.lower():
                    output_text = output_text.split(assistant_marker)[-1].strip()
                    output_text = re.sub(r'^[:\s]+', '', output_text) # Remove leading colons/spaces
                    
                # 2. Remove the original user prompt if it appears at the beginning
                # Need to handle potential minor variations (whitespace, etc.)
                # We use the original_user_prompt retrieved earlier
                # Check if the output starts with the user prompt (stripping whitespace for comparison)
                if output_text.strip().startswith(original_user_prompt.strip()):
                     # Find the end index of the prompt within the output
                     prompt_end_index = len(original_user_prompt)
                     # Add some flexibility for potential extra whitespace or minor changes
                     output_start_segment = output_text[:prompt_end_index + 10].strip()
                     if output_start_segment.startswith(original_user_prompt.strip()):
                         # Be careful with slicing, ensure we don't cut too much or too little
                         # Find the actual end based on the original prompt length
                         output_text = output_text[prompt_end_index:].strip()
                         output_text = re.sub(r'^[:\s]+', '', output_text) # Clean again after slicing

                # --- End Improved Output Cleaning ---

                # Post-process output based on task type
                processed_output = self.post_process_output(output_text, self.task_type)
                outputs.append(processed_output)
                
                # Print progress
                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(chat_formatted_prompts)} prompts")
                    
                # Free memory to avoid OOM
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error generating output for prompt {i}: {e}")
                outputs.append("")
                
        # Save results to file
        self.save_results(questions, outputs, answers, subjects)
        
        return outputs, answers, questions, subjects 