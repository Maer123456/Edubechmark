import json
import os
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_gen import ModelGenerator
import jsonlines
from pathlib import Path

class QwenGenerator(ModelGenerator): # Rename class
    """
    Generator class for Qwen models
    """
    def __init__(self, task_path, model_path, model_name, device="0", is_few_shot=False,
                 few_shot_path=None, is_vllm=False, tensor_parallel_size=1,
                 gpu_memory_utilization=0.9, output_file=None):
        """
        Initialize Qwen generator
        """
        # Convert task_path to absolute path if it's not already
        if not os.path.isabs(task_path):
            task_path = os.path.abspath(task_path)

        # Ensure the file exists
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"Task file not found: {task_path}")

        # Determine task type based on filename (Keep this logic)
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
        Initialize Qwen model and tokenizer
        """
        if self.is_vllm:
            # Use vLLM for faster inference
            return self.model_init_vllm() # Assumes model_gen.py has vLLM init

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

            # Explicitly set the chat template if it's missing
            if tokenizer.chat_template is None:
                print("Tokenizer missing chat template. Setting Qwen default template.")
                tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{% if loop.first %}"
                    "{% if message['role'] == 'system' %}"
                    "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% else %}"
                    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% endif %}"
                    "{% elif message['role'] == 'user' %}"
                    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% elif message['role'] == 'assistant' %}"
                    "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                    "{{ '<|im_start|>assistant\n' }}"
                    "{% endif %}"
                )

            print(f"Loading model from {self.model_path}")
            # Keep settings adjusted for Qwen-14B
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": f"cuda:{self.device}",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "max_memory": {0: "40GiB"}, # <-- Increase memory limit further
                # Let transformers decide attn implementation
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
            print("Trying to load with low_cpu_mem_usage=True (fallback)")
            # Try again with low_cpu_mem_usage (fallback, keep params consistent)
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
            
            # Explicitly set the chat template if it's missing (in except block too)
            if tokenizer.chat_template is None:
                print("Tokenizer missing chat template (fallback). Setting Qwen default template.")
                tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{% if loop.first %}"
                    "{% if message['role'] == 'system' %}"
                    "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% else %}"
                    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% endif %}"
                    "{% elif message['role'] == 'user' %}"
                    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% elif message['role'] == 'assistant' %}"
                    "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                    "{{ '<|im_start|>assistant\n' }}"
                    "{% endif %}"
                )

            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": f"cuda:{self.device}",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "max_memory": {0: "40GiB"}, # <-- Increase memory limit further
                # Let transformers decide attn implementation
            }

            if is_local_dir:
                model_kwargs["local_files_only"] = True

            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )

            model.eval()

        return model, tokenizer

    # --- Task-specific post-processing functions (Keep these as they are) ---
    def extract_choice_answer(self, text):
        """
        提取选择题答案，特别优化对伦理题目的支持
        增强父类的提取逻辑
        """
        # 首先清理文本
        cleaned_text = text.strip()
        
        # 伦理题目常见答案前缀的专门处理
        if "正确选项" in cleaned_text:
            cleaned_text = cleaned_text.split("正确选项")[1].strip()
            cleaned_text = cleaned_text.strip("：").strip(":").strip()
            if re.match(r'^[A-Da-d]$', cleaned_text):
                print(f"从'正确选项'后提取答案: {cleaned_text.upper()}")
                return cleaned_text.upper()
                
        # 伦理题目专用模式
        ethics_patterns = [
            r'(?:正确答案|正确选项|正确的选项)[是为：:]\s*([A-Da-d])',
            r'(?:答案|选择|应选)[是为：:]*\s*([A-Da-d])\s*[。.]?$',
            r'(?:在伦理上|从伦理角度)[^A-Da-d]*([A-Da-d])[是为]?正确'
        ]
        
        for pattern in ethics_patterns:
            match = re.search(pattern, cleaned_text)
            if match:
                print(f"使用伦理题目专用模式提取答案: {match.group(1).upper()}")
                return match.group(1).upper()
        
        # 如果以上模式都未匹配到，则使用父类的方法
        return super().extract_choice_answer(text)

    def cleanup_essay_output(self, text):
        """
        清理作文输出，增强父类的实现以处理Qwen模型的特殊格式
        """
        # 先清理Qwen特有的前缀和后缀
        text = text.strip()
        qwen_prefixes = [
            "请根据以下题目写一篇作文。", 
            "<|im_start|>assistant\n",
            "<|im_end|>",
            "<|im_start|>user\n"
        ]
        
        for prefix in qwen_prefixes:
            if prefix in text:
                text = text.replace(prefix, "")
                
        # 调用父类方法进行通用清理
        return super().cleanup_essay_output(text)

    def extract_score(self, text):
        """
        增强父类提取分数的方法，适用于Qwen模型的输出格式
        """
        # 清理Qwen特定的标记
        cleaned_text = text.replace("<|im_start|>assistant\n", "").replace("<|im_end|>", "")
        
        # 首先尝试提取明确的分数格式
        qwen_score_patterns = [
            r'总体评分[:：]\s*?(\d+)\s*分',
            r'总分[:：]\s*?(\d+)[/／]100',
            r'最终得分[:：]\s*?(\d+)'
        ]
        
        for pattern in qwen_score_patterns:
            match = re.search(pattern, cleaned_text)
            if match:
                score = int(match.group(1))
                if 0 <= score <= 100:
                    print(f"从Qwen特有格式提取分数: {score}")
                    return str(score)
        
        # 如果没有提取到，使用父类方法
        return super().extract_score(cleaned_text)

    def extract_conversation_label(self, text):
        """
        从对话分类任务的输出中提取标签 (1-9)
        """
        cleaned_text = text.strip().lower() # 转小写处理

        # 优先匹配明确的标签指示
        patterns = [
            r'(?:分类|标签|类别)[是为：:]?\s*(\d)', # 分类是: 1
            r'属于第?\s*(\d)\s*[类组别]',      # 属于第 1 类
            r'对话的类别是\s*(\d)',             # 对话的类别是 1
            r'label[:\s]*(\d)',                 # label: 1 (英文)
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned_text)
            if match:
                label = int(match.group(1))
                if 1 <= label <= 9: # 检查标签范围
                    print(f"Extracted label {label} using pattern: {pattern}")
                    return str(label)
                else:
                    print(f"Extracted label {label} out of range (1-9).")

        # 如果没有明确指示，查找句子末尾或单独存在的数字 (1-9)
        end_patterns = [
            r'\b(\d)\b[.。]?$',     # 结尾是单个数字
            r'^\s*(\d)\s*$'        # 整行只有一个数字
        ]
        for pattern in end_patterns:
             match = re.search(pattern, cleaned_text)
             if match:
                label = int(match.group(1))
                if 1 <= label <= 9:
                    print(f"Extracted label {label} from end/standalone pattern: {pattern}")
                    return str(label)

        # 如果无法提取有效标签，尝试使用父类方法
        parent_result = super().post_process_output(text, "conversation_classification")
        if parent_result != "-1":
            return parent_result

        # 如果都找不到，返回默认值
        print(f"Could not extract a valid label (1-9) from: '{cleaned_text}'. Returning default -1.")
        return "-1"

    def post_process_output(self, output_text, task_type):
        """
        根据任务类型后处理模型输出，继承并增强父类方法
        """
        print(f"Post-processing output for task type: {task_type}") # Debug print

        if task_type == "multiple_choice":
            return self.extract_choice_answer(output_text)
        elif task_type == "short_answer":
            # 对于简答题，我们通常只需要基本的清理
            return self.cleanup_generation_output(output_text) # 使用父类或重写
        elif task_type == "essay":
            return self.cleanup_essay_output(output_text)
        elif task_type == "essay_grading":
            return self.extract_score(output_text)
        elif task_type == "conversation_classification": # <--- 添加处理逻辑
            return self.extract_conversation_label(output_text) # <--- 调用新方法
        elif task_type == "question_generation" or task_type == "teaching_design":
            # 对于生成任务，也使用基本清理
            return self.cleanup_generation_output(output_text) # 使用父类或重写
        else:
            # 对于未知或其他任务类型，调用父类的通用处理
            print(f"Task type '{task_type}' not explicitly handled, using default parent processing.")
            return super().post_process_output(output_text, task_type)

    def generate_output(self, tokenizer, model, batch_size=1, max_items=100, offset=0):
        """
        Generate model outputs for educational tasks using Qwen model specifics
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
        # 准备提示词，使用父类的方法
        prompts, answers, questions, subjects = self.prepare_prompts(max_items=max_items, offset=offset)

        if not prompts:
            print("No prompts were prepared, check if the task file is empty or invalid.")
            return [], [], [], [] # <-- Return empty lists including subjects

        print(f"Generating outputs for {len(prompts)} prompts, batch size {batch_size}")

        # Qwen特有的处理：将提示词转换为对话格式
        chat_formatted_prompts = []
        if hasattr(tokenizer, "apply_chat_template"):
            print("Using tokenizer.apply_chat_template for Qwen prompt formatting.")
            for prompt in prompts:
                # Apply Qwen's chat template
                # Assuming the template expects a user message
                chat_formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True # Important for inference
                )
                chat_formatted_prompts.append(chat_formatted)
        else:
            print("Warning: tokenizer does not have apply_chat_template. Using raw prompts.")
            chat_formatted_prompts = prompts # Fallback

        outputs = []
        # Store mapping from formatted prompt to original prompt for cleaning
        raw_prompts_map = {chat_prompt: raw_prompt for chat_prompt, raw_prompt in zip(chat_formatted_prompts, prompts)}

        for i, chat_prompt in enumerate(tqdm(chat_formatted_prompts)):
            original_user_prompt = raw_prompts_map.get(chat_prompt, "") # Get original prompt safely
            try:
                # Truncate prompt if necessary (using the formatted prompt length)
                processed_prompt = self.truncate_prompt(chat_prompt, tokenizer)

                inputs = tokenizer(processed_prompt, return_tensors="pt").to(f"cuda:{self.device}")

                # Generate with parameters suitable for consistent chat output
                with torch.no_grad():
                    outputs_tensor = model.generate(
                        **inputs,
                        max_new_tokens=512, # Adjust if needed
                        do_sample=False,    # Use greedy decoding for consistency
                        temperature=0.01,   # Low temperature for deterministic output
                        top_p=0.9,          # Top-p sampling (less relevant with do_sample=False)
                        repetition_penalty=1.1 # Penalize repetition
                    )

                # Decode output tokens, keeping special tokens to properly find assistant response
                output_text_full = tokenizer.decode(outputs_tensor[0], skip_special_tokens=False)

                # --- Qwen Output Cleaning (Revised) ---
                output_text = "" # Initialize empty output
                assistant_marker = "<|im_start|>assistant\n"
                end_marker = "<|im_end|>"

                # Find the start of the assistant's response
                assist_start_index = output_text_full.rfind(assistant_marker)
                if assist_start_index != -1:
                    # Extract text after the last assistant marker
                    start_pos = assist_start_index + len(assistant_marker)
                    output_text_raw = output_text_full[start_pos:].strip()
                    
                    # Find the end marker after the assistant start
                    end_pos = output_text_raw.find(end_marker)
                    if end_pos != -1:
                        # Extract content before the end marker
                        output_text = output_text_raw[:end_pos].strip()
                    else:
                        # If no end marker found after assistant start, take the rest (might happen with truncation)
                        output_text = output_text_raw
                else:
                    # Fallback if assistant marker isn't found (shouldn't happen with chat template)
                    # Try the old method using skip_special_tokens=True as a last resort
                    print(f"Warning: Assistant marker not found in output for prompt {i}. Using fallback cleaning.")
                    output_text_skipped = tokenizer.decode(outputs_tensor[0], skip_special_tokens=True)
                    if original_user_prompt and output_text_skipped.strip().startswith(original_user_prompt.strip()):
                        output_text = output_text_skipped[len(original_user_prompt):].strip()
                        output_text = re.sub(r'^[:\s]+', '', output_text)
                    else:
                         output_text = output_text_skipped # Use the skipped version directly

                # --- End Qwen Output Cleaning ---

                # 使用统一的后处理方法
                processed_output = self.post_process_output(output_text, self.task_type)
                outputs.append(processed_output)

                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(chat_formatted_prompts)} prompts")
                    # 定期保存中间结果
                    self.save_results(
                        questions[:len(outputs)],
                        outputs,
                        answers[:len(outputs)],
                        subjects[:len(outputs)] if subjects else None
                    )

                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error generating output for prompt {i}: {e}")
                outputs.append("") # Append empty string on error

        # 最终保存结果
        self.save_results(questions, outputs, answers, subjects)
        
        # 返回结果
        if subjects:
            return outputs, answers, questions, subjects
        else:
            return outputs, answers, questions 