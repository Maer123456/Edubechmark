import os
import json
import re
import requests
import time
from tqdm import tqdm
from model_gen import ModelGenerator
import jsonlines
from openai import OpenAI  # 添加OpenAI客户端导入

class ApiGenerator(ModelGenerator):
    """
    Generator class for API-based models
    """
    def __init__(self, task_path, model_name="x1", base_url="https://spark-api-open.xf-yun.com/v2.1", 
                 api_key=None, device="0", is_few_shot=False, few_shot_path=None, output_file=None):
        """
        Initialize API generator
        """
        # Call parent initializer with minimal parameters
        super().__init__(
            task_path=task_path,
            model_path=None,  # Not applicable for API model
            model_name=model_name,
            device=device,
            is_few_shot=is_few_shot,
            few_shot_path=few_shot_path,
            is_vllm=False,  # Not applicable for API model
            output_file=output_file
        )
        
        # Check for TASK_TYPE_HINT environment variable
        task_type_hint = os.environ.get("TASK_TYPE_HINT", "")
        if task_type_hint:
            self.task_type = task_type_hint
            print(f"由于环境变量TASK_TYPE_HINT设置，任务类型被设为: {self.task_type}")
        
        # Check if task path contains '伦理' or '伦理2' 
        if "/伦理/" in self.task_path.lower() or "/伦理2/" in self.task_path.lower() or \
           "\\伦理\\" in self.task_path.lower() or "\\伦理2\\" in self.task_path.lower():
            self.task_type = "multiple_choice"
            print(f"检测到伦理相关目录路径，将任务类型设为: {self.task_type}")
        
        # API-specific attributes
        self.base_url = base_url
        self.api_key = api_key
        self.client = None  # 将在model_init中初始化OpenAI客户端
        
        # Parse API key as key_id:key_secret format
        if self.api_key and ":" in self.api_key:
            key_parts = self.api_key.split(":")
            self.key_id = key_parts[0]
            self.key_secret = key_parts[1]
        else:
            self.key_id = None
            self.key_secret = None
            print("警告: API密钥格式不正确，应为 'key_id:key_secret' 格式")

    def model_init(self):
        """
        Initialize OpenAI client for API access
        Returns None for model and tokenizer
        """
        # Check if API key is provided
        if not self.api_key:
            print("警告: 未提供API密钥，可能无法成功调用API")
        else:
            try:
                # 初始化OpenAI客户端
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                print(f"✅ 成功初始化OpenAI API客户端，基础URL: {self.base_url}, 模型: {self.model_name}")
            except Exception as e:
                print(f"❌ 初始化OpenAI客户端失败: {e}")
                self.client = None
            
        return None, None
    
    def generate_output(self, tokenizer=None, model=None, batch_size=1, max_items=None, offset=0):
        """
        Generate outputs via API
        Does not use the tokenizer and model parameters
        """
        # First prepare prompts
        print(f"准备{self.task_type}任务的提示词...")
        prompts, answers, questions, subjects = self.prepare_prompts(max_items, offset)
        
        if not prompts:
            print("没有需要处理的提示词。退出。")
            return [], [], [], []
            
        print(f"处理 {len(prompts)} 个提示词")
        outputs = []
        
        # 确保OpenAI客户端已初始化
        if self.client is None:
            self.model_init()
            if self.client is None:
                print("错误: 无法初始化API客户端，无法继续生成")
                return [], [], [], []
        
        # Generate outputs for each prompt via API
        for i, prompt in enumerate(tqdm(prompts, desc="通过API生成")):
            try:
                # 使用OpenAI客户端发送请求
                chat_completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2048
                )
                
                # 提取生成的文本
                generated_text = chat_completion.choices[0].message.content
                
                # Post-process output (same as standard model)
                processed_output = self.post_process_output(generated_text, self.task_type)
                outputs.append(processed_output)
                
                # 输出token使用情况
                if hasattr(chat_completion, 'usage') and chat_completion.usage:
                    print(f"Token使用 #{i+1}: 提示={chat_completion.usage.prompt_tokens}, "
                          f"完成={chat_completion.usage.completion_tokens}, "
                          f"总计={chat_completion.usage.total_tokens}")
                
            except Exception as e:
                print(f"错误: 处理第{i+1}个提示词时出现错误: {e}")
                outputs.append(f"ERROR: API request failed: {str(e)}")
                
            # Optional: Add delay between requests to avoid API rate limits
            time.sleep(0.5)
        
        # Save results
        self.save_results(questions, outputs, answers, subjects)
        return outputs, answers, questions, subjects 