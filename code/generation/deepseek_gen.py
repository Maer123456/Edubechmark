import os
import json
import re
import requests
import jsonlines
from tqdm import tqdm
from pathlib import Path
from model_gen import ModelGenerator

class DeepSeekGenerator(ModelGenerator):
    """
    Generator class for DeepSeek models through Ollama API
    """
    def __init__(self, task_path, model_path=None, model_name="deepseek-r1:32b", device="0", is_few_shot=False,
                 few_shot_path=None, is_vllm=False, tensor_parallel_size=1,
                 gpu_memory_utilization=0.9, output_file=None, ollama_base_url="http://localhost:11434"):
        """
        Initialize DeepSeek generator using Ollama API
        
        Args:
            task_path: Path to the task data file
            model_path: Not used for Ollama (can be None)
            model_name: Name of the model in Ollama (default: deepseek-r1:32b)
            ollama_base_url: Base URL for Ollama API
            Other parameters are passed to parent class but not used for Ollama
        """
        # 调用父类初始化，但model_path对Ollama无意义
        super().__init__(
            task_path=task_path,
            model_path=model_path or model_name,  # 如果没有model_path，使用model_name
            model_name=model_name,
            device=device,
            is_few_shot=is_few_shot,
            few_shot_path=few_shot_path,
            is_vllm=False,  # Ollama模式下不使用vLLM
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            output_file=output_file
        )
        
        # Ollama特定的属性
        self.ollama_base_url = ollama_base_url
        self.ollama_api_url = f"{ollama_base_url}/api/generate"
        print(f"使用Ollama API: {self.ollama_api_url}")
        print(f"目标模型: {self.model_name}")
        
        # 验证Ollama服务是否可用
        self._validate_ollama_service()
    
    def _validate_ollama_service(self):
        """验证Ollama服务是否可用"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            if self.model_name in model_names:
                print(f"✅ 成功连接到Ollama服务，并且找到目标模型: {self.model_name}")
            else:
                available_models = ", ".join(model_names[:5]) + ("..." if len(model_names) > 5 else "")
                print(f"⚠️ 成功连接到Ollama服务，但未找到目标模型: {self.model_name}")
                print(f"可用模型: {available_models}")
        except Exception as e:
            print(f"❌ 无法连接到Ollama服务: {e}")
            print(f"请确保Ollama服务在 {self.ollama_base_url} 运行中")

    def model_init(self):
        """
        对于Ollama API模式，无需初始化实际模型
        返回None作为模型和分词器
        """
        print("使用Ollama API模式，无需加载本地模型")
        return None, None

    def generate_output(self, tokenizer=None, model=None, batch_size=1, max_items=None, offset=0):
        """
        通过Ollama API生成输出
        不使用传入的tokenizer和model参数
        """
        # 首先准备提示词
        print(f"准备{self.task_type}任务的提示词...")
        prompts, answers, questions, subjects = self.prepare_prompts(max_items, offset)
        
        if not prompts:
            print("没有需要处理的提示词。退出。")
            return []
            
        print(f"处理 {len(prompts)} 个提示词")
        outputs = []
        
        # 通过Ollama API生成每个提示词的输出
        for i, prompt in enumerate(tqdm(prompts, desc="通过Ollama生成")):
            try:
                # 准备请求数据
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "top_k": 40,
                        "num_predict": 2048
                    }
                }
                
                # 发送请求
                response = requests.post(self.ollama_api_url, json=payload, timeout=180)
                response.raise_for_status()
                result = response.json()
                
                # 提取生成的文本
                generated_text = result.get('response', '').strip()
                
                # 处理输出（与标准模型处理方式相同）
                processed_output = self.post_process_output(generated_text, self.task_type)
                outputs.append(processed_output)
                
            except requests.exceptions.Timeout:
                print(f"错误: 请求第{i+1}个提示词时超时")
                outputs.append("ERROR: API request timeout")
            except requests.exceptions.RequestException as e:
                print(f"错误: 请求第{i+1}个提示词时出错: {e}")
                outputs.append(f"ERROR: API request failed: {str(e)}")
            except Exception as e:
                print(f"错误: 处理第{i+1}个提示词时出现意外错误: {e}")
                outputs.append(f"ERROR: Unexpected error: {str(e)}")
                
            # 可选：添加请求之间的延迟，避免API限制
            # import time
            # time.sleep(0.5)
        
        # 保存结果
        self.save_results(questions, outputs, answers, subjects)
        return outputs 

    def post_process_output(self, output_text, task_type):
        """
        根据任务类型处理输出文本
        增强父类的方法，只对特定任务类型进行特殊处理
        """
        # 对于需要增强处理的特定任务类型，应用自定义处理
        if task_type == "essay_grading":
            # 增强作文评分的提取逻辑
            return self.extract_enhanced_score(output_text)
        elif task_type == "conversation_classification":
            # 增强对话分类标签的提取逻辑
            return self.extract_enhanced_conversation_label(output_text)
        else:
            # 其他任务类型使用父类的方法
            return super().post_process_output(output_text, task_type)

    # 重命名为增强版本，避免与父类方法冲突
    def extract_enhanced_score(self, text):
        """增强版的作文评分提取逻辑"""
        print(f"使用增强版提取分数，原始文本: {text[:10]}...")
        
        # 定义更多分数模式，处理多种可能的格式
        score_patterns = [
            r'总分[:：\s]*?(\d+)[\s分]',
            r'得分[:：\s]*?(\d+)[\s分]',
            r'分数[:：\s]*?(\d+)[\s分]',
            r'评分[:：\s]*?(\d+)[\s分]',
            r'(\d+)[\s]*?[/／][\s]*?100',
            r'(\d+)[\s]*?分',
            r'给予(\d+)分',
            r'打分[:：\s]*?(\d+)',
            r'评分为[:：\s]*?(\d+)',
            r'成绩为[:：\s]*?(\d+)'
        ]
        
        # 尝试匹配各种模式
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                score = int(match.group(1))
                # 验证分数范围
                if 0 <= score <= 100:
                    return str(score)
        
        # 提取所有数字，寻找可能的分数
        all_numbers = re.findall(r'\b(\d{1,3})\b', text)
        for num in all_numbers:
            score = int(num)
            if 60 <= score <= 100:  # 只考虑合理范围内的分数
                return str(score)
        
        print("无法提取分数，使用父类方法")
        # 如果无法提取，调用父类方法
        return super().extract_score(text)

    def extract_enhanced_conversation_label(self, text):
        """增强版的对话分类标签提取逻辑"""
        # 清理文本
        cleaned_text = text.strip()
        
        # 尝试匹配常见的标签格式
        patterns = [
            r'分类[结果为是选择][:：\s]*?([1-9])\s*$',  # "分类结果为: 5"
            r'标签[结果为是选择][:：\s]*?([1-9])\s*$',   # "标签是: 3"
            r'结果[结果为是选择][:：\s]*?([1-9])\s*$',   # "结果为: 7"
            r'对话类型[结果为是选择][:：\s]*?([1-9])\s*$', # "对话类型是: 2"
            r'([1-9])\s*$',                           # 最后只有一个数字
            r'选择\s*([1-9])',                         # "选择 4"
            r'分类号\s*[:：]?\s*([1-9])',               # "分类号: 6"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned_text)
            if match:
                return match.group(1)
        
        # 检查最后一行，是否只包含一个数字
        last_line = cleaned_text.split('\n')[-1].strip()
        if re.match(r'^[1-9]$', last_line):
            return last_line
        
        # 查找文本中所有的1-9数字
        all_labels = re.findall(r'(?<!\d)([1-9])(?!\d)', cleaned_text)
        if all_labels:
            # 使用最后一个出现的标签
            return all_labels[-1]
        
        print("无法从输出文本中提取分类标签，使用默认值-1")
        return "-1"  # 默认返回-1表示未找到标签 