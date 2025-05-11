import os
import json
import re
import requests
import jsonlines
from tqdm import tqdm
from pathlib import Path
from model_gen import ModelGenerator

class OllamaBaichuanGenerator(ModelGenerator):
    """
    Generator class for Baichuan models through Ollama API
    结合baichuan_gen.py的处理逻辑和deepseek_gen.py的Ollama访问能力
    """
    def __init__(self, task_path, model_path=None, model_name="maxkb/baichuan2:13b-chat", device="0", is_few_shot=False,
                 few_shot_path=None, is_vllm=False, tensor_parallel_size=1,
                 gpu_memory_utilization=0.9, output_file=None, ollama_base_url="http://localhost:11434"):
        """
        Initialize Baichuan generator using Ollama API
        
        Args:
            task_path: Path to the task data file
            model_path: Not used for Ollama (can be None)
            model_name: Name of the model in Ollama (default: maxkb/baichuan2:13b-chat)
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
        
        # 确定特定任务类型
        filename = os.path.basename(task_path)
        if "5_qg" in filename:
            self.task_type = "question_generation"
        elif "3_zuowen" in filename:
            self.task_type = "essay_grading"
        elif "3_conversation_classification" in filename:
            self.task_type = "conversation_classification"
            
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
        
    def extract_score(self, text):
        """
        从文本中提取作文评分 - 基于baichuan_gen.py中的实现
        简洁而高效的评分提取方法
        """
        # 直接提取数字+分的模式
        score_patterns = [
            r'总分[:：\s]*?(\d+)[\s分]',
            r'得分[:：\s]*?(\d+)[\s分]', 
            r'分数[:：\s]*?(\d+)[\s分]',
            r'评分[:：\s]*?(\d+)[\s分]',
            r'(\d+)[\s]*?[/／][\s]*?100',
            r'(\d+)[\s]*?分'
        ]
        
        # 尝试匹配常见的分数表达方式
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                score = int(match.group(1))
                if 0 <= score <= 100:
                    print(f"找到分数: {score}")
                    return str(score)
        
        # 如果没有找到明确的分数表示，查找数字打分
        digits = re.findall(r'\b(\d{2,3})\b', text)
        possible_scores = [int(d) for d in digits if 60 <= int(d) <= 100]
        if possible_scores:
            print(f"从数字中提取可能的分数: {possible_scores[0]}")
            return str(possible_scores[0])
        
        print("无法提取分数，返回默认值75")
        return "75"  # 默认分数
    
    def extract_conversation_label(self, text):
        """
        提取对话分类标签，直接提取1-9的数字
        结合了baichuan的简洁性
        """
        print(f"提取对话标签，原始文本: {text[:50]}...")
        
        # 清理文本
        cleaned_text = text.strip()
        
        # 尝试提取明确的标签格式
        explicit_patterns = [
            r'分类[是为][:：\s]*?([1-9])\b',
            r'标签[是为][:：\s]*?([1-9])\b',
            r'类别[是为][:：\s]*?([1-9])\b',
            r'结果[是为][:：\s]*?([1-9])\b',
            r'对话类型[是为][:：\s]*?([1-9])\b'
        ]
        
        for pattern in explicit_patterns:
            match = re.search(pattern, cleaned_text)
            if match:
                print(f"找到明确标签: {match.group(1)}")
                return match.group(1)
        
        # 检查输出的最后一行是否只有一个数字
        last_line = cleaned_text.split('\n')[-1].strip()
        if re.match(r'^[1-9]$', last_line):
            print(f"使用最后一行的数字: {last_line}")
            return last_line
        
        # 搜索所有单独出现的1-9数字
        all_labels = re.findall(r'(?<!\d)([1-9])(?!\d)', cleaned_text)
        if all_labels:
            # 使用最后一个出现的数字
            print(f"使用文本中的最后一个数字: {all_labels[-1]}")
            return all_labels[-1]
            
        # 检查是否包含类别名称
        category_names = {
            '基础知识': '1', '个人信息': '2', '分析': '3', '归纳': '4', 
            '推断与迁移': '5', '回应与拓展': '6', '认同': '7', '质疑': '8', '指导': '9'
        }
        
        for name, num in category_names.items():
            if name in cleaned_text:
                print(f"根据类别名称找到标签: {name} -> {num}")
                return num
        
        print("无法提取对话标签，返回默认值-1")
        return "-1"
    
    def cleanup_generation_output(self, text):
        """
        清理创造型任务的输出，移除不必要的前缀和后缀
        保持简洁高效的处理
        """
        text = text.strip()
        
        # 移除常见的前缀
        prefixes_to_remove = [
            "请设计符合要求的题目：", "请提供完整的教学设计方案：",
            "下面是", "### ", "好的，", "以下是", "我将",
            "题目："
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                
        # 移除不必要的后缀
        unwanted_suffixes = ["\\n", "###", "希望这个", "希望以上", "以上是"]
        for suffix in unwanted_suffixes:
            if suffix in text[-20:]:  # 只检查最后20个字符
                text = text[:text.rfind(suffix)].strip()
                
        return text.strip()
    
    def post_process_output(self, output_text, task_type):
        """
        根据任务类型进行后处理
        采用和baichuan_gen类似的简洁处理逻辑
        """
        if task_type == "essay_grading":
            # 作文评分任务
            return self.extract_score(output_text)
        elif task_type == "conversation_classification":
            # 对话分类任务
            return self.extract_conversation_label(output_text)
        elif task_type == "question_generation" or task_type == "teaching_design":
            # 创造型任务
            return self.cleanup_generation_output(output_text)
        else:
            # 其他类型的任务
            return output_text.strip()
    
    def generate_output(self, tokenizer=None, model=None, batch_size=1, max_items=None, offset=0):
        """
        通过Ollama API生成输出
        专注于处理ollama_maxkb/baichuan2:13b-chat模型的特殊输出
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
                
                # 处理输出
                processed_output = self.post_process_output(generated_text, self.task_type)
                outputs.append(processed_output)
                
                # 每10个样本保存一次中间结果
                if (i+1) % 10 == 0 or i == len(prompts) - 1:
                    self.save_results(
                        questions[:len(outputs)],
                        outputs,
                        answers[:len(outputs)],
                        subjects[:len(outputs)] if subjects else None
                    )
                    print(f"已保存 {len(outputs)} 个结果")
                
            except requests.exceptions.Timeout:
                print(f"错误: 请求第{i+1}个提示词时超时")
                outputs.append("ERROR: API request timeout")
            except requests.exceptions.RequestException as e:
                print(f"错误: 请求第{i+1}个提示词时出错: {e}")
                outputs.append(f"ERROR: API request failed: {str(e)}")
            except Exception as e:
                print(f"错误: 处理第{i+1}个提示词时出现意外错误: {e}")
                outputs.append(f"ERROR: Unexpected error: {str(e)}")
        
        # 最终保存结果
        self.save_results(questions, outputs, answers, subjects)
        
        # 返回生成结果
        if subjects:
            return (outputs, answers, questions, subjects)
        else:
            return (outputs, answers, questions) 