import json
import os
import torch
import jsonlines
from vllm import LLM, SamplingParams
from pathlib import Path
import re
from tqdm import tqdm

# 添加课堂对话分类标签系统
DIALOGUE_LABEL_SYSTEM = {
    1: {"name": "基础知识", "description": "参照教科书或教师以前教过的知识，可以判断出正确或错误的答案。", "examples": ["课本上说put forward和suggest意思相同。"]},
    2: {"name": "个人信息", "description": "说话人生活中的事件，不被认为是其他参与者知道的；个人对神情或艺术作品等的想象性反应；发言人对个人关系或情况的个人看法。", "examples": ["我努力工作了一年，终于获得了一等奖。"]},
    3: {"name": "分析", "description": "将一个整体抽象地分离其组成部分，以研究这些部分及其关系；它涉及推理，使知识变得明朗和易于理解。", "examples": ["我不认为这是最好的方式，因为有些人可能会如此选择所有的朋友。"]},
    4: {"name": "归纳", "description": "通过对详细事实进行推理而形成一般概念的过程；它涉及到归纳推理和思想的发展，目的是对信息进行以外的问题作出回应。", "examples": ["我比较了文章A和文章B，发现它们都研究了两个变量之间的关系。"]},
    5: {"name": "推断与迁移", "description": "对可能性的考虑，超越了目前的知识水平；但基于理论或事实依据。", "examples": ["汤姆刚做了一笔成功的投资，他哥哥也许能给他提一些资金。"]},
    6: {"name": "回应与拓展", "description": "这里的问题涉及到别人之前的回答被动态地用来吸收；可以通过评论来实现，明确强调之前的回应，并在此基础上发展。", "examples": ["汤姆希望你开了那些接子。但他怎么知道布拉德在那里？"]},
    7: {"name": "认同", "description": "对陈述的明确接受或同意。", "examples": ["太棒了", "很好", "好的", "我同意"]},
    8: {"name": "质疑", "description": "怀疑、完全/部分不同意，质疑或拒绝一个陈述，包括一个简单的\"no\"回答，当它表示拒绝一个想法，而不是回答一个问题。", "examples": ["你真的认为有度是一样的吗？"]},
    9: {"name": "指导", "description": "根据学生的学习速度和认知水平提供帮助和支持；老师对如何组织学习活动出明确的指导，并要求其他人做出相应的反应。", "examples": ["做完后检查一下答案。"]}
}

class ModelGenerator:
    """
    Base class for generating model outputs for educational evaluation tasks.
    Standardized implementation that can be used as a benchmark for different models.
    """
    def __init__(self, task_path, model_path, model_name, device="0", is_few_shot=False, 
                 few_shot_path=None, is_vllm=False, tensor_parallel_size=1, 
                 gpu_memory_utilization=0.9, output_file=None):
        """
        Args:
            task_path: Path to the task data file
            model_path: Path to the model weights
            model_name: Name of the model
            device: GPU device to use
            is_few_shot: Whether to use few-shot examples
            few_shot_path: Path to few-shot examples
            is_vllm: Whether to use vLLM for acceleration
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            output_file: File path to save outputs
        """
        # Convert task_path to absolute path if it's not already
        if not os.path.isabs(task_path):
            task_path = os.path.abspath(task_path)

        # Ensure the file exists
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"Task file not found: {task_path}")
            
        self.task_path = task_path
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.is_few_shot = is_few_shot
        self.few_shot_path = few_shot_path
        
        if is_few_shot and few_shot_path is None:
            raise ValueError("Few-shot path must be provided when is_few_shot=True")
            
        self.is_vllm = is_vllm
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        
        self.output_file = output_file
        
        # Determine task type based on file path
        self.task_type = self._determine_task_type()
        
    def _determine_task_type(self):
        """
        Determine the task type from the file name using standardized rules
        """
        # Determine task type based on filename
        filename = os.path.basename(self.task_path)
        filepath = self.task_path.lower()  # 转为小写以便于匹配
        self.task_type = "unknown" # Default task type
        
        # 先检查文件路径中是否包含伦理目录，这是优先级最高的
        if "/伦理/" in filepath or "/伦理2/" in filepath or "\\伦理\\" in filepath or "\\伦理2\\" in filepath:
            self.task_type = "multiple_choice"
            print(f"Determined task type for {filename} (based on ethics directory): {self.task_type}")
            return self.task_type
        
        # 然后按照文件名匹配规则处理其他类型
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
        # 新增题目生成任务类型
        elif "5_qg_" in filename or "qg_100.jsonl" in filename:
            self.task_type = "question_generation"
        # 新增教学设计任务类型
        elif "5_teachingdesign_" in filename or "teachingdesign_50.jsonl" in filename:
            self.task_type = "teaching_design"
        # 新增课堂对话分类任务类型
        elif "3_conversation_classification" in filename:
            self.task_type = "conversation_classification"
        else:
            # Additional fallback rules
            if "writing" in filename:
                self.task_type = "essay"
            elif "shige" in filename or "yuedu" in filename:
                self.task_type = "short_answer"
            elif "zuowen" in filename:
                self.task_type = "essay_grading"
            elif "qg" in filename:
                self.task_type = "question_generation"
            elif "teachingdesign" in filename:
                self.task_type = "teaching_design"
            elif "conversation_classification" in filename:
                self.task_type = "conversation_classification"
            # 伦理相关文件名识别
            elif "伦理" in filename or "ethics" in filename.lower():
                self.task_type = "multiple_choice"
            
        # Debug print to confirm task type
        print(f"Determined task type for {filename}: {self.task_type}")
        if self.task_type == "unknown":
            print(f"Warning: Could not determine task type for file {filename}. Please check filename conventions.")
            
        return self.task_type
        
    def model_init_vllm(self):
        """Initialize model using vLLM for faster inference"""
        model = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True
        )
        tokenizer = model.get_tokenizer()
        return model, tokenizer
    
    def model_init(self):
        """
        Initialize model and tokenizer - each model subclass must implement this
        """
        raise NotImplementedError("Subclasses must implement model_init")
    
    def load_few_shot_examples(self):
        """
        加载few-shot示例，根据任务类型从few_shot目录读取对应的示例
        
        Returns:
            str: 格式化的few-shot示例文本
        """
        if not self.is_few_shot or not self.few_shot_path:
            return ""
            
        if not os.path.exists(self.few_shot_path):
            print(f"Few-shot example file not found: {self.few_shot_path}")
            return ""
            
        # 读取few-shot示例文件
        try:
            with jsonlines.open(self.few_shot_path) as f:
                examples = list(f)[:3]  # 默认使用前3个示例
                
            if not examples:
                print(f"No examples found in {self.few_shot_path}")
                return ""
                
            examples_text = "以下是几个示例：\n\n"
            
            # 根据不同的任务类型构造不同的示例格式
            for i, ex in enumerate(examples, 1):
                if self.task_type == "multiple_choice":
                    options = ex.get("options", [])
                    if options:
                        options_text = "\n".join([f"{opt['id']}. {opt['content']}" for opt in options])
                        examples_text += f"示例{i}:\n问题: {ex['ques_content']}\n{options_text}\n答案: {ex['ques_answer']}\n\n"
                    else:
                        examples_text += f"示例{i}:\n问题: {ex['ques_content']}\n答案: {ex['ques_answer']}\n\n"
                        
                elif self.task_type == "short_answer":
                    examples_text += f"示例{i}:\n问题: {ex['ques_content']}\n答案: {ex['ques_answer']}\n\n"
                    
                elif self.task_type == "essay":
                    # 限制作文长度避免prompt过长
                    answer_preview = ex.get('ques_answer', '')[:300]
                    if len(ex.get('ques_answer', '')) > 300:
                        answer_preview += "..."
                    examples_text += f"示例{i}:\n题目: {ex['ques_content']}\n作文: {answer_preview}\n\n"
                    
                elif self.task_type == "essay_grading":
                    # 更正：使用与prepare_prompts相同的字段名
                    essay_title = ex.get("question", "")
                    essay_content = ex.get("ques_answer", "")    
                
                    score = ex.get("score", "")
                    
                    examples_text += f"示例{i}:\n作文题目: {essay_title}\n作文内容: {essay_content}\n评分: {score}\n\n"
                    
                elif self.task_type == "question_generation":
                    # 题目生成任务
                    grade = ex.get("grade", "")
                    knowledge_point = ex.get("knowledge_point", "")
                    task_description = ex.get("task_description", "")
                    answer = ex.get("answer", "")
                    
                    examples_text += f"示例{i}:\n年级: {grade}\n知识点: {knowledge_point}\n任务描述: {task_description}\n生成的题目: {answer}\n\n"
                    
                elif self.task_type == "teaching_design":
                    # 教学设计任务 - 更正为与prepare_prompts相同的字段
                    grade = ex.get("grade", "")
                    subject = ex.get("subject", "")
                    topic = ex.get("topic", "")
                    teaching_design_requirements = ex.get("teaching_design_requirements", "")
                    answer = ex.get("answer", "")
                  
                        
                    examples_text += f"示例{i}:\n年级: {grade}\n学科: {subject}\n主题: {topic}\n教学设计要求: {teaching_design_requirements}\n教学设计: {answer}\n\n"
                    
                elif self.task_type == "conversation_classification":
                    # 对话分类任务
                    dialogue = ex.get("dialogue", "")
                    label = ex.get("label", "")
                    
                    examples_text += f"示例{i}:\n对话内容: {dialogue}\n类别: {label}\n\n"
                    
                else:
                    # 默认格式
                    examples_text += f"示例{i}:\n问题: {ex.get('ques_content', '')}\n答案: {ex.get('ques_answer', '')}\n\n"
            
            examples_text += "现在，请回答下面的问题：\n"
            return examples_text
            
        except Exception as e:
            print(f"Error loading few-shot examples: {e}")
            return ""
    
    def truncate_prompt(self, prompt, tokenizer, max_length=2048):
        """Truncate prompt to fit within model context window"""
        encoded = tokenizer.encode(prompt)
        if len(encoded) > max_length:
            print(f"Input too long, truncating to {max_length} tokens")
            half_length = max_length // 2
            # Keep beginning and end, cut out middle
            tokens = encoded[:half_length] + encoded[-half_length:]
            prompt = tokenizer.decode(tokens)
        return prompt
    
    def prepare_prompts(self, max_items=500, offset=0):
        """
        Prepare standardized prompts for all models based on task type
        Returns: list of prompts, list of answers, list of questions, list of subjects
        
        Args:
            max_items: Maximum number of items to process in a single batch
            offset: Starting offset for data processing (for batch processing)
        """
        questions = []
        prompts = []
        answers = []
        subjects = []
        
        # 加载few-shot示例
        few_shot_examples = ""
        if self.is_few_shot and self.few_shot_path:
            print(f"Loading few-shot examples from {self.few_shot_path}")
            try:
                few_shot_examples = self.load_few_shot_examples()
                print(f"Loaded few-shot examples: {len(few_shot_examples.split())} words")
            except Exception as e:
                print(f"Error loading few-shot examples: {e}")
                few_shot_examples = ""
        
        # Ensure task_path exists
        if not os.path.exists(self.task_path):
            raise FileNotFoundError(f"Task file not found: {self.task_path}\nCurrent working directory: {os.getcwd()}")
        
        print(f"Opening task file: {self.task_path}")
        
        # 计算文件中的数据条数
        item_count = 0
        with jsonlines.open(self.task_path) as f:
            for _ in f:
                item_count += 1
        
        print(f"Total items in file: {item_count}")
        
        if offset >= item_count:
            print(f"Warning: Offset {offset} exceeds total items {item_count}")
            return [], [], [], []
            
        # 根据offset和max_items计算实际要处理的数据范围
        if max_items is None:
            # If max_items is None, process all items from the offset
            end_pos = item_count
        else:
            # Otherwise, calculate the end position based on max_items
            end_pos = min(offset + max_items, item_count)

        print(f"Processing items from {offset} to {end_pos-1} (total: {end_pos-offset})")
        
        # 读取数据
        processed_count = 0
        current_pos = 0
        with jsonlines.open(self.task_path) as f:
            for item in f:
                # 跳过offset之前的数据
                if current_pos < offset:
                    current_pos += 1
                    continue
                    
                # 如果达到了结束位置，停止处理
                if current_pos >= end_pos:
                    break
                    
                current_pos += 1
                
                subject = item.get("subject", "")
                
                # 特别处理课堂对话分类任务
                if self.task_type == "conversation_classification":
                    dialogue = item.get("dialogue", "")
                    label = item.get("label", "")
                    
                    # 保存问题和答案，用于结果保存
                    question = f"对话内容：{dialogue}"
                    questions.append(question)
                    answers.append(label)
                    
                    # 构造系统提示词和用户提示词
                    label_descriptions = []
                    for label_num, label_info in DIALOGUE_LABEL_SYSTEM.items():
                        description = f"{label_num}. {label_info['name']}: {label_info['description']}"
                        label_descriptions.append(description)
                    
                    system_prompt = "你是一个专业的教育对话分类器。请根据以下9种类别对给定的对话内容进行分类：\n"
                    system_prompt += "\n".join(label_descriptions)
                    system_prompt += "\n请只返回分类的数字标签（1-9）。"
                    
                    user_prompt = f"对话内容：{dialogue}\n\n请问这段对话属于哪个类别？只需返回分类的数字标签（1-9）。"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        user_prompt = few_shot_examples + "\n" + user_prompt
                    
                    # 组合提示词
                    prompt = f"{system_prompt}\n\n{user_prompt}"
                    prompts.append(prompt)
                
                # 处理选择题
                elif self.task_type == "multiple_choice":
                    options = item.get("options", [])
                    question_content = item.get("ques_content", "")
                    answer = item.get("ques_answer", "")
                    
                    # 构造问题和选项
                    if options:
                        # 如果有单独的选项字段，格式化它们
                        options_text = "\n".join([f"{opt['id']}. {opt['content']}" for opt in options])
                        question_with_options = f"{question_content}\n\n{options_text}"
                    else:
                        # 否则直接使用问题内容
                        question_with_options = question_content
                    
                    questions.append(question_with_options)
                    answers.append(answer)
                    
                    # 构造提示词
                    prompt = f"请回答下面的选择题，直接给出选项字母即可（如A、B、C或D）:\n\n{question_with_options}"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        prompt = few_shot_examples + "\n" + prompt
                    
                    prompts.append(prompt)
                
                # 处理简答题
                elif self.task_type == "short_answer":
                    question_content = item.get("ques_content", "")
                    answer = item.get("ques_answer", "")
                    
                    questions.append(question_content)
                    answers.append(answer)
                    
                    # 构造提示词
                    prompt = f"请回答下面的问题：\n\n{question_content}"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        prompt = few_shot_examples + "\n" + prompt
                    
                    prompts.append(prompt)
                
                # 处理作文题
                elif self.task_type == "essay":
                    question_content = item.get("ques_content", "")
                    # 作文题可能没有标准答案
                    answer = item.get("ques_answer", "")
                    
                    questions.append(question_content)
                    answers.append(answer)
                    
                    # 构造提示词
                    prompt = f"请根据以下题目写一篇作文：\n\n{question_content}"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        prompt = few_shot_examples + "\n" + prompt
                    
                    prompts.append(prompt)
                
                # 处理作文评分
                elif self.task_type == "essay_grading":
                    # 使用question字段作为作文题目，ques_answer字段作为学生作答内容，score字段作为参考分数
                    essay_title = item.get("question", "")
                    essay_content = item.get("ques_answer", "")
                    # 获取参考分数，如果存在，否则为空字符串
                    score = item.get("score", "")
                    
                    # 组合题目和内容作为问题
                    question_info = f"作文题目：{essay_title}\n\n作文内容：\n{essay_content}"
                    questions.append(question_info)
                    answers.append(score)
                    
                    # 构造提示词
                    prompt = f"请对下面的作文进行评分（满分100分）不需要解释理由：\n\n作文题目：{essay_title}\n\n作文内容：\n{essay_content}"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        prompt = few_shot_examples + "\n" + prompt
                    
                    prompts.append(prompt)
                
                # 处理题目生成任务
                elif self.task_type == "question_generation":
                    task_description = item.get("task_description", "")
                    knowledge_point = item.get("knowledge_point", "")
                    grade = item.get("grade", "")
                    
                    # 组合问题信息
                    question_info = f"年级: {grade}\n知识点: {knowledge_point}\n任务: {task_description}"
                    questions.append(question_info)
                    
                    # 获取标准答案（如果有）
                    answer = item.get("answer", "")
                    answers.append(answer)
                    
                    # 构造提示词
                    prompt = f"请设计符合要求的题目：\n\n年级: {grade}\n知识点: {knowledge_point}\n任务描述: {task_description}"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        prompt = few_shot_examples + "\n" + prompt
                    
                    prompts.append(prompt)
                
                # 处理教学设计任务
                elif self.task_type == "teaching_design":
                    # 更正：使用正确的字段名
                    grade = item.get("grade", "")
                    subject = item.get("subject", "")
                    topic = item.get("topic", "")
                    teaching_design_requirements = item.get("teaching_design_requirements", "")
                    
                    # 组合问题信息
                    question_info = f"年级: {grade}\n学科: {subject}\n主题: {topic}\n教学设计要求: {teaching_design_requirements}"
                    questions.append(question_info)
                    
                    # 获取标准答案（如果有）
                    answer = item.get("answer", "")
                    answers.append(answer)
                    
                    # 构造提示词
                    prompt = f"请提供完整的教学设计方案：\n\n年级: {grade}\n学科: {subject}\n主题: {topic}\n教学设计要求: {teaching_design_requirements}"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        prompt = few_shot_examples + "\n" + prompt
                    
                    prompts.append(prompt)
                
                # 对于未知任务类型，使用通用处理
                else:
                    question_content = item.get("ques_content", "")
                    answer = item.get("ques_answer", "")
                    
                    questions.append(question_content)
                    answers.append(answer)
                    
                    # 构造提示词
                    prompt = f"请回答下面的问题：\n\n{question_content}"
                    
                    # 添加few-shot示例
                    if few_shot_examples:
                        prompt = few_shot_examples + "\n" + prompt
                    
                    prompts.append(prompt)
                
                # 保存学科信息
                subjects.append(subject)
                
                processed_count += 1
                
        print(f"Prepared {processed_count} prompts")
        
        return prompts, answers, questions, subjects
    
    # --- STANDARDIZED POST-PROCESSING METHODS FROM QwenGenerator ---
    
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
        清理作文输出，移除思考过程和系统提示内容
        """
        text = text.strip()
        prefixes_to_remove = [
            "请根据以下题目写一篇作文。", "题目:", "作文:", "下面是", "### ", "好的，"
        ]
        for prefix in prefixes_to_remove:
            if prefix in text:
                parts = text.split(prefix, 1)
                if len(parts) > 1: text = parts[1].strip()
        unwanted_suffixes = ["\\n", "###", "我希望", "总结：", "总之，"]
        for suffix in unwanted_suffixes:
            if text.endswith(suffix): text = text[:text.rfind(suffix)].strip()
        return text.strip()

    def extract_score(self, text):
        """
        从文本中提取分数，确保只返回1-100的数字
        """
        print(f"提取分数的原始文本: {text[:200]}...")
        eval_pattern = r'(?:评分|得分|分数)[：:]\s*(\d+)(?:分)?'
        match = re.search(eval_pattern, text)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 100: print(f"找到明确评分格式，分数为: {score}"); return str(score)
        score_patterns = [
            r'(?:分数|评分|得分|成绩)(?:为|是|应该是|应为|给|打)[^\d]*?(\d+)(?:分)?',
            r'(?:给予|打|评|得|获得)[^\d]*?(\d+)(?:分)?',
            r'(\d+)[^\d]*?(?:分)',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                score = int(match.group(1))
                if 1 <= score <= 100: print(f"找到分数: {score}"); return str(score)
        # 如果没有找到匹配的评分，则尝试提取文本中出现的任何1-100之间的数字
        number_pattern = r'\b(\d{1,3})\b'
        matches = re.findall(number_pattern, text)
        for match in matches:
            score = int(match)
            if 1 <= score <= 100: print(f"从文本中提取到数字: {score}"); return str(score)
        print("无法提取分数，返回默认值75")
        return "75"  # 如果无法提取分数，返回默认值

    def post_process_output(self, output_text, task_type):
        """
        统一后处理输出文本，根据任务类型应用不同的处理方法
        """
        if task_type == "multiple_choice":
            return self.extract_choice_answer(output_text)
        elif task_type == "essay":
            return self.cleanup_essay_output(output_text)
        elif task_type == "essay_grading":
            return self.extract_score(output_text)
        elif task_type == "conversation_classification":
            # 对于对话分类任务，提取1-9之间的数字标签
            cleaned_text = output_text.strip()
            
            # 直接使用与base_classifier.py相同的逻辑
            # 尝试找到响应中的任何数字
            digits = re.findall(r'\d+', cleaned_text)
            if digits:
                # 取第一个在范围1-9内的数字
                for digit in digits:
                    num = int(digit)
                    if 1 <= num <= 9:
                        return str(num)
            
            # 如果没有找到直接的数字，尝试查找类别名称
            category_names = {
                '基础知识': '1', '个人信息': '2', '分析': '3', '归纳': '4', 
                '推断与迁移': '5', '回应与拓展': '6', '认同': '7', '质疑': '8', '指导': '9'
            }
            
            for name, num in category_names.items():
                if name in cleaned_text:
                    return num
            
            # 如果无法确定分类，返回默认值
            print("无法从输出文本中提取分类标签，返回-1")
            return "-1"
        elif task_type == "question_generation" or task_type == "teaching_design":
            # 对于题目生成和教学设计，我们保留完整的输出，只进行基本清理
            return self.cleanup_generation_output(output_text)
        else:
            # short_answer或其他任务类型，直接返回清理后的文本
            return output_text.strip()

    def cleanup_generation_output(self, text):
        """
        清理生成任务的输出，移除可能的提示词和系统消息
        """
        text = text.strip()
        # 移除常见的前缀
        prefixes_to_remove = [
            "请设计符合要求的题目：", "请提供完整的教学设计方案：", 
            "下面是", "### ", "好的，", "以下是", "我将"
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

    def generate_output(self, tokenizer, model, batch_size=1, max_items=None, offset=0):
        """
        统一的输出生成方法，适用于所有模型
        
        Args:
            tokenizer: 模型的分词器
            model: 已加载的模型
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
        
        if self.is_vllm:
            # Handle vLLM generation
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=2048
            )
            
            print("Generating outputs with vLLM...")
            outputs_raw = model.generate(prompts, sampling_params)
            for output in outputs_raw:
                generated_text = output.outputs[0].text
                outputs.append(generated_text)
        else:
            # Handle regular generation
            for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
                batch_prompts = prompts[i:i + batch_size]
                
                for prompt in batch_prompts:
                    # Ensure prompt fits in context window
                    truncated_prompt = self.truncate_prompt(prompt, tokenizer)
                    
                    # Convert prompt to chat format if model supports it
                    messages = [{"role": "user", "content": truncated_prompt}]
                    
                    # Check if the tokenizer has chat_template attribute
                    if hasattr(tokenizer, 'apply_chat_template') and callable(getattr(tokenizer, 'apply_chat_template')):
                        try:
                            # Use chat template if available
                            input_ids = tokenizer.apply_chat_template(
                                messages, 
                                return_tensors="pt", 
                                add_generation_prompt=True
                            ).to(f"cuda:{self.device}")
                        except Exception as e:
                            print(f"Error applying chat template: {e}")
                            print("Falling back to regular tokenization")
                            input_ids = tokenizer(truncated_prompt, return_tensors="pt").input_ids.to(f"cuda:{self.device}")
                    else:
                        # Fall back to standard tokenization
                        input_ids = tokenizer(truncated_prompt, return_tensors="pt").input_ids.to(f"cuda:{self.device}")
                    
                    # Set generation parameters
                    gen_kwargs = {
                        "max_new_tokens": 2048,
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "do_sample": True
                    }
                    
                    # Handle potential custom stopping criteria
                    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
                        
                    with torch.no_grad():
                        try:
                            output = model.generate(input_ids, **gen_kwargs)
                            # Decode output tokens
                            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
                            
                            # Check if we need to extract specific part of the output
                            # Some models include the prompt in the output
                            if truncated_prompt in output_text:
                                output_text = output_text.split(truncated_prompt)[-1]
                            
                            # Apply post-processing based on task type
                            processed_output = self.post_process_output(output_text, self.task_type)
                            outputs.append(processed_output)
                            
                        except Exception as e:
                            print(f"Error generating output: {e}")
                            outputs.append("ERROR: Generation failed")
                
                # Optional: Add delay between batches
                # time.sleep(0.5)
        
        # Save results
        self.save_results(questions, outputs, answers, subjects)
        return outputs
    
    def infer_category_from_filename(self, filename):
        """从文件名推断文件所属的认知层次"""
        # 正确的六个层次分类
        category_patterns = {
            "记忆": ["Memory", "记忆", "memory"],
            "理解": ["Understanding", "理解", "understanding", "yuedu", "shige"],
            "应用": ["Application", "应用", "application", "zuowen"],
            "分析": ["Analysis", "分析", "analysis"],
            "评价": ["Evaluation", "评价", "evaluation"],
            "创造": ["Creation", "创造", "creation", "写作", "writing"]
        }
        
        # 检查文件名是否匹配任何一个类别的模式
        for category, patterns in category_patterns.items():
            for pattern in patterns:
                if pattern.lower() in filename.lower():
                    return category
        
        # 如果没有匹配到，则根据其他规则推断
        if "essay_grading" in filename or "zuowen" in filename:
            return "应用"
        elif "essay" in filename or "writing" in filename:
            return "创造"
        elif "logiqa" in filename:
            return "分析"
        else:
            return "未知"  # 默认分类
        
    def save_results(self, questions, outputs, answers, subjects=None):
        """
        Save results to output file with standardized format

        Args:
            questions: List of questions
            outputs: List of model outputs
            answers: List of reference answers
            subjects: List of subjects (optional)
        """
        if not self.output_file:
            # 自动确定输出文件路径
            # 从任务文件路径中提取文件名
            task_filename = os.path.basename(self.task_path)
            
            # 确定类别名称
            category = self.infer_category_from_filename(self.task_path)
            
            # 处理模型名称，使其适合作为文件夹名称
            safe_model_name = self.model_name.replace('/', '_').replace(':', '_')
            
            # 根据是否为few-shot模式确定基础输出目录
            if self.is_few_shot:
                base_output_dir = "../outputs_few_shot"  # few-shot专用输出目录
            else:
                base_output_dir = "../outputs"  # 默认输出目录
            
            # 确保输出目录存在
            category_output_dir = os.path.join(base_output_dir, safe_model_name, category)
            os.makedirs(category_output_dir, exist_ok=True)
            
            # 完整的输出文件路径
            output_path = os.path.join(category_output_dir, task_filename)
            
            print(f"Auto-determined output path: {output_path}")
        else:
            output_path = self.output_file
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        
        # 准备保存结果
        results = []
        for i in range(len(questions)):
            if i >= len(outputs):
                break
                
            # 使用统一格式，无论是否为few-shot模式
            result = {
                "question": questions[i],
                "model_answer": outputs[i],
                "reference_answer": answers[i] if i < len(answers) else "",
                "category": self.infer_category_from_filename(self.task_path),
                "task_type": self.task_type
            }
            
            # 添加subjects信息，如果存在
            if subjects and i < len(subjects):
                result["subject"] = subjects[i]
                
            results.append(result)
        
        # 保存结果
        try:
            with jsonlines.open(output_path, 'w') as f:
                for result in results:
                    f.write(result)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
            # 尝试使用备用路径
            try:
                backup_path = f"backup_results_{self.model_name.replace('/', '_').replace(':', '_')}.jsonl"
                with jsonlines.open(backup_path, 'w') as f:
                    for result in results:
                        f.write(result)
                print(f"Results saved to backup file: {backup_path}")
            except Exception as e2:
                print(f"Error saving to backup file: {e2}") 