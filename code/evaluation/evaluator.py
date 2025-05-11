import re
import os
import json
import string
import numpy as np
import pandas as pd
import jsonlines
from pathlib import Path
import jieba
from rouge import Rouge
import torch
import time
import math

# 新增：openai包用于API调用
try:
    import openai
except ImportError:
    print("未能导入openai包，请先安装: pip install openai")

class Evaluator:
    """
    Evaluator class for assessing model outputs across different educational tasks
    """
    def __init__(self, result_file, task_type=None, bert_model_path=None):
        """
        Initialize evaluator
        Args:
            result_file: Path to the jsonl file with model outputs
            task_type: Type of task (optional, will be auto-detected if not provided)
            bert_model_path: Path to BERT model for similarity calculation
        """
        self.result_file = result_file
        self.bert_model_path = "/home/amax/mgq/model/bert-base-chinese"  # 默认设置BERT模型路径
        self._bert_model = None  # Lazy loading BERT model
        
        # 配置API（默认为星火API，可在评测时覆盖）
        self.openai_api_key = "lkHUhmpkDBixtQklGcKp:LBQYCLPIgfOkuXuzvSNu"
        self.openai_base_url = "https://spark-api-open.xf-yun.com/v2"
        self.openai_model = "x1"  # 默认使用星火x1模型
        self.openai_client = None  # 可以从外部传入OpenAI客户端
        
        # Load results and map fields (model_output/model_answer -> output, reference_answer -> answer)
        with jsonlines.open(result_file) as f:
            results = []
            for item in f:
                # 映射字段名
                result_item = {}
                if "model_output" in item:
                    result_item["output"] = item["model_output"]
                elif "model_answer" in item:  # 添加对model_answer字段的支持
                    result_item["output"] = item["model_answer"]
                elif "output" in item:
                    result_item["output"] = item["output"]
                else:
                    # 如果既没有model_output/model_answer也没有output字段，尝试其他可能的字段
                    print(f"警告: 在{result_file}中找不到model_output、model_answer或output字段")
                    # 跳过此项
                    continue
                
                # 处理回答字段
                if "reference_answer" in item:
                    # 如果reference_answer是数组，取第一个元素
                    if isinstance(item["reference_answer"], list):
                        result_item["answer"] = item["reference_answer"][0]
                    else:
                        result_item["answer"] = item["reference_answer"]
                elif "answer" in item:
                    result_item["answer"] = item["answer"]
                else:
                    print(f"警告: 在{result_file}中找不到answer或reference_answer字段")
                    # 跳过此项
                    continue
                    
                # 复制其他字段
                if "question" in item:
                    result_item["input"] = item["question"]
                    result_item["question"] = item["question"]
                elif "input" in item:
                    result_item["input"] = item["input"]
                else:
                    # 如果没有问题字段，使用空字符串
                    result_item["input"] = ""
                    print(f"警告: 在{result_file}中找不到question或input字段")
                
                # 添加到结果列表
                results.append(result_item)
                
            self.results = results
        
        print(f"读取{result_file}：找到{len(self.results)}条有效结果")
        
        # Determine task type if not provided
        if task_type:
            self.task_type = task_type
        else:
            self.task_type = self._determine_task_type()
        
        # 保存文件路径属性，用于获取文件所在目录和类别
        self.file_path = result_file
        
        # 确定评估的认知层次
        self.category = self._determine_category()
        
        # Initialize evaluation metrics
        self.metrics = self._initialize_metrics()
        
        # Punctuation list for normalization
        self.punctuation = list(string.punctuation) + list("""！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏.""")
    
    def _determine_category(self):
        """确定文件所属的认知层次"""
        file_path = self.file_path.lower()
        
        if "理解" in file_path:
            return "理解"
        elif "记忆" in file_path:
            return "记忆"
        elif "推理" in file_path:
            return "推理"
        elif "应用" in file_path:
            return "应用"
        elif "创造" in file_path:
            return "创造"
        elif "伦理" in file_path:
            return "伦理"
        else:
            # 尝试从文件名推断
            file_name = os.path.basename(file_path)
            if "primary" in file_name or "elementary" in file_name:
                return "记忆"
            elif "junior" in file_name:
                return "理解"
            elif "senior" in file_name or "logiqa" in file_name:
                return "推理"
            elif "writing" in file_name or "essay" in file_name:
                return "创造"
            return "未知"
    
    def _determine_task_type(self):
        """Determine the task type from file name and content"""
        file_name = os.path.basename(self.result_file)
        file_path = self.result_file
        
        # Check sample result to help determine task type
        sample_output = self.results[0]["output"] if self.results else ""
        sample_input = self.results[0]["input"] if self.results else ""
        
        # 新增：判断5_qg_100.jsonl、5_teachingdesign_50.jsonl和5_writing_50.jsonl
        if "5_qg_100" in file_name:
            return "qg_llm_scoring"
        elif "5_teachingdesign_50" in file_name:
            return "teachingdesign_llm_scoring"
        elif "5_writing_50" in file_name:
            return "writing_llm_scoring"
        
        # 针对记忆目录、理解目录、推理目录、伦理目录和伦理2目录下的选择题
        if any(x in file_path for x in ["记忆", "理解", "推理", "伦理"]):
            # 理解目录下的特定文件
            if "理解" in file_path and ("2_shige_100" in file_name or "2_yuedu_100" in file_name):
                return "reading_poetry_similarity"
            # 其他默认为选择题
            if len(sample_output) < 10 or re.search(r'[A-E]+', sample_output):
                return "multiple_choice"
        
        # 应用目录下的特定任务
        if "应用" in file_path:
            if "3_conversation_classification" in file_name:
                return "conversation_classification"
            elif "3_zuowen_100" in file_name:
                return "essay_grading"
        
        # 原有判断逻辑
        if "primary" in file_name or "junior" in file_name or "senior" in file_name or "logiqa" in file_name:
            if len(sample_output) < 10 and re.search(r'[A-E]+', sample_output):
                return "multiple_choice"
            elif "writing" in file_name:
                return "essay"
        elif "writing" in file_name:
            return "essay"
        elif "shige" in file_name or "yuedu" in file_name:
            return "short_answer"
        elif "zuowen" in file_name:
            return "essay_grading"
        
        # Default to multiple choice if can't determine
        return "multiple_choice"
    
    def _initialize_metrics(self):
        """根据任务类型和认知层次初始化评估指标"""
        # 根据用户需求简化评估指标
        
        # 记忆、推理、伦理层次的选择题只计算准确率
        if self.category in ["记忆", "推理", "伦理"] or self.task_type == "multiple_choice":
            return ["accuracy"]
        
        # 理解层次的诗歌和阅读理解计算ROUGE-L和BERT相似度
        elif self.category == "理解" and self.task_type in ["reading_poetry_similarity", "short_answer"]:
            metrics = ["rouge-l"]
            if self.bert_model_path:
                metrics.append("bert_similarity")
            return metrics
        
        # 应用层次的对话分类计算分类准确率
        elif self.category == "应用" and self.task_type == "conversation_classification":
            return ["classification_accuracy"]
        
        # 应用层次的作文评分计算分数差值和RMSE
        elif self.category == "应用" and self.task_type == "essay_grading":
            return ["score_difference", "rmse"]  # 新增RMSE指标
        
        # 创造层次使用API评分
        elif self.category == "创造" or self.task_type in ["qg_llm_scoring", "teachingdesign_llm_scoring", "writing_llm_scoring"]:
            return ["llm_score"]
        
        # 默认评估指标
        else:
            return ["accuracy"]
    
    def _load_bert_model(self):
        """Lazy load BERT model for similarity computation"""
        if self.bert_model_path and self._bert_model is None:
            try:
                # 条件导入避免强制依赖
                from sentence_transformers import SentenceTransformer
                print(f"正在加载BERT模型: {self.bert_model_path}")
                
                # 检查模型路径是否存在
                if not os.path.exists(self.bert_model_path):
                    print(f"错误: BERT模型路径不存在: {self.bert_model_path}")
                    return None
                
                # 尝试加载模型
                try:
                    self._bert_model = SentenceTransformer(self.bert_model_path)
                    print(f"BERT模型加载成功！模型类型: {type(self._bert_model)}")
                    
                    # 测试模型是否可用
                    test_embedding = self._bert_model.encode(["测试句子"])
                    print(f"模型测试成功，生成的嵌入维度: {test_embedding.shape}")
                    
                    return self._bert_model
                except Exception as model_error:
                    print(f"加载BERT模型时出现具体错误: {model_error}")
                    print("尝试使用默认模型...")
                    try:
                        # 尝试使用默认模型
                        self._bert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                        print("成功加载默认多语言模型")
                        return self._bert_model
                    except Exception as default_error:
                        print(f"加载默认模型也失败: {default_error}")
                        return None
            except ImportError:
                print("无法导入sentence_transformers，请确保已安装此库: pip install sentence-transformers")
                return None
            except Exception as e:
                print(f"加载BERT模型时出现未知错误: {e}")
                return None
        
        return self._bert_model
    
    def compute_bert_similarity(self, text1, text2):
        """Compute semantic similarity between two texts using BERT"""
        # 确保输入是字符串
        if not isinstance(text1, str) or not isinstance(text2, str):
            print(f"警告：输入不是字符串类型，text1类型: {type(text1)}, text2类型: {type(text2)}")
            text1 = str(text1) if not isinstance(text1, str) else text1
            text2 = str(text2) if not isinstance(text2, str) else text2
        
        # 检查文本是否为空
        if not text1.strip() or not text2.strip():
            print("警告：输入文本为空，无法计算相似度")
            return 0.0
        
        # 加载模型
        model = self._load_bert_model()
        if model is None:
            print("BERT模型未加载，无法计算相似度")
            return 0.0
        
        try:
            # 对输入文本做一些预处理，保留关键内容
            text1 = text1[:1500]  # 限制文本长度
            text2 = text2[:1500]  # 限制文本长度
            
            # Encode texts to get embeddings
            print(f"计算文本相似度，text1长度: {len(text1)}, text2长度: {len(text2)}")
            print(f"文本1前50字符: {text1[:50]}...")
            print(f"文本2前50字符: {text2[:50]}...")
            
            # 分别编码两个文本，以便于捕获具体哪个文本出现问题
            try:
                embedding1 = model.encode(text1)
                print(f"文本1编码成功，维度: {embedding1.shape}")
            except Exception as e1:
                print(f"文本1编码失败: {e1}")
                return 0.0
                
            try:
                embedding2 = model.encode(text2)
                print(f"文本2编码成功，维度: {embedding2.shape}")
            except Exception as e2:
                print(f"文本2编码失败: {e2}")
                return 0.0
            
            # 计算余弦相似度
            try:
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                
                if norm1 == 0 or norm2 == 0:
                    print("警告：嵌入向量范数为0，无法计算相似度")
                    return 0.0
                    
                similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
                print(f"计算得到的相似度为: {similarity:.4f}")
                return float(similarity)
            except Exception as e3:
                print(f"计算相似度时出错: {e3}")
                return 0.0
        except Exception as e:
            print(f"计算BERT相似度时出现未捕获的错误: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def normalize_text(self, text):
        """Remove punctuation and normalize text for comparison"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase and remove punctuation
        text = text.lower()
        for p in self.punctuation:
            text = text.replace(p, "")
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
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
    
    def evaluate_multiple_choice(self):
        """Evaluate multiple choice questions"""
        correct = 0
        total = 0
        
        for result in self.results:
            output = result["output"]
            answer = result["answer"]
            
            # Skip if no output or answer
            if not output or not answer:
                continue
                
            # Extract choice answer
            choice = self.extract_choice_answer(output)
            # Normalize answer to uppercase single letter
            answer = self.extract_choice_answer(answer)
            
            if choice and answer and choice.upper() == answer.upper():
                correct += 1
        
            total += 1
        
        return {"accuracy": correct / total if total > 0 else 0}
    
    def _extract_key_points(self, text):
        """从文本中提取关键点，用于简化比较"""
        if not isinstance(text, str):
            return ""
            
        # 去除HTML标签
        text = re.sub(r'<.*?>', '', text)
        
        # 去除多余空白和换行
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 去除常见的Markdown格式符号和特殊字符
        text = re.sub(r'[#*\\_`()\[\]{}]+', '', text)
        text = re.sub(r'[:：，,。\.；;！!？?…—]+', ' ', text)
        
        # 移除常见答题前缀
        prefixes_to_remove = ["作答", "回答", "解答", "我的答案", "我的回答", "解析", "分析"]
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # 提取关键句子（如答案、结论等）
        key_phrases = ["答案是", "答案：", "答案:", "答案为", "关键点是", "主要内容是", "结论是", 
                       "：", ":", "是"]
        for phrase in key_phrases:
            if phrase in text:
                parts = text.split(phrase, 1)
                if len(parts) > 1 and len(parts[1].strip()) > 0:
                    text = parts[1].strip()
                    break
        
        # 如果文本仍然很长（>200字符），尝试提取更核心的部分
        if len(text) > 200:
            # 寻找最有可能包含答案的句子
            sentences = re.split(r'[。.;；!！?？]', text)
            # 保留较短且有意义的句子
            filtered_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                # 过滤掉太短或明显是过渡句的句子
                if len(sentence) > 5 and not any(word in sentence for word in ["首先", "其次", "然后", "最后", "总之", "因此"]):
                    filtered_sentences.append(sentence)
            
            if filtered_sentences:
                # 按句子长度排序，优先选择中等长度的句子（既不太长也不太短）
                sorted_sentences = sorted(filtered_sentences, key=lambda s: abs(len(s) - 50))
                text = sorted_sentences[0]
        
        # 进一步清理和规范化文本
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'["""]', '', text)  # 移除引号
        
        return text
    
    def _extract_keywords(self, text):
        """提取文本中的关键词"""
        if not isinstance(text, str):
            return set()
            
        # 去除HTML标签和特殊字符
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[#*\\_`()\[\]{}]+', '', text)
        text = re.sub(r'[:：，,。\.；;！!？?…—]+', ' ', text)
        
        # 分词并去除停用词
        stopwords = {"的", "是", "在", "了", "和", "与", "或", "而", "及", "等", "从", "到", "中", "对", "为", "以", "及"}
        words = jieba.lcut(text)
        keywords = [word for word in words if len(word) > 1 and word not in stopwords]
        
        return set(keywords)
    
    def compute_keyword_similarity(self, text1, text2):
        """计算两段文本的关键词相似度"""
        keywords1 = self._extract_keywords(text1)
        keywords2 = self._extract_keywords(text2)
        
        if not keywords1 or not keywords2:
            return 0.0
            
        # 计算关键词相似度
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_short_answer(self):
        """Evaluate short answer questions (used for reading and poetry)"""
        rouge = Rouge()
        rouge_scores = []
        bert_scores = []
        keyword_scores = []
        
        print(f"开始评估短答案问题，共有{len(self.results)}个样本")
        error_count = 0
        
        for i, result in enumerate(self.results):
            output = result.get("output", "")
            answer = result.get("answer", "")
            
            # Skip if no output or answer
            if not output or not answer:
                print(f"跳过样本 #{i+1}：输出或答案为空")
                continue
            
            # 确保输出和答案是字符串类型
            if not isinstance(output, str):
                print(f"警告：样本 #{i+1} 的输出不是字符串类型，而是 {type(output)}，尝试转换...")
                try:
                    output = str(output)
                except Exception as e:
                    print(f"转换输出为字符串时出错: {e}")
                    error_count += 1
                    continue
            
            if not isinstance(answer, str):
                print(f"警告：样本 #{i+1} 的答案不是字符串类型，而是 {type(answer)}，尝试转换...")
                try:
                    answer = str(answer)
                except Exception as e:
                    print(f"转换答案为字符串时出错: {e}")
                    error_count += 1
                    continue
                    
            # 处理列表类型的答案（参考答案可能是列表形式）
            if isinstance(answer, list) and len(answer) > 0:
                print(f"样本 #{i+1} 的答案是列表类型，取第一个元素")
                answer = str(answer[0])
            
            # 预处理答案和输出
            # 对于模型输出，提取关键部分，简化比较
            processed_output = self._extract_key_points(output)
            processed_answer = self._extract_key_points(answer)
            
            # 打印处理后的文本摘要以便调试
            print(f"样本 #{i+1} - 处理后输出: {processed_output[:50]}..., 处理后答案: {processed_answer[:50]}...")
            
            try:
                # 计算ROUGE分数
                # 如果处理后文本差异大，尝试不同的处理策略
                rouge_score = 0.0
                try:
                    # 首先尝试直接计算ROUGE分数
                    rouge_score = rouge.get_scores(processed_output, processed_answer)[0]["rouge-l"]["f"]
                except Exception as e:
                    print(f"计算常规ROUGE分数时出错 (样本 #{i+1}): {e}")
                    # 如果失败，尝试更简单的评分方式，如基于关键词的重叠率
                    output_words = set(re.findall(r'\w+', processed_output.lower()))
                    answer_words = set(re.findall(r'\w+', processed_answer.lower()))
                    
                    if output_words and answer_words:
                        # 计算词汇重叠比例
                        overlap = len(output_words.intersection(answer_words))
                        total = len(output_words.union(answer_words))
                        rouge_score = overlap / total if total > 0 else 0
                        print(f"使用词汇重叠计算替代ROUGE分数: {rouge_score:.4f}")
                
                rouge_scores.append(rouge_score)
                print(f"样本 #{i+1} - ROUGE-L分数: {rouge_score:.4f}")
                
                # 计算关键词相似度
                keyword_score = self.compute_keyword_similarity(output, answer)
                keyword_scores.append(keyword_score)
                print(f"样本 #{i+1} - 关键词相似度: {keyword_score:.4f}")
                
                # 计算BERT相似度（如果可用）
                if self.bert_model_path:
                    # 使用原始输出和答案计算BERT相似度，因为BERT能更好地理解语义
                    bert_score = self.compute_bert_similarity(processed_output, processed_answer)
                    bert_scores.append(bert_score)
                    print(f"样本 #{i+1} - BERT相似度: {bert_score:.4f}")
            except Exception as e:
                print(f"计算评分时出错 (样本 #{i+1}): {e}")
                error_count += 1
        
        print(f"评估完成: {len(rouge_scores)}/{len(self.results)} 个样本计算了ROUGE分数, {len(keyword_scores)} 个样本计算了关键词相似度, {len(bert_scores)} 个样本计算了BERT分数, {error_count} 个样本出错")
        
        # 准备结果
        results = {}
        if rouge_scores:
            avg_rouge = sum(rouge_scores) / len(rouge_scores)
            results["rouge-l"] = avg_rouge
            print(f"平均ROUGE-L分数: {avg_rouge:.4f}")
        
        if keyword_scores:
            avg_keyword = sum(keyword_scores) / len(keyword_scores)
            results["keyword_similarity"] = avg_keyword
            print(f"平均关键词相似度: {avg_keyword:.4f}")
        
        if bert_scores:
            avg_bert = sum(bert_scores) / len(bert_scores)
            results["bert_similarity"] = avg_bert
            print(f"平均BERT相似度: {avg_bert:.4f}")
            
            # 为短答案评估同时使用BERT分数、关键词相似度和ROUGE分数的组合评分
            if rouge_scores and "rouge-l" in results and keyword_scores and "keyword_similarity" in results:
                # 对于诗歌阅读理解任务，组合多种评估指标
                combined_score = 0.6 * avg_bert + 0.3 * avg_keyword + 0.1 * avg_rouge
                results["combined_score"] = combined_score
                print(f"组合评分 (0.6*BERT + 0.3*关键词 + 0.1*ROUGE): {combined_score:.4f}")
        
        return results
    
    def evaluate_reading_poetry_similarity(self):
        """专门评估阅读和诗歌理解任务"""
        # 使用相同的评估方法
        return self.evaluate_short_answer()
    
    def evaluate_conversation_classification(self):
        """Evaluate conversation classification task"""
        correct = 0
        total = 0
        
        for result in self.results:
            output = result["output"]
            answer = result["answer"]
            
            # Skip if no output or answer
            if not output or not answer:
                continue
            
            # 确保输出和答案是字符串类型
            if not isinstance(output, str):
                try:
                    output = str(output)
                except:
                    continue
                    
            if not isinstance(answer, str):
                try:
                    answer = str(answer)
                except:
                    continue
                
            # Normalize and compare
            output_clean = output.strip()
            answer_clean = answer.strip()
            
            # Extract numeric classification
            output_nums = re.findall(r'\d+', output_clean)
            answer_nums = re.findall(r'\d+', answer_clean)
            
            output_num = int(output_nums[0]) if output_nums else -1
            answer_num = int(answer_nums[0]) if answer_nums else -1
            
            if output_num == answer_num:
                correct += 1
        
            total += 1
        
        return {"classification_accuracy": correct / total if total > 0 else 0}
    
    def evaluate_essay(self):
        """Evaluate essay generation task using LLM-based scoring"""
        return self.evaluate_llm_scoring()
    
    def evaluate_essay_grading(self):
        """Evaluate essay grading task (score prediction)"""
        differences = []
        squared_differences = []  # 新增：用于计算RMSE的平方差
        error_count = 0  # 统计错误次数
        
        print(f"开始评估作文评分任务，共有{len(self.results)}个样本")
        
        for i, result in enumerate(self.results):
            output = result.get("output", "")
            answer = result.get("answer", "")
            
            # Skip if no output or answer
            if not output or not answer:
                print(f"跳过样本 #{i+1}：输出或答案为空")
                continue
            
            # 确保输出和答案是字符串类型
            if not isinstance(output, str):
                print(f"警告：样本 #{i+1} 的输出不是字符串类型，而是 {type(output)}，尝试转换...")
                try:
                    output = str(output)
                except Exception as e:
                    print(f"转换输出为字符串时出错: {e}")
                    error_count += 1
                    continue
            
            if not isinstance(answer, str):
                print(f"警告：样本 #{i+1} 的答案不是字符串类型，而是 {type(answer)}，尝试转换...")
                try:
                    answer = str(answer)
                except Exception as e:
                    print(f"转换答案为字符串时出错: {e}")
                    error_count += 1
                    continue
            
            # Extract scores
            try:
                # Extract numeric scores
                output_nums = re.findall(r'\d+', output)
                answer_nums = re.findall(r'\d+', answer)
                
                if not output_nums:
                    print(f"警告：样本 #{i+1} 的输出中未找到数字分数: {output[:100]}...")
                    error_count += 1
                    continue
                
                if not answer_nums:
                    print(f"警告：样本 #{i+1} 的答案中未找到数字分数: {answer[:100]}...")
                    error_count += 1
                    continue
                
                output_score = int(output_nums[0])
                answer_score = int(answer_nums[0])
                
                # 记录分数信息便于调试
                print(f"样本 #{i+1} - 模型分数: {output_score}, 参考分数: {answer_score}")
                
                # Calculate absolute difference
                diff = abs(output_score - answer_score)
                differences.append(diff)
                
                # 新增：计算平方差
                squared_diff = (output_score - answer_score) ** 2
                squared_differences.append(squared_diff)
            except Exception as e:
                print(f"提取分数时出错 (样本 #{i+1}): {e}, 输出: {output[:50]}, 答案: {answer[:50]}")
                error_count += 1
        
        print(f"评估完成: {len(differences)}/{len(self.results)} 个样本有效, {error_count} 个样本出错")
        
        # Calculate mean absolute score difference
        avg_diff = sum(differences) / len(differences) if differences else float('inf')
        
        # 新增：计算RMSE（均方根误差）
        rmse = math.sqrt(sum(squared_differences) / len(squared_differences)) if squared_differences else float('inf')
        
        # 打印评估结果
        print(f"平均分数差: {avg_diff:.4f}, RMSE: {rmse:.4f}")
        
        return {
            "score_difference": avg_diff,
            "rmse": rmse  # 新增：返回RMSE指标
        }
    
    def evaluate_llm_scoring(self):
        """使用LLM评分对创造类任务进行评估"""
        # 导入必要的模块
        import time
        import re
        import openai
        from openai import OpenAI
        
        # 如果从外部传入了客户端，优先使用
        if hasattr(self, 'openai_client') and self.openai_client:
            openai_client = self.openai_client
            print(f"使用外部传入的OpenAI客户端，模型: {self.openai_model}")
        elif hasattr(self, 'openai_api_key') and hasattr(self, 'openai_base_url') and self.openai_api_key and self.openai_base_url:
            # 使用配置的API密钥和URL创建客户端
            openai_client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
            print(f"使用配置的API参数创建OpenAI客户端，Base URL: {self.openai_base_url}")
        else:
            print(f"警告：未配置OpenAI Client，无法进行LLM评分")
            return {"llm_score": 0.5}  # 默认分数
        
        scores = []
        
        # 根据任务类型确定评分提示和评分维度
        if self.task_type == "teachingdesign_llm_scoring":
            prompt_template = """请按照以下评分标准，对这份教学设计进行评分（总分100分）：

教学设计内容：
{content}

评分标准：
教学目标（20分）：
明确性（10分）：教学目标是否清晰、具体，能够明确阐述学生在课程结束后应掌握的知识、技能和态度。
适切性（10分）：教学目标是否符合课程标准和学生的年龄特点及学习水平。
教学内容（20分）：
准确性（10分）：教学内容是否准确无误，是否符合学科知识体系。
相关性（10分）：教学内容是否与教学目标紧密相关，是否有助于学生达成学习目标。
教学方法（20分）：
多样性（10分）：是否采用了多种教学方法（如讲授、讨论、实验、案例分析等），以满足不同学生的学习需求。
适切性（10分）：所选教学方法是否适合教学内容和学生的认知水平，能否有效促进学生的学习。
教学过程（20分）：
逻辑性（10分）：教学过程是否逻辑清晰，环节之间过渡自然，是否有助于学生逐步理解和掌握知识。
完整性（10分）：教学过程是否包括导入、新课讲授、实践练习、总结评价等完整环节，是否合理分配了各环节的时间。
教学资源（10分）：
丰富性（5分）：是否充分考虑了多种教学资源（如教材、教具、多媒体、实验器材等）的使用，以增强教学效果。
适切性（5分）：所选教学资源是否适合教学内容和学生的实际情况，能否有效支持教学活动。
教学评价（10分）：
多样性（5分）：是否采用了多种评价方式（如课堂提问、作业、测验、项目作业等），全面评估学生的学习效果。
有效性（5分）：评价方式是否能够有效测量学生对教学目标的达成度，是否注重过程性评价与终结性评价相结合。
创新性（10分）：
独特性（5分）：教学设计是否有独特之处，是否采用了新颖的教学策略或方法，能够激发学生的学习兴趣。
实用性（5分）：创新的教学策略是否具有实际可操作性，是否能够在实际教学中有效实施。
可行性（10分）：
时间管理（5分）：教学设计是否考虑了实际教学时间的限制，各环节时间分配是否合理。
资源可行性（5分）：所设计的教学活动是否能够在现有的教学资源和条件下顺利实施。

请根据以上标准进行评分，并给出总分（满分100分）。请只返回分数，不要有任何解释。
"""
        elif self.task_type == "qg_llm_scoring":
            prompt_template = """请按照以下评分标准，对这个题目生成内容进行评分（总分100分）：

题目内容：
{content}

评分标准：
一、题目准确性（20分）
知识准确性（10分）：题目所涉及的知识点是否准确无误，是否符合学科的基本概念、原理和规律。
表述清晰性（10分）：题目表述是否清晰、准确，是否存在歧义或模糊之处，是否能够使学生明确题意。
二、题目难易程度（20分）
难度适切性（10分）：题目难度是否符合相应年级和学科的要求，是否能够区分不同水平的学生。
层次分明性（10分）：题目是否具有层次性，是否能够涵盖不同难度层次的知识点，是否有利于学生逐步提高。
三、题目实用性（15分）
教学相关性（8分）：题目是否与教学内容紧密相关，是否能够有效支持教学目标的实现。
可操作性（7分）：题目是否具有可操作性，是否能够在实际教学中有效实施，是否考虑到了教学时间和资源的限制。
四、题目创新性（15分）
题目新颖性（8分）：题目是否具有新颖性，是否能够激发学生的学习兴趣和创造力，是否与传统题目有所不同。
形式多样性（7分）：题目是否采用了多样化的形式，是否能够综合考查学生的不同能力和素质。
五、题目灵活性（10分）
思维灵活性（6分）：题目是否能够考查学生的灵活思维能力，是否鼓励学生从不同角度思考问题。
解法多样性（4分）：题目是否具有多种解法，是否能够培养学生的发散思维和创新能力。
六、题目综合性（10分）
知识综合度（6分）：题目是否能够综合考查多个知识点，是否能够促进学生对知识的系统理解和综合运用。
能力覆盖面（4分）：题目是否能够考查学生的多种能力，如记忆、理解、应用、分析、综合和评价等。
七、题目思政融入情况（10分）
思政元素贴合度（6分）：题目是否自然地融入了思政元素，是否与学科知识有机结合，是否生硬或牵强。
育人功能（4分）：题目是否具有积极的育人功能，是否能够引导学生树立正确的价值观和人生观。

请根据以上标准进行评分，并给出总分（满分100分）。请只返回分数，不要有任何解释。
"""
        elif self.task_type == "writing_llm_scoring":
            prompt_template = """请按照以下评分标准，对这篇写作内容进行评分（总分100分）：

写作内容：
{content}

评分标准：
一、内容完整性（20分）
主题明确性（10分）：文本是否紧扣主题，中心思想是否鲜明且贯穿始终，是否存在偏题或离题的现象。
内容丰富度（10分）：文本是否围绕主题展开了充分的论述或描述，是否提供了足够的细节、例证或论据来支撑观点或丰富情节。
二、结构逻辑性（20分）
整体连贯性（10分）：文本段落之间是否过渡自然，是否形成了紧密的逻辑联系，使读者能够顺畅地理解作者的思路。
段落合理性（10分）：每个段落是否都有明确的主题句或中心思想，句子之间的组织是否逻辑清晰，是否符合常见的写作结构（如总分总、分总等）。
三、语言表达能力（20分）
语法正确性（10分）：文本是否存在语法错误，如主谓一致、时态、语态、句子结构等方面的问题。
用词准确性（10分）：用词是否恰当、准确，是否能够精确地表达作者的意图，是否存在滥用、误用或不当的词汇选择。
四、创意与想象力（15分）
新颖性（8分）：文本在主题、观点、情节或表现手法上是否具有新颖性，是否能够给人耳目一新的感觉。
想象力（7分）：在描述性或叙事性文本中，是否展现了丰富的想象力，是否创造出了独特的场景、人物或情节。
五、写作风格与适应性（10分）
风格一致性（6分）：文本是否保持了一致的写作风格，包括语气、语调和表达方式等，是否符合目标读者的期望和需求。
读者适应性（4分）：文本是否能够根据目标读者的年龄、背景知识和兴趣等特点，选择合适的表达方式和内容深度，以提高文本的可读性和吸引力。
六、深度与洞察力（15分）
思考深度（6分）：文本是否对主题进行了深入的思考和分析，是否展现出了作者对问题的深刻理解和独到见解。
批判性思维（9分）：在论述性或评论性文本中，是否体现了批判性思维，是否能够对不同的观点或现象进行客观的评价和分析。

请根据以上标准进行评分，并给出总分（满分100分）。请只返回分数，不要有任何解释。
"""
        else:
            # 对于其他类型的任务，使用通用提示
            prompt_template = """请评估以下内容的质量，给出1-100的分数：

内容：
{content}

请根据内容的准确性、完整性、逻辑性、创新性等方面进行综合评分，满分100分。
请只返回分数，不要有任何解释。
"""
        
        # 评估所有样本，不限制数量
        print(f"正在使用 {self.openai_model} 模型评估所有 {len(self.results)} 个样本...")
        
        for i, result in enumerate(self.results):
            prompt = result["input"]
            output = result.get("output") or result.get("model_output") or result.get("model_answer") or ""
            
            # 跳过无输出的情况
            if not output:
                continue
            
            try:
                # 构造评分内容
                if self.task_type in ["teachingdesign_llm_scoring", "qg_llm_scoring", "writing_llm_scoring"]:
                    # 使用详细的评分标准模板
                    prompt_content = prompt_template.format(content=output)
                    messages = [
                        {"role": "system", "content": "你是一位专业的教育评估专家，善于客观公正地评价内容质量。"},
                        {"role": "user", "content": prompt_content}
                    ]
                else:
                    # 使用简化的评分请求
                    messages = [
                        {"role": "system", "content": "你是一位专业的教育评估专家。请根据提供的题目和回答，给回答质量评分(1-10分)。评分应基于回答的准确性、完整性、逻辑性和创新性。"},
                        {"role": "user", "content": f"题目：{prompt}\n\n回答：{output}\n\n请给这个回答打分（1-10分），只需返回分数。"}
                    ]
                
                # 调用API获取评分
                try:
                    # 使用传入的客户端或创建的客户端
                    response = openai_client.chat.completions.create(
                        model=self.openai_model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=10
                    )
                    
                    # 提取分数
                    score_text = response.choices[0].message.content.strip()
                    score_nums = re.findall(r'\d+\.?\d*', score_text)
                    
                    if score_nums:
                        score = float(score_nums[0])
                        # 归一化到0-1区间
                        if self.task_type in ["teachingdesign_llm_scoring", "qg_llm_scoring", "writing_llm_scoring"]:
                            normalized_score = score / 100.0  # 这些任务使用100分制
                        else:
                            normalized_score = score / 10.0   # 其他任务使用10分制
                        scores.append(normalized_score)
                        print(f"LLM评分: 样本 #{i+1}/{len(self.results)}: {score}分 -> {normalized_score:.4f}")
                    else:
                        print(f"无法从API响应中提取分数: {score_text}")
                except Exception as api_error:
                    print(f"调用API时出错: {api_error}")
                
                # 避免API限流
                time.sleep(1.0)  # 增加等待时间，避免ChatAnywhere API限流
                
            except Exception as e:
                print(f"LLM评分时出错 (样本 #{i+1}): {e}")
        
        # 计算平均分
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"评分完成。共评估了 {len(scores)} 个有效样本，平均分: {avg_score:.4f}")
        
        return {"llm_score": avg_score}
    
    def evaluate(self):
        """Evaluate model outputs based on task type"""
        if self.task_type == "multiple_choice":
            return self.evaluate_multiple_choice()
        elif self.task_type == "short_answer":
            return self.evaluate_short_answer()
        elif self.task_type == "reading_poetry_similarity":
            return self.evaluate_reading_poetry_similarity()
        elif self.task_type == "essay":
            return self.evaluate_essay()
        elif self.task_type == "essay_grading":
            return self.evaluate_essay_grading()
        elif self.task_type == "conversation_classification":
            return self.evaluate_conversation_classification()
        elif self.task_type in ["qg_llm_scoring", "teachingdesign_llm_scoring", "writing_llm_scoring"]:
            return self.evaluate_llm_scoring()
        else:
            return {"error": "Unknown task type"}


class BenchmarkEvaluator:
    """
    Evaluate multiple result files and generate a combined report
    """
    def __init__(self, result_dir, bert_model_path=None, openai_api_key=None, openai_base_url=None):
        """
        Initialize benchmark evaluator
        Args:
            result_dir: Directory containing model output files
            bert_model_path: Path to BERT model for similarity calculation
            openai_api_key: API key for OpenAI-compatible services
            openai_base_url: Base URL for OpenAI-compatible services
        """
        self.result_dir = result_dir
        self.model_name = os.path.basename(result_dir)  # 从结果目录名提取模型名称
        self.result_files = self._get_result_files()
        
        # 设置BERT模型路径为固定值
        self.bert_model_path = "/home/amax/mgq/model/bert-base-chinese"
        
        # 设置星火API信息
        self.openai_api_key = openai_api_key or "lkHUhmpkDBixtQklGcKp:LBQYCLPIgfOkuXuzvSNu"
        self.openai_base_url = openai_base_url or "https://spark-api-open.xf-yun.com/v2"
        
        # 设置OpenAI API（如果提供了凭据）
        try:
            # 使用客户端方式配置OpenAI
            from openai import OpenAI
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
            openai.client = client
            print(f"已配置OpenAI API客户端，Base URL: {self.openai_base_url}")
        except Exception as e:
            print(f"配置OpenAI API时出错: {e}")
        
    def _get_result_files(self):
        """Get all result files in the directory (including subdirectories)"""
        # 递归搜索所有jsonl文件
        return list(Path(self.result_dir).glob("**/*.jsonl"))
    
    def evaluate_all(self):
        """Evaluate all result files and generate a report"""
        results = []
        task_specific_results = {}
        
        for file_path in self.result_files:
            try:
                # 使用结果目录名作为模型名
                model_name = self.model_name
                
                # 获取文件名
                filename = file_path.name
                
                # Create evaluator
                evaluator = Evaluator(str(file_path), bert_model_path=self.bert_model_path)
                
                # 配置API信息
                evaluator.openai_api_key = self.openai_api_key
                evaluator.openai_base_url = self.openai_base_url
                
                # 获取认知层次和任务类型
                category = evaluator.category
                task_type = evaluator.task_type
                
                # 执行评估
                scores = evaluator.evaluate()
                
                # 保存任务特定结果
                task_name = filename.replace(".jsonl", "")
                task_specific_results[task_name] = {
                    "category": category,
                    "task_type": task_type,
                    "scores": scores
                }
                
                # Add to results
                for metric, score in scores.items():
                    results.append({
                        "model": model_name,
                        "task": filename,
                        "category": category,
                        "task_type": task_type,
                        "metric": metric,
                        "score": score
                    })
                
                print(f"已评测: {category}/{filename} - {metric}={score:.4f}")
                
            except Exception as e:
                print(f"评测出错 {file_path}: {e}")
        
        # 如果没有结果，返回空数据
        if not results:
            print("未找到可评测的结果文件。请确认路径是否正确: " + self.result_dir)
            return {}
        
        # 创建DataFrame
        df = pd.DataFrame(results)
        
        # 保存详细结果
        output_path = os.path.join(self.result_dir, "evaluation_results.csv")
        df.to_csv(output_path, index=False)
        
        # 生成按认知层次和指标分组的摘要
        summary = df.pivot_table(
            index=['category', 'task_type'], 
            columns=['metric'], 
            values='score',
            aggfunc='mean'
        ).reset_index()
        
        # 保存摘要表格
        summary_path = os.path.join(self.result_dir, "evaluation_summary.csv")
        summary.to_csv(summary_path, index=False)
        
        # 创建更友好的表格格式
        task_summary = self._create_task_summary_table(task_specific_results)
        task_summary_path = os.path.join(self.result_dir, "task_summary.csv")
        task_summary.to_csv(task_summary_path, index=True)
        
        # 创建按认知层次汇总的结果
        category_summary = df.groupby('category')['score'].mean().reset_index()
        category_summary_path = os.path.join(self.result_dir, "category_summary.csv")
        category_summary.to_csv(category_summary_path, index=False)
        
        print(f"评测结果已保存至: {output_path}")
        print(f"评测摘要已保存至: {summary_path}")
        print(f"任务评测摘要已保存至: {task_summary_path}")
        print(f"认知层次汇总已保存至: {category_summary_path}")
        
        return task_specific_results
    
    def _create_task_summary_table(self, task_results):
        """创建更友好的任务评测摘要表格"""
        data = []
        
        # 按照认知层次分组整理数据
        for task_name, result in task_results.items():
            category = result["category"]
            task_type = result["task_type"]
            scores = result["scores"]
            
            # 获取主要得分
            main_score = None
            if "accuracy" in scores:
                main_score = scores["accuracy"]
            elif "rouge-l" in scores:
                main_score = scores["rouge-l"]
            elif "bert_similarity" in scores:
                main_score = scores["bert_similarity"]
            elif "score_difference" in scores:
                main_score = scores["score_difference"]
            elif "classification_accuracy" in scores:
                main_score = scores["classification_accuracy"]
            elif "llm_score" in scores:
                main_score = scores["llm_score"]
            
            if main_score is not None:
                row = {
                    "认知层次": category,
                    "任务类型": task_type,
                    "文件名": task_name,
                    "得分": f"{main_score:.4f}"
                }
                data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 透视表，按认知层次分组
        if not df.empty:
            pivot_df = df.pivot_table(
                index="文件名", 
                values="得分",
                aggfunc=lambda x: x
            )
            return pivot_df
        else:
            return pd.DataFrame() 