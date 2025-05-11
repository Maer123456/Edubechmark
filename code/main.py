import os
import sys
import argparse
import logging
import json
import jsonlines
import time
from pathlib import Path

# Add the directory containing generation modules to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "generation"))

# Import generator modules
from model_gen import ModelGenerator
from qwen_gen import QwenGenerator
from glm_gen import GLMGenerator
from deepseek_gen import DeepSeekGenerator
from baichuan_gen import BaichuanGenerator
from ollama_baichuan_gen import OllamaBaichuanGenerator
from api_gen import ApiGenerator

# Import evaluators
sys.path.append(os.path.join(os.path.dirname(__file__), "evaluation"))
from evaluator import Evaluator, BenchmarkEvaluator

# Check for environment variables
TASK_TYPE_HINT = os.environ.get("TASK_TYPE_HINT", "")  # 可以通过环境变量设置任务类型提示

def setup_logging(log_file="edueval.log"):
    """配置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("edueval")

def get_task_files(data_dir, category=None, task_type=None):
    """
    获取数据目录中的所有任务文件
    
    Args:
        data_dir: 基础数据目录
        category: 可选，按类别过滤(理解, 记忆, etc.)
        task_type: 可选，按任务类型过滤(primary, junior, etc.)
    
    Returns:
        任务文件路径列表
    """
    # 将data_dir转换为绝对路径(如果需要)
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(data_dir)
        
    data_path = Path(data_dir)
    
    # 获取所有类别或按指定类别过滤
    if category:
        categories = [data_path / category]
    else:
        categories = [p for p in data_path.iterdir() if p.is_dir()]
    
    # 收集所有任务文件
    task_files = []
    for cat_dir in categories:
        if task_type:
            # 按任务类型过滤 - 支持数字前缀模式
            files = list(cat_dir.glob(f"*{task_type}*.jsonl"))
            task_files.extend(files)
        else:
            # 获取所有JSONL文件
            files = list(cat_dir.glob("*.jsonl"))
            task_files.extend(files)
    
    # 将所有路径转换为字符串形式的绝对路径
    return [str(f.absolute()) for f in task_files]

def generate_model_outputs(args):
    """Generate model outputs for evaluation"""
    logger = logging.getLogger("edueval.generation")
    
    # 获取任务文件列表
    if args.task_files:
        # 使用用户指定的任务文件列表
        task_files = args.task_files
        logger.info(f"使用用户指定的 {len(task_files)} 个任务文件。")
    elif args.task_file:
        # 单个指定的任务文件
        if not os.path.isabs(args.task_file):
            args.task_file = os.path.abspath(args.task_file)
        
        # 检查文件是否存在
        if not os.path.exists(args.task_file):
            logger.error(f"Task file not found: {args.task_file}")
            logger.error(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"Task file not found: {args.task_file}")
        
        task_files = [args.task_file]
    else:
        # 否则获取所有任务文件
        task_files = get_task_files(args.data_dir, args.category, args.task_type)
    
    # 如果指定了默认任务文件但task_files为空
    if not task_files:
        # 默认任务文件列表
        data_base_dir = args.data_dir
        task_files = [
            f"{data_base_dir}/理解/2_yuedu_100.jsonl",
            f"{data_base_dir}/理解/2_shige_100.jsonl",
            f"{data_base_dir}/应用/3_zuowen_100.jsonl"
        ]
        # 检查默认文件是否存在
        task_files = [f for f in task_files if os.path.exists(f)]
    
    logger.info(f"处理 {len(task_files)} 个任务文件")
    for i, tf in enumerate(task_files):
        logger.info(f"任务文件 {i+1}: {tf} (存在: {os.path.exists(tf)})")
    
    # 确定模型访问类型和输出目录
    if args.model_access_type == "api":
        logger.info(f"使用API模式访问模型: {args.api_model}")
        model_name = args.api_model
        model_name_safe = model_name.replace(':', '_')
        if args.is_few_shot:
            output_dir = os.path.join(args.output_dir + "_few_shot", f"api_{model_name_safe}")
        else:
            output_dir = os.path.join(args.output_dir, f"api_{model_name_safe}")
    elif args.model_access_type == "ollama":
        logger.info(f"使用Ollama模式访问模型: {args.ollama_model}")
        model_name = args.ollama_model
        model_name_safe = model_name.replace(':', '_')
        if args.is_few_shot:
            output_dir = os.path.join(args.output_dir + "_few_shot", f"ollama_{model_name_safe}")
        else:
            output_dir = os.path.join(args.output_dir, f"ollama_{model_name_safe}")
    else:  # 默认为local模式
        logger.info(f"使用本地模式访问模型: {args.model_name}")
        model_name = args.model_name
        model_name_safe = model_name.replace('/', '_').replace(':', '_')
        if args.is_few_shot:
            output_dir = os.path.join(args.output_dir + "_few_shot", model_name_safe)
        else:
            output_dir = os.path.join(args.output_dir, model_name_safe)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果设置了任务类型提示，显示提示信息
    if TASK_TYPE_HINT:
        logger.info(f"已通过环境变量设置任务类型提示为: {TASK_TYPE_HINT}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 记录成功和失败的文件
    successful_files = []
    failed_files = []
    
    # 处理每个任务文件
    for task_path in task_files:
        # 准备输出文件路径
        if args.direct_output_file and len(task_files) == 1:
            output_file_path = args.direct_output_file
            # 确保输出文件的目录存在
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        else:
            # 原来的路径计算逻辑
            relative_path = os.path.relpath(task_path, args.data_dir)
            output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
            output_filename = os.path.basename(relative_path)
            output_file_path = os.path.join(output_subdir, output_filename)
            # 确保特定的输出子目录存在
            Path(output_subdir).mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'-'*75}")
        logger.info(f"正在处理任务文件: {task_path}")
        logger.info(f"输出将保存到: {output_file_path}")

        # 如果需要few-shot学习，尝试自动确定few-shot文件路径
        few_shot_path = args.few_shot_path
        if args.is_few_shot and not few_shot_path:
            # 获取任务类型和所属目录
            basename = os.path.basename(task_path)
            dirname = os.path.dirname(task_path)
            category = os.path.basename(dirname)
            
            # 自动确定任务类型
            task_type = "unknown"
            if "primary" in basename or "junior" in basename or "senior" in basename:
                task_type = "multiple_choice"
            elif "logiqa" in basename:
                task_type = "multiple_choice"
            elif "shige" in basename or "yuedu" in basename:
                task_type = "short_answer"
            elif "zuowen" in basename:
                task_type = "essay_grading"
            elif "writing" in basename:
                task_type = "essay"
            elif "qg" in basename:
                task_type = "question_generation"
            elif "teachingdesign" in basename:
                task_type = "teaching_design"
            elif "conversation_classification" in basename:
                task_type = "conversation_classification"
            elif "伦理" in basename:
                task_type = "multiple_choice"
                
            # 尝试找到对应的few-shot文件
            if task_type != "unknown":
                # 目标文件：对应的目录下的同名或同类型文件
                possible_few_shot_files = [
                    f"../few_shot/{category}/{basename}",
                    f"../few_shot/{category}/{task_type}_sample.jsonl"
                ]
                
                # 检查文件是否存在
                for fsf in possible_few_shot_files:
                    if os.path.exists(fsf):
                        few_shot_path = fsf
                        logger.info(f"自动找到few-shot文件: {few_shot_path}")
                        break
            
            # 如果找不到，使用默认示例文件
            if not few_shot_path:
                # 首先检查是否有同类别下的默认示例文件
                default_few_shot = f"../few_shot/{category}/default.jsonl"
                if os.path.exists(default_few_shot):
                    few_shot_path = default_few_shot
                    logger.info(f"使用默认few-shot文件: {few_shot_path}")
                else:
                    # 检查是否有创造目录下对应的示例文件（创造目录包含了几种常见的任务类型）
                    if task_type == "question_generation":
                        default_few_shot = f"../few_shot/创造/5_qg_100.jsonl"
                    elif task_type == "teaching_design":
                        default_few_shot = f"../few_shot/创造/5_teachingdesign_50.jsonl"
                    elif task_type == "essay":
                        default_few_shot = f"../few_shot/创造/5_writing_50.jsonl"
                    elif task_type == "multiple_choice":
                        default_few_shot = f"../few_shot/记忆/1_primary_400.jsonl"
                    elif task_type == "short_answer":
                        default_few_shot = f"../few_shot/理解/2_shige_100.jsonl"
                    elif task_type == "essay_grading":
                        default_few_shot = f"../few_shot/应用/3_zuowen_100.jsonl"
                    elif task_type == "conversation_classification":
                        default_few_shot = f"../few_shot/应用/3_conversation_classification.jsonl"
                    else:
                        default_few_shot = None
                    
                    if default_few_shot and os.path.exists(default_few_shot):
                        few_shot_path = default_few_shot
                        logger.info(f"使用默认few-shot文件: {few_shot_path}")
            
            # 如果仍找不到，则关闭few-shot模式
            if not few_shot_path:
                logger.warning(f"找不到适合任务 {basename} 的few-shot文件，将禁用few-shot模式")
                is_few_shot = False
            else:
                is_few_shot = True
        else:
            is_few_shot = args.is_few_shot

        # 初始化生成器
        generator = None
        model = None
        tokenizer = None
        
        try:
            # 根据模型访问类型选择相应的生成器
            if args.model_access_type == "api":
                # 使用API模式
                generator = ApiGenerator(
                    task_path=task_path,
                    model_name=args.api_model,
                    base_url=args.api_base_url,
                    api_key=args.api_key,
                    device=args.device,
                    is_few_shot=is_few_shot,
                    few_shot_path=few_shot_path,
                    output_file=output_file_path
                )
            elif args.model_access_type == "ollama":
                # 使用Ollama模式
                if "baichuan" in args.ollama_model.lower():
                    # 对于Baichuan模型，使用专用的OllamaBaichuanGenerator
                    generator = OllamaBaichuanGenerator(
                        task_path=task_path,
                        model_path=None,  # Ollama模式下不需要模型路径
                        model_name=args.ollama_model,
                        device=args.device,
                        is_few_shot=is_few_shot,
                        few_shot_path=few_shot_path,
                        ollama_base_url=args.ollama_base_url,
                        output_file=output_file_path
                    )
                else:
                    # 对于其他模型，使用通用的DeepSeekGenerator
                    generator = DeepSeekGenerator(
                        task_path=task_path,
                        model_path=None,  # Ollama模式下不需要模型路径
                        model_name=args.ollama_model,
                        device=args.device,
                        is_few_shot=is_few_shot,
                        few_shot_path=few_shot_path,
                        ollama_base_url=args.ollama_base_url,
                        output_file=output_file_path
                    )
            else:  # 默认使用本地模型
                # 根据模型名称选择合适的生成器
                if "qwen" in args.model_name.lower():
                    generator = QwenGenerator(
                        task_path=task_path,
                        model_path=args.model_path,
                        model_name=args.model_name,
                        device=args.device,
                        is_few_shot=is_few_shot,
                        few_shot_path=few_shot_path,
                        is_vllm=args.vllm,
                        tensor_parallel_size=args.tensor_parallel_size,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        output_file=output_file_path
                    )
                elif "glm" in args.model_name.lower():
                    generator = GLMGenerator(
                        task_path=task_path,
                        model_path=args.model_path,
                        model_name=args.model_name,
                        device=args.device,
                        is_few_shot=is_few_shot,
                        few_shot_path=few_shot_path,
                        is_vllm=args.vllm,
                        tensor_parallel_size=args.tensor_parallel_size,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        output_file=output_file_path
                    )
                elif "deepseek" in args.model_name.lower():
                    generator = DeepSeekGenerator(
                        task_path=task_path,
                        model_path=args.model_path,
                        model_name=args.model_name,
                        device=args.device,
                        is_few_shot=is_few_shot,
                        few_shot_path=few_shot_path,
                        is_vllm=args.vllm,
                        tensor_parallel_size=args.tensor_parallel_size,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        output_file=output_file_path
                    )
                elif "baichuan" in args.model_name.lower() or "educhat" in args.model_name.lower():
                    generator = BaichuanGenerator(
                        task_path=task_path,
                        model_path=args.model_path,
                        model_name=args.model_name,
                        device=args.device,
                        is_few_shot=is_few_shot,
                        few_shot_path=few_shot_path,
                        is_vllm=args.vllm,
                        tensor_parallel_size=args.tensor_parallel_size,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        output_file=output_file_path
                    )
                else:
                    # 对于其他模型类型，默认使用QwenGenerator
                    logger.warning(f"未明确识别的模型类型 '{args.model_name}'，使用QwenGenerator作为默认处理器")
                    generator = QwenGenerator(
                        task_path=task_path,
                        model_path=args.model_path,
                        model_name=args.model_name,
                        device=args.device,
                        is_few_shot=is_few_shot,
                        few_shot_path=few_shot_path,
                        is_vllm=args.vllm,
                        tensor_parallel_size=args.tensor_parallel_size,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        output_file=output_file_path
                    )
            
            # 初始化模型并生成输出
            try:
                model, tokenizer = generator.model_init()
                outputs = generator.generate_output(
                    tokenizer=tokenizer,
                    model=model,
                    batch_size=args.batch_size,
                    max_items=args.max_items,
                    offset=args.offset
                )
                
                # 记录成功处理的文件
                successful_files.append(task_path)
                logger.info(f"为{task_path}生成了{len(outputs) if outputs else '?'}个输出")
                
            except Exception as e:
                # 处理生成过程中的错误
                logger.error(f"在为{task_path}生成输出时发生错误: {str(e)}")
                failed_files.append((task_path, str(e)))
                
        except Exception as e:
            # 处理初始化过程中的错误
            logger.error(f"初始化生成器时发生错误: {str(e)}")
            failed_files.append((task_path, str(e)))
    
    # 记录结束时间和统计信息
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("\n" + "="*30 + " 生成摘要 " + "="*30)
    logger.info(f"总共处理了 {len(task_files)} 个任务文件")
    logger.info(f"成功: {len(successful_files)} 个文件")
    logger.info(f"失败: {len(failed_files)} 个文件")
    if failed_files:
        logger.info("失败的文件:")
        for file_path, error in failed_files:
            logger.info(f"  - {file_path}: {error}")
    logger.info(f"总耗时: {elapsed_time:.2f} 秒")
    logger.info(f"使用{'few-shot' if args.is_few_shot else '标准'}模式，输出保存到: {output_dir}")

def evaluate_model_outputs(args):
    """评估模型输出"""
    logger = logging.getLogger("edueval.evaluation")
    
    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        args.output_dir = os.path.join(args.result_dir, "evaluation")
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建评估器
    evaluator = BenchmarkEvaluator(
        result_dir=args.result_dir,
        bert_model_path=args.bert_model_path,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url
    )
    
    # 执行评估
    results = evaluator.evaluate_all()
    
    # 保存整体评估结果
    output_path = os.path.join(args.output_dir, "evaluation_results.json")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"评估结果已保存至: {output_path}")
    except Exception as e:
        logger.error(f"保存评估结果失败: {str(e)}")
    
    logger.info("评估完成")
    return results

def main():
    parser = argparse.ArgumentParser(description="Educational Language Model Evaluation Tool")
    subparsers = parser.add_subparsers(dest="command", help="选择命令")
    
    # 生成命令参数
    gen_parser = subparsers.add_parser("generate", help="使用模型生成答案")
    
    # 模型访问类型选择
    model_access_group = gen_parser.add_argument_group("Model Access Type (choose one)")
    gen_parser.add_argument("--model_access_type", choices=["local", "api", "ollama"], default="local",
                        help="模型访问类型: local(本地模型)，api(API访问)，ollama(Ollama服务)")
    
    # 本地模型参数
    local_group = gen_parser.add_argument_group("Local Model Parameters")
    local_group.add_argument("--model_path", help="模型路径")
    local_group.add_argument("--model_name", help="模型名称")
    local_group.add_argument("--vllm", action="store_true", help="是否使用vLLM加速")
    local_group.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor并行大小")
    local_group.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU内存利用率")
    
    # API模型参数
    api_group = gen_parser.add_argument_group("API Model Parameters")
    api_group.add_argument("--api_model", help="API模型名称")
    api_group.add_argument("--api_key", help="API密钥")
    api_group.add_argument("--api_base_url", help="API基础URL")
    
    # Ollama参数
    ollama_group = gen_parser.add_argument_group("Ollama Parameters")
    ollama_group.add_argument("--ollama_model", help="Ollama模型名称")
    ollama_group.add_argument("--ollama_base_url", help="Ollama基础URL")
    
    # 通用参数
    gen_parser.add_argument("--device", default="0", help="设备ID")
    gen_parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    gen_parser.add_argument("--max_items", type=int, help="最大处理项目数")
    gen_parser.add_argument("--offset", type=int, default=0, help="起始偏移量")
    gen_parser.add_argument("--data_dir", default="../data", help="数据目录")
    gen_parser.add_argument("--output_dir", default="../outputs", help="输出目录")
    gen_parser.add_argument("--category", help="类别过滤(理解,记忆,etc.)")
    gen_parser.add_argument("--task_type", help="任务类型过滤(primary,junior,etc.)")
    gen_parser.add_argument("--task_file", help="单个任务文件路径")
    gen_parser.add_argument("--task_files", nargs="+", help="多个任务文件路径")
    gen_parser.add_argument("--is_few_shot", action="store_true", help="是否使用few-shot方式")
    gen_parser.add_argument("--few_shot_path", help="few-shot示例文件路径")
    gen_parser.add_argument("--direct_output_file", help="直接指定输出文件路径")
    
    # 评估命令参数
    eval_parser = subparsers.add_parser("evaluate", help="评估模型输出")
    eval_parser.add_argument("--result_dir", required=True, help="模型输出结果目录")
    eval_parser.add_argument("--output_dir", help="评测结果输出目录")
    eval_parser.add_argument("--bert_model_path", help="BERT模型路径，用于计算语义相似度 (可选)")
    eval_parser.add_argument("--openai_api_key", help="OpenAI API密钥，用于LLM评测 (可选)")
    eval_parser.add_argument("--openai_base_url", help="OpenAI API基础URL (可选，用于兼容其他API)")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info("启动EduEval工具...")
    
    if args.command == "generate":
        logger.info("执行模型生成...")
        generate_model_outputs(args)
    elif args.command == "evaluate":
        logger.info("执行评测...")
        evaluate_model_outputs(args)
    else:
        parser.print_help()
    
    logger.info("EduEval工具执行完成")

if __name__ == "__main__":
    main()