# edueval/code/generation/run_generation.py
import os
import sys
import torch
import time
import argparse
from pathlib import Path

# Add the generation code directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 通用导入，可以根据需要使用不同模型的生成器
from glm_gen import GLMGenerator  # 取消注释以支持GLM模型
from qwen_gen import QwenGenerator
# 当需要使用DeepSeek模型时取消注释下面的导入
# from deepseek_gen import DeepSeekGenerator
# API生成器
from api_gen import ApiGenerator

# 检查是否有任务类型提示环境变量
TASK_TYPE_HINT = os.environ.get("TASK_TYPE_HINT", "")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行EduEval模型生成评测')
    parser.add_argument('--model', type=str, default='qwen', choices=['qwen', 'deepseek', 'api', 'glm'], 
                        help='要使用的模型类型 (默认: qwen)')
    parser.add_argument('--model_path', type=str, 
                        default='/home/disk2/mgq/benchmark/model/Qwen-14B-Chat',
                        help='模型路径')
    parser.add_argument('--model_name', type=str, default='Qwen-14B-Chat',
                        help='模型名称，用于输出目录命名')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/disk2/mgq/benchmark/edueval/data',
                        help='数据目录路径')
    parser.add_argument('--output_dir', type=str,
                        default='/home/disk2/mgq/benchmark/edueval/outputs',
                        help='输出根目录路径')
    parser.add_argument('--direct_output_file', type=str, default=None,
                        help='直接指定输出文件的绝对路径，优先级高于output_dir')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU设备ID')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批处理大小')
    parser.add_argument('--max_items', type=int, default=None,
                        help='每个文件处理的最大项目数 (None表示全部处理)')
    parser.add_argument('--task_files', type=str, nargs='*', 
                        help='要处理的特定任务文件列表 (如果未指定，将处理数据目录中的所有.jsonl文件)')
    parser.add_argument('--vllm', action='store_true',
                        help='使用vLLM加速推理')
    # Ollama相关参数
    parser.add_argument('--use_ollama', action='store_true',
                        help='使用Ollama服务而非直接加载模型')
    parser.add_argument('--ollama_url', type=str, default='http://localhost:11434',
                        help='Ollama服务的URL (默认: http://localhost:11434)')
    parser.add_argument('--ollama_model', type=str, default='deepseek-r1:32b',
                        help='Ollama服务中的模型名称 (默认: deepseek-r1:32b)')
    # API相关参数
    parser.add_argument('--use_api', action='store_true',
                        help='使用OpenAI兼容API接口')
    parser.add_argument('--api_mode', action='store_true',
                        help='与--use_api相同，使用API模式 (为了兼容性)')
    parser.add_argument('--api_base_url', type=str, default='https://spark-api-open.xf-yun.com/v2.1',
                        help='API基础URL (默认: https://spark-api-open.xf-yun.com/v2.1)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API密钥 (格式为 key_id:key_secret)')
    parser.add_argument('--api_model', type=str, default='spark',
                        help='API接口使用的模型名称 (默认: spark)')
    
    args = parser.parse_args()

    # --- 配置设置 ---
    model_type = args.model
    model_name = args.model_name
    model_path = args.model_path
    base_data_dir = args.data_dir
    output_dir = args.output_dir
    direct_output_file = args.direct_output_file
    base_output_dir = f"{output_dir}/{model_name}"
    device = args.device
    batch_size = args.batch_size
    max_items_per_file = args.max_items
    use_vllm = args.vllm
    use_ollama = args.use_ollama
    ollama_url = args.ollama_url
    ollama_model = args.ollama_model
    # API相关
    use_api = args.use_api or args.api_mode or args.model == 'api'  # 添加api_mode支持
    api_base_url = args.api_base_url
    api_key = args.api_key
    api_model = args.api_model
    # -----------------

    print(f"{'='*30} EduEval Benchmark {'='*30}")
    if use_api:
        print(f"开始为API接口中的模型生成评测结果: {api_model}")
        print(f"API基础URL: {api_base_url}")
        # 使用api_model作为模型名，覆盖之前的model_name
        model_name = api_model
        base_output_dir = f"{output_dir}/api_{model_name.replace(':', '_')}"
    elif use_ollama:
        print(f"开始为Ollama服务中的模型生成评测结果: {ollama_model}")
        print(f"Ollama服务URL: {ollama_url}")
        # 使用ollama_model作为模型名，覆盖之前的model_name
        model_name = ollama_model
        base_output_dir = f"{output_dir}/ollama_{model_name.replace(':', '_')}"
    else:
        print(f"开始为模型生成评测结果: {model_name} (类型: {model_type})")
        print(f"模型路径: {model_path}")
    
    print(f"数据目录: {base_data_dir}")
    print(f"输出目录: {base_output_dir}")
    print(f"设备: GPU {device}")
    print(f"批处理大小: {batch_size}")
    if not use_ollama and not use_api:
        print(f"使用vLLM: {use_vllm}")
    start_time = time.time()

    # 如果设置了任务类型提示，显示提示信息
    if TASK_TYPE_HINT:
        print(f"已通过环境变量设置任务类型提示为: {TASK_TYPE_HINT}")

    # 确保输出基础目录存在
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)

    # 查找所有.jsonl文件
    if args.task_files:
        # 使用用户指定的任务文件列表
        task_files = args.task_files
        print(f"使用用户指定的 {len(task_files)} 个任务文件。")
    else:
        # 默认任务文件列表
        task_files = [
            "/home/disk2/mgq/benchmark/edueval/data/理解/2_yuedu_100.jsonl",
            "/home/disk2/mgq/benchmark/edueval/data/理解/2_shige_100.jsonl",
            "/home/disk2/mgq/benchmark/edueval/data/应用/3_zuowen_100.jsonl"
        ]
        # 自动查找数据目录中的所有.jsonl文件
        for root, _, files in os.walk(base_data_dir):
            for file in files:
                if file.endswith(".jsonl"):
                    file_path = os.path.join(root, file)
                    if file_path not in task_files:  # 避免重复
                        task_files.append(file_path)

    print(f"找到 {len(task_files)} 个任务文件需要处理。")

    # 记录成功和失败的文件
    successful_files = []
    failed_files = []

    # 遍历处理每个任务文件
    for task_path in task_files:
        # 如果直接指定了输出文件路径，使用它
        if direct_output_file and len(task_files) == 1:
            output_file_path = direct_output_file
            # 确保输出文件的目录存在
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        else:
            # 原来的路径计算逻辑
            relative_path = os.path.relpath(task_path, base_data_dir)
            output_subdir = os.path.join(base_output_dir, os.path.dirname(relative_path))
            output_filename = os.path.basename(relative_path)
            output_file_path = os.path.join(output_subdir, output_filename)
            # 确保特定的输出子目录存在
            Path(output_subdir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'-'*75}")
        print(f"正在处理任务文件: {task_path}")
        print(f"输出将保存到: {output_file_path}")

        # 初始化生成器，模型和分词器变量（在try块外）
        generator = None
        model = None
        tokenizer = None
        try:
            # 根据模型类型和运行模式选择相应的生成器
            if use_api:
                # 使用API模式
                generator = ApiGenerator(
                    task_path=task_path,
                    model_name=api_model,
                    base_url=api_base_url,
                    api_key=api_key,
                    device=device,
                    is_few_shot=False,
                    few_shot_path=None,
                    output_file=output_file_path
                )
            elif use_ollama:
                # 使用Ollama模式，需要导入DeepSeekGenerator
                from deepseek_gen import DeepSeekGenerator
                generator = DeepSeekGenerator(
                    task_path=task_path,
                    model_path=None,  # Ollama模式下不需要模型路径
                    model_name=ollama_model,
                    device=device,
                    is_few_shot=False,
                    few_shot_path=None,
                    ollama_base_url=ollama_url,
                    output_file=output_file_path
                )
            elif model_type.lower() == 'qwen':
                generator = QwenGenerator(
                    task_path=task_path,
                    model_path=model_path,
                    model_name=model_name,
                    device=device,
                    is_few_shot=False,
                    few_shot_path=None,
                    is_vllm=use_vllm,
                    output_file=output_file_path
                )
            elif model_type.lower() == 'deepseek':
                # 确保已经导入了DeepSeekGenerator
                from deepseek_gen import DeepSeekGenerator
                generator = DeepSeekGenerator(
                    task_path=task_path,
                    model_path=model_path,
                    model_name=model_name,
                    device=device,
                    is_few_shot=False,
                    few_shot_path=None,
                    is_vllm=use_vllm,
                    output_file=output_file_path
                )
            elif model_type.lower() == 'glm':
                # 使用GLM生成器
                generator = GLMGenerator(
                    task_path=task_path,
                    model_path=model_path,
                    model_name=model_name,
                    device=device,
                    is_few_shot=False,
                    few_shot_path=None,
                    is_vllm=use_vllm,
                    output_file=output_file_path
                )
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

            # 如果通过环境变量TASK_TYPE_HINT指定了任务类型，则覆盖自动判断的任务类型
            if TASK_TYPE_HINT:
                print(f"原始任务类型确定为: {generator.task_type}")
                generator.task_type = TASK_TYPE_HINT
                print(f"根据环境变量覆盖任务类型为: {generator.task_type}")
            else:
                # 否则按照正常方式判断任务类型
                print(f"任务类型确定为: {generator.task_type}")
                
            if generator.task_type == "unknown":
                print(f"由于任务类型未知，跳过文件 {task_path}。")
                failed_files.append((task_path, "未知任务类型"))
                continue

            # 初始化模型和分词器
            print("正在初始化模型和分词器...")
            model_init_start = time.time()
            model, tokenizer = generator.model_init()
            model_init_time = time.time() - model_init_start
            print(f"模型和分词器初始化完成，耗时: {model_init_time:.2f}秒。")

            # 生成输出
            print("开始生成...")
            generation_start = time.time()
            generator.generate_output(
                tokenizer=tokenizer,
                model=model,
                batch_size=batch_size,
                max_items=max_items_per_file,
                offset=0
            )
            generation_time = time.time() - generation_start
            print(f"处理完成 {task_path}。结果已保存。")
            print(f"生成耗时: {generation_time:.2f}秒。")
            
            successful_files.append(task_path)

        except FileNotFoundError as e:
            print(f"错误: {e}。跳过此文件。")
            failed_files.append((task_path, str(e)))
        except Exception as e:
            print(f"处理 {task_path} 时发生意外错误: {e}")
            failed_files.append((task_path, str(e)))
        finally:
            # 清理GPU内存（处理多个文件时很重要）
            # 即使在生成/加载过程中发生错误，这部分也会执行
            print("尝试清理GPU内存...")
            if model is not None:
                del model
                print("已删除模型对象。")
            if tokenizer is not None:
                del tokenizer
                print("已删除分词器对象。")
            if generator is not None:
                del generator
                print("已删除生成器对象。")
            torch.cuda.empty_cache()
            print("执行了 torch.cuda.empty_cache()。")

    # 总结处理结果
    total_time = time.time() - start_time
    print(f"\n{'='*75}")
    print(f"所有任务处理完成。总耗时: {total_time:.2f}秒")
    print(f"成功处理: {len(successful_files)}/{len(task_files)} 文件")
    
    if failed_files:
        print("\n处理失败的文件:")
        for idx, (file_path, error) in enumerate(failed_files, 1):
            print(f"{idx}. {file_path} - 原因: {error}")

if __name__ == "__main__":
    main() 