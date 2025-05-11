# EduEval Benchmark Tool

EduEval is a comprehensive tool for generating and evaluating model outputs in the educational domain, supporting various model access methods and evaluation task types.

## Main Features

1. **Model Generation**: Generate answers using multiple large language models
2. **Model Evaluation**: Assess model performance across different educational tasks
3. **Multiple Model Access Methods**: Support for local models, Ollama service, and API calls

## Supported Model Access Methods

1. **Local Models**: Directly load locally stored model files (e.g., GLM4, Qwen14B, EduChat, etc.)
2. **Ollama Service**: Call models through Ollama service (e.g., DeepSeek, Llama, Baichuan, etc.)
3. **API Calls**: Use third-party API interfaces (e.g., Spark API, Qwen-Plus, etc.)

## Supported Task Types

- Multiple Choice Questions (multiple_choice)
- Short Answer Questions (short_answer)
- Essay Writing (essay)
- Essay Grading (essay_grading)
- Question Generation (question_generation)
- Teaching Design (teaching_design)
- Classroom Conversation Classification (conversation_classification)
- Reading/Poetry Similarity (reading_poetry_similarity)

## Command Line Usage

### 1. Model Generation Command

```bash
python main.py generate [parameters]
```

#### Parameter Description

##### Model Access Parameters

- `--model_access_type`: Specify model access method [`local`, `ollama`, `api`], default is `local`

##### Local Model Parameters

- `--model_path`: Local model path
- `--model_name`: Model name for output directory
- `--vllm`: Use vLLM for accelerated inference
- `--tensor_parallel_size`: vLLM tensor parallel size
- `--gpu_memory_utilization`: vLLM GPU memory utilization rate

##### Ollama Parameters

- `--ollama_base_url`: Ollama service URL, default is `http://localhost:11434`
- `--ollama_model`: Model name in Ollama service, default is `deepseek-r1:32b`

##### API Parameters

- `--api_base_url`: API base URL, default is `https://spark-api-open.xf-yun.com/v2.1`
- `--api_key`: API key (format: `key_id:key_secret`)
- `--api_model`: Model name used for API interface, default is `spark`

##### General Parameters

- `--data_dir`: Data directory path
- `--output_dir`: Output root directory path
- `--direct_output_file`: Directly specify absolute path of output file
- `--device`: GPU device ID
- `--batch_size`: Batch size
- `--max_items`: Maximum number of items to process per file
- `--offset`: Starting offset for data processing
- `--category`: Data category to process (e.g., "Understanding", "Memory", etc.)
- `--task_type`: Task type to process (e.g., "primary", "junior", etc.)
- `--task_file`: Path to a specific task file
- `--task_files`: List of specific task files to process
- `--is_few_shot`: Whether to use few-shot learning
- `--few_shot_path`: Path to few-shot examples file

### 2. Model Evaluation Command

```bash
python main.py evaluate [parameters]
```

#### Parameter Description

- `--result_dir`: Model output result directory (required)
- `--output_dir`: Evaluation result output directory
- `--bert_model_path`: BERT model path for semantic similarity calculation (optional)
- `--openai_api_key`: OpenAI API key for LLM evaluation (optional)
- `--openai_base_url`: OpenAI API base URL (optional, for compatibility with other APIs)

## Usage Examples

### Using Local Models (e.g., Qwen-14B)

```bash
python main.py generate --model_access_type local \
                      --model_path /path/to/Qwen-14B-Chat \
                      --model_name Qwen-14B-Chat \
                      --data_dir /path/to/data \
                      --output_dir /path/to/outputs \
                      --device 0
```

### Using Ollama Service

```bash
python main.py generate --model_access_type ollama \
                      --ollama_model deepseek-r1:32b \
                      --ollama_base_url http://localhost:11434 \
                      --data_dir /path/to/data \
                      --output_dir /path/to/outputs
```

### Using API Interface (e.g., Spark API)

```bash
python main.py generate --model_access_type api \
                      --api_model spark \
                      --api_base_url https://spark-api-open.xf-yun.com/v2.1 \
                      --api_key your_key_id:your_key_secret \
                      --data_dir /path/to/data \
                      --output_dir /path/to/outputs
```

### Evaluating Model Outputs

```bash
python main.py evaluate --result_dir /path/to/outputs/model_name \
                      --output_dir /path/to/results \
                      --bert_model_path /path/to/bert_model
```

## Environment Variables

- `TASK_TYPE_HINT`: Set this environment variable to manually specify task type, overriding automatic detection

## File Structure

```
edueval/code/
├── main.py                         # Main program entry
├── generation/                     # Generation module directory
│   ├── model_gen.py                # Base class defining common generation interface
│   ├── qwen_gen.py                 # Qwen model generator
│   ├── glm_gen.py                  # GLM model generator
│   ├── deepseek_gen.py             # DeepSeek model generator (supports Ollama)
│   ├── baichuan_gen.py             # Baichuan model generator
│   ├── ollama_baichuan_gen.py      # Baichuan model generator for Ollama
│   ├── api_gen.py                  # API call generator
│   └── run_generation.py           # Standalone script for running generations
├── evaluation/                     # Evaluation module directory
│   └── evaluator.py                # Evaluator implementation
└── README.md                       # Documentation
```

## Generation Module Updates

The generation directory has been updated with:

1. **Enhanced Base Class**: The `model_gen.py` file now includes support for more task types:
   - Question generation task type
   - Teaching design task type
   - Classroom conversation classification task type

2. **Additional Model Support**:
   - Added `baichuan_gen.py` for Baichuan model support
   - Added `ollama_baichuan_gen.py` for Baichuan model via Ollama

3. **Standalone Generation Script**: 
   - New `run_generation.py` provides a standalone interface for running model generations
   - Supports all model types (local, Ollama, API) with unified command-line arguments

4. **Improved Task Type Detection**:
   - Enhanced file path and name-based task type detection
   - Support for detecting tasks in ethical reasoning and education directories

## Extending to New Models

To support a new model, you can create a new generator class by referencing existing implementations (like `qwen_gen.py`) and inheriting from the `ModelGenerator` base class. The main requirement is to implement the `model_init()` method. 