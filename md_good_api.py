import os
import shutil
import time
import logging
import re
import requests
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
import sys
import concurrent.futures
from multiprocessing import cpu_count
from openai import OpenAI  # 添加OpenAI客户端

# ===================== 配置区域 =====================
INPUT_FOLDER ="/home/fusion/profile/before0621/md_output_20250731_115811/batch_1/" # 输入目录
DASHSCOPE_API_KEY = "sk-73748cae71ed4f758c93b0ccd03c0cda"  # 阿里云API密钥
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云API地址
MODEL_NAME = "deepseek-r1-distill-llama-70b"  # 阿里云模型名称
MAX_CHARS_FIRST = 3000  # 首次检测读取字符数
MAX_CHARS_SECOND = 5000  # 二次检测读取字符数
GPU_ENABLED = True  # 模型已确保GPU加速
NUM_GPU = 2  # 使用的GPU数量（模型已配置）
MAX_WORKERS = min(cpu_count(), 2)  # 基于CPU核心数动态调整
TIMEOUT = 300  # 超时时间
RETRY_LIMIT = 3  # 重试次数
REQUEST_INTERVAL = 1  # 减小请求间隔
BATCH_SIZE = 10  # 批次大小
PARALLEL_PROCESSING = False  # 启用并行处理
# ===================================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("md_classifier_gpu.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# 生成时间戳，用于输出文件夹
timestamp = time.strftime("%Y%m%d_%H%M%S")

# 使用时间戳创建带时间戳的输出文件夹
GOOD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(INPUT_FOLDER)), f"good_{timestamp}")
TMP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(INPUT_FOLDER)), f"tmp_{timestamp}")
BAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(INPUT_FOLDER)), f"bad_{timestamp}")

os.makedirs(GOOD_FOLDER, exist_ok=True)
os.makedirs(TMP_FOLDER, exist_ok=True)
os.makedirs(BAD_FOLDER, exist_ok=True)

# 领域关键词定义（优化为集合，提高查找速度）
CORE_KEYWORDS = {
    # 中文核心关键词
    "海上风电", "风机", "风力发电机", "海缆", "海底电缆",
    "单桩", "单桩基础", "升压站", "防腐涂层", "LCOE",
    "平准化度电成本", "尾流效应", "地质勘探", "风电场设计",
    "海上施工", "风机安装", "运维", "海上风电场规划",
    # 英文核心关键词
    "offshore wind", "wind turbine", "wind generator",
    "submarine cable", "undersea cable", "monopile",
    "monopile foundation", "substation", "anticorrosive coating",
    "levelized cost of electricity", "wake effect", "geological survey",
    "wind farm design", "offshore construction", "turbine installation",
    "operation and maintenance", "offshore wind planning"
}

RELATED_FIELDS = {
    # 中文邻近领域
    "海洋工程", "海洋技术", "动力学", "结构工程", "岩土工程",
    "并网技术", "风场", "风电场", "腐蚀防护", "海上气象",
    "海上可再生能源", "可再生能源", "海洋能源",
    "清洁能源技术", "船舶工程", "港口建设", "海洋环境",
    # 英文邻近领域
    "marine engineering", "ocean technology", "dynamics",
    "structural engineering", "geotechnical engineering",
    "grid connection technology", "wind farm", "corrosion protection",
    "marine meteorology", "offshore renewable energy",
    "renewable energy", "marine energy", "clean energy technology",
    "naval architecture", "port construction", "marine environment"
}

# 首次检测提示词（核心和邻近领域）
FIRST_PROMPT_TEMPLATE = """
### 海上风电领域首次分类
**核心关键词**: {core_keywords}
**邻近领域**: {related_fields}

**判断标准**:
1. 包含任一核心关键词或邻近领域关键词 → 是
2. 内容涉及海洋、风能、工程等海上风电相关领域 → 是
3. 文件可能包含与海上风电间接相关的信息 → 是
4. 完全不涉及任何技术或相关领域内容 → 否

**待评估文本(前{max_chars}字符)**:
{content}

**输出**: 仅"是"或"否"
"""

# 二次检测提示词（核心和邻近领域）
SECOND_PROMPT_TEMPLATE = """
### 海上风电领域二次分类
**核心关键词**: {core_keywords}
**邻近领域**: {related_fields}

**判断标准**:
1. 文本中包含任何海上风电相关关键词 → 是
2. 内容涉及海洋、能源或工程领域的通用知识 → 是
3. 文件可能包含对海上风电有价值的信息 → 是
4. 完全不相关的领域（如纯文学、娱乐、个人日常等） → 否

**待评估文本(前{max_chars}字符)**:
{content}

**输出**: 仅"是"或"否"
"""

# 线程安全锁
print_lock = threading.Lock()
output_lock = threading.Lock()

# 预编译正则表达式（优化关键词匹配）
ALL_KEYWORDS = CORE_KEYWORDS.union(RELATED_FIELDS)
KEYWORD_PATTERN = re.compile(r'|'.join(re.escape(kw) for kw in ALL_KEYWORDS), re.IGNORECASE)
CORE_KEYWORD_PATTERN = re.compile(r'|'.join(re.escape(kw) for kw in CORE_KEYWORDS), re.IGNORECASE)
RELATED_KEYWORD_PATTERN = re.compile(r'|'.join(re.escape(kw) for kw in RELATED_FIELDS), re.IGNORECASE)

# 创建OpenAI客户端实例
dashscope_client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL
)


def test_dashscope_connection():
    """测试阿里云百炼连接和模型可用性"""
    try:
        logger.info(f"测试阿里云百炼服务连接: {DASHSCOPE_BASE_URL}")

        # 测试简单请求
        response = dashscope_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "测试连接"}],
            max_tokens=5,
            timeout=10
        )

        if response.choices[0].message.content:
            logger.info(f"阿里云百炼服务连接成功，模型响应正常")
            return True
        else:
            logger.error(f"模型验证失败: 无有效响应内容")
            return False

    except Exception as e:
        logger.error(f"阿里云百炼连接失败: {str(e)}")
        return False


def find_md_files(input_folder):
    """递归遍历文件夹，查找所有 .md 文件"""
    md_files = []
    for root, dirs, files in os.walk(input_folder, topdown=True):  # 使用os.walk遍历多级目录
        for file in files:
            if file.lower().endswith('.md'):  # 只查找 .md 文件
                md_files.append(os.path.join(root, file))
    return md_files


def check_dashscope_health():
    """检查阿里云百炼服务状态"""
    try:
        logger.info(f"检查阿里云百炼服务状态: {DASHSCOPE_BASE_URL}")

        # 测试简单请求
        response = dashscope_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "健康检查"}],
            max_tokens=5,
            timeout=15
        )

        if response.choices[0].message.content:
            logger.info("阿里云百炼服务健康检查通过")
            return True
        else:
            logger.warning("服务响应异常，但模型可能可用")
            return True

    except Exception as e:
        logger.error(f"阿里云百炼健康检查失败: {str(e)}")
        logger.warning("忽略错误，继续执行...")
        return True


def contains_keywords(text):
    """关键词快速匹配（中英文不区分大小写）"""
    try:
        return bool(KEYWORD_PATTERN.search(text))
    except Exception as e:
        logger.error(f"关键词匹配失败: {str(e)}")
        return False


def check_filename_relevance(filename):
    """文件名多语言检测（优化正则匹配）"""
    base_name = os.path.basename(filename).lower()
    # 核心关键词匹配
    if CORE_KEYWORD_PATTERN.search(base_name):
        return "core"  # 直接通过
    # 相关领域匹配
    if RELATED_KEYWORD_PATTERN.search(base_name):
        return "related"  # 需要检查内容
    return "unrelated"  # 文件名无关


def classify_text(content, prompt_template, max_chars, retry_count=0):
    """通用分类函数（优化请求参数）"""
    if retry_count > RETRY_LIMIT:
        logger.warning(f"达最大重试次数，放弃")
        return True  # 重试失败后默认通过（更宽松）

    try:
        prompt = prompt_template.format(
            core_keywords=", ".join(CORE_KEYWORDS),
            related_fields=", ".join(RELATED_FIELDS),
            max_chars=max_chars,
            content=content[:max_chars]
        )

        time.sleep(random.uniform(0, REQUEST_INTERVAL))

        # 使用阿里云百炼API
        response = dashscope_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            top_p=0.9,
            max_tokens=50,  # 限制生成长度
            timeout=TIMEOUT
        )

        res_text = response.choices[0].message.content.strip()

        # 响应判断
        if "是" in res_text or "相关" in res_text or "yes" in res_text.lower() or "保留" in res_text:
            return True
        elif "否" in res_text or "无关" in res_text or "no" in res_text.lower() or "删除" in res_text:
            return False
        else:
            logger.warning(f"异常响应: {res_text[:50]}，默认保留")
            # 默认认为是相关文档（更宽松）
            return True

    except Exception as e:
        logger.error(f"分类失败: {str(e)}")
        return classify_text(content, prompt_template, max_chars, retry_count + 1)


def process_first_pass(file_path):
    """首次处理：核心和邻近领域标准"""
    try:
        # 跳过已处理文件
        relative_path = os.path.relpath(file_path, INPUT_FOLDER)
        good_path = os.path.join(GOOD_FOLDER, relative_path)
        bad_path = os.path.join(BAD_FOLDER, relative_path)

        if os.path.exists(good_path) or os.path.exists(bad_path):
            return ("skip", None)

        # 空文件检测
        if os.path.getsize(file_path) == 0:
            os.makedirs(os.path.dirname(bad_path), exist_ok=True)
            shutil.move(file_path, bad_path)
            return ("empty", None)

        # 文件名检测 - 核心领域直接通过
        filename_relevance = check_filename_relevance(file_path)
        if filename_relevance == "core":
            os.makedirs(os.path.dirname(good_path), exist_ok=True)
            shutil.move(file_path, good_path)
            return ("moved", None)
        elif filename_relevance == "related":
            # 文件名相关的文件需要检查内容
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(MAX_CHARS_FIRST)

            if not content.strip():
                os.makedirs(os.path.dirname(bad_path), exist_ok=True)
                shutil.move(file_path, bad_path)
                return ("empty", None)

            # 分类
            is_related = classify_text(content, FIRST_PROMPT_TEMPLATE, MAX_CHARS_FIRST)

            if is_related:
                os.makedirs(os.path.dirname(good_path), exist_ok=True)
                shutil.move(file_path, good_path)
                return ("moved", None)
            else:
                tmp_path = os.path.join(TMP_FOLDER, relative_path)
                os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                shutil.move(file_path, tmp_path)
                return ("tmp", tmp_path)
        else:  # 文件名无关
            tmp_path = os.path.join(TMP_FOLDER, relative_path)
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            shutil.move(file_path, tmp_path)
            return ("tmp", tmp_path)

    except Exception as e:
        logger.error(f"首次处理失败 {file_path}: {str(e)}")
        # 出错时默认移动到临时文件夹
        relative_path = os.path.relpath(file_path, INPUT_FOLDER)
        tmp_path = os.path.join(TMP_FOLDER, relative_path)
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        shutil.move(file_path, tmp_path)
        return ("tmp", tmp_path)


def process_second_pass(file_path):
    """二次处理：核心和邻近领域标准"""
    try:
        # 读取更多内容
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(MAX_CHARS_SECOND)

        if not content.strip():
            relative_path = os.path.relpath(file_path, TMP_FOLDER)
            bad_path = os.path.join(BAD_FOLDER, relative_path)
            os.makedirs(os.path.dirname(bad_path), exist_ok=True)
            shutil.move(file_path, bad_path)
            return False

        # 二次分类
        is_related = classify_text(content, SECOND_PROMPT_TEMPLATE, MAX_CHARS_SECOND)

        if is_related:
            relative_path = os.path.relpath(file_path, TMP_FOLDER)
            good_path = os.path.join(GOOD_FOLDER, relative_path)
            os.makedirs(os.path.dirname(good_path), exist_ok=True)
            shutil.move(file_path, good_path)
            return True
        else:
            # 二次分类不通过，移动到带时间戳的bad文件夹
            relative_path = os.path.relpath(file_path, TMP_FOLDER)
            bad_path = os.path.join(BAD_FOLDER, relative_path)
            os.makedirs(os.path.dirname(bad_path), exist_ok=True)
            shutil.move(file_path, bad_path)
            return False

    except Exception as e:
        logger.error(f"二次处理失败 {file_path}: {str(e)}")
        # 出错时默认保留在临时文件夹
        return False


def cleanup_empty_directories(root_folder):
    """清理空目录（优化为多进程）"""
    if not PARALLEL_PROCESSING:
        for root, dirs, _ in os.walk(root_folder, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                        logger.debug(f"清理空目录: {dir_path}")
                except Exception as e:
                    logger.warning(f"无法清理目录 {dir_path}: {str(e)}")
        return

    # 并行清理空目录
    dirs_to_check = []
    for root, dirs, _ in os.walk(root_folder, topdown=False):
        for dir_name in dirs:
            dirs_to_check.append(os.path.join(root, dir_name))

    def remove_if_empty(dir_path):
        try:
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                logger.debug(f"清理空目录: {dir_path}")
        except Exception as e:
            logger.warning(f"无法清理目录 {dir_path}: {str(e)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(remove_if_empty, dirs_to_check),
                  total=len(dirs_to_check), desc="清理空目录"))


def main():
    logger.info("===== 海上风电文档分类器（阿里云百炼版）启动 =====")
    logger.info(f"输入目录: {INPUT_FOLDER}")
    logger.info(f"使用模型: {MODEL_NAME} | 批次大小: {BATCH_SIZE}")
    logger.info(f"API地址: {DASHSCOPE_BASE_URL}")

    # 检查阿里云百炼服务
    if not test_dashscope_connection() or not check_dashscope_health():
        logger.error("服务检查失败，但继续执行...")

    # 第一阶段：处理原始文件夹中的文件
    logger.info("===== 开始第一阶段：处理原始文件夹中的文件 =====")
    first_pass_files = find_md_files(INPUT_FOLDER)  # 递归查找所有MD文件
    total_files = len(first_pass_files)
    if total_files == 0:
        logger.info("原始文件夹中未找到MD文件，直接进入第二阶段")
    else:
        logger.info(f"共找到 {total_files} 个MD文件，开始分批次处理...")

        # 初始化统计
        first_stats = {
            "moved": 0, "tmp": 0, "empty": 0, "error": 0, "skip": 0
        }
        total_batch = (total_files + BATCH_SIZE - 1) // BATCH_SIZE

        # 分批次处理
        for batch_idx in range(total_batch):
            start = batch_idx * BATCH_SIZE
            end = min((batch_idx + 1) * BATCH_SIZE, total_files)
            batch_files = first_pass_files[start:end]
            batch_num = batch_idx + 1
            logger.info(f"\n===== 批次 {batch_num}/{total_batch} 开始（{len(batch_files)}个文件） =====")

            # 批次内统计
            batch_stats = {
                "moved": 0, "tmp": 0, "empty": 0, "error": 0, "skip": 0
            }

            # 处理批次文件
            if not PARALLEL_PROCESSING:
                # 串行处理（适合调试）
                for file in tqdm(batch_files, desc=f"批次{batch_num}处理"):
                    try:
                        result, _ = process_first_pass(file)
                        batch_stats[result] += 1
                    except Exception as e:
                        logger.error(f"处理异常: {str(e)}")
                        batch_stats["error"] += 1
            else:
                # 并行处理
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {executor.submit(process_first_pass, f): f for f in batch_files}
                    with tqdm(total=len(batch_files), desc=f"批次{batch_num}处理") as pbar:
                        for future in as_completed(futures):
                            try:
                                result, _ = future.result()
                                batch_stats[result] += 1
                                pbar.update(1)
                            except Exception as e:
                                logger.error(f"批次{batch_num}任务异常: {str(e)}")
                                batch_stats["error"] += 1
                                pbar.update(1)

            # 更新全局统计
            first_stats["moved"] += batch_stats["moved"]
            first_stats["tmp"] += batch_stats["tmp"]
            first_stats["empty"] += batch_stats["empty"]
            first_stats["error"] += batch_stats["error"]
            first_stats["skip"] += batch_stats["skip"]

            # 批次结果
            logger.info(f"批次{batch_num}处理结果: "
                        f"通过={batch_stats['moved']}, "
                        f"待二次检测={batch_stats['tmp']}, "
                        f"空文件={batch_stats['empty']}, "
                        f"错误={batch_stats['error']}")

        # 第一阶段结果
        logger.info("\n" + "=" * 60)
        logger.info("第一阶段处理完成！")
        logger.info(f"第一阶段统计:")
        logger.info(f"  直接通过文件: {first_stats['moved']}（已移动到good文件夹）")
        logger.info(f"  待二次检测文件: {first_stats['tmp']}（已移动到tmp文件夹）")
        logger.info(f"  空文件: {first_stats['empty']}（已移动到bad文件夹）")
        logger.info(f"  错误文件: {first_stats['error']}（已移动到bad文件夹）")
        logger.info(f"  跳过文件: {first_stats['skip']}")
        logger.info("=" * 60)

        # 清理原始文件夹中的空目录
        cleanup_empty_directories(INPUT_FOLDER)
        logger.info(f"原始文件夹空目录清理完成，当前状态: {'空' if not os.listdir(INPUT_FOLDER) else '非空'}")

    # 第二阶段：处理tmp文件夹中的文件
    logger.info("\n===== 开始第二阶段：处理tmp文件夹中的文件 =====")
    second_pass_files = find_md_files(TMP_FOLDER)  # 递归查找所有MD文件
    total_second_files = len(second_pass_files)
    if total_second_files == 0:
        logger.info("tmp文件夹中未找到MD文件，处理完成")
    else:
        logger.info(f"tmp文件夹中共找到 {total_second_files} 个MD文件，开始二次检测...")

        # 初始化统计
        second_stats = {
            "moved": 0, "rejected": 0, "error": 0
        }
        total_second_batch = (total_second_files + BATCH_SIZE - 1) // BATCH_SIZE

        # 分批次处理
        for batch_idx in range(total_second_batch):
            start = batch_idx * BATCH_SIZE
            end = min((batch_idx + 1) * BATCH_SIZE, total_second_files)
            batch_files = second_pass_files[start:end]
            batch_num = batch_idx + 1
            logger.info(f"\n===== 二次检测批次 {batch_num}/{total_second_batch} 开始 =====")

            # 批次内统计
            batch_stats = {
                "moved": 0, "rejected": 0, "error": 0
            }

            # 处理批次文件
            if not PARALLEL_PROCESSING:
                # 串行处理
                for file in tqdm(batch_files, desc=f"批次{batch_num}二次检测"):
                    try:
                        if process_second_pass(file):
                            batch_stats["moved"] += 1
                        else:
                            batch_stats["rejected"] += 1
                    except Exception as e:
                        logger.error(f"处理异常: {str(e)}")
                        batch_stats["error"] += 1
            else:
                # 并行处理
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {executor.submit(process_second_pass, f): f for f in batch_files}
                    with tqdm(total=len(batch_files), desc=f"批次{batch_num}二次检测") as pbar:
                        for future in as_completed(futures):
                            try:
                                if future.result():
                                    batch_stats["moved"] += 1
                                else:
                                    batch_stats["rejected"] += 1
                                pbar.update(1)
                            except Exception as e:
                                logger.error(f"批次{batch_num}任务异常: {str(e)}")
                                batch_stats["error"] += 1
                                pbar.update(1)

            # 更新全局统计
            second_stats["moved"] += batch_stats["moved"]
            second_stats["rejected"] += batch_stats["rejected"]
            second_stats["error"] += batch_stats["error"]

            # 批次结果
            logger.info(f"二次检测批次{batch_num}结果: "
                        f"通过={batch_stats['moved']}（已移动到good文件夹）, "
                        f"拒绝={batch_stats['rejected']}（已移动到bad文件夹）, "
                        f"错误={batch_stats['error']}")

        # 第二阶段结果
        logger.info("\n" + "=" * 60)
        logger.info("第二阶段处理完成！")
        logger.info(f"第二阶段统计:")
        logger.info(f"  二次检测通过文件: {second_stats['moved']}（已移动到good文件夹）")
        logger.info(f"  最终拒绝文件: {second_stats['rejected']}（已移动到bad文件夹）")
        logger.info(f"  错误文件: {second_stats['error']}（保留在tmp文件夹）")
        logger.info("=" * 60)

        # 清理tmp文件夹中的空目录
        cleanup_empty_directories(TMP_FOLDER)
        logger.info(f"tmp文件夹空目录清理完成，当前状态: {'空' if not os.listdir(TMP_FOLDER) else '非空'}")

    # 最终统计
    logger.info("\n" + "=" * 60)
    logger.info("所有文件处理完成！")
    logger.info(f"最终统计:")
    logger.info(f"  通过文件总数: {first_stats.get('moved', 0) + second_stats.get('moved', 0)}")
    logger.info(f"  拒绝文件: {first_stats.get('empty', 0) + second_stats.get('rejected', 0)}")
    logger.info(f"  临时保留文件: {first_stats.get('tmp', 0) + second_stats.get('error', 0)}")
    logger.info(f"  原始文件夹状态: {'空' if not os.listdir(INPUT_FOLDER) else '非空'}")
    logger.info(f"  通过文件存放于: {GOOD_FOLDER}")
    logger.info(f"  拒绝文件存放于: {BAD_FOLDER}")
    logger.info(f"  临时保留文件存放于: {TMP_FOLDER}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("程序被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序异常终止: {str(e)}")
        sys.exit(1)
