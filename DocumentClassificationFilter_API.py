# -*- coding: utf-8 -*-
"""
海上风电领域专业文档分类器（云端API版）
功能：
1. 使用云端AI模型进行严格领域判定
2. 两阶段分类流程（核心领域→专业相关领域）
3. 并行批量处理Markdown文件
4. 自动分类到时间戳标记的文件夹
"""

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
from typing import List, Dict, Tuple, Optional

# ===================== 配置区域 =====================
INPUT_FOLDER = "/home/fusion/profile/汇总需要处理的md文件/"  # 输入目录
CLOUD_API_URL = "https://api.your-cloud-ai.com/v1/chat/completions"  # 替换为实际云端API地址
API_KEY = "your-api-key-here"  # 替换为实际API密钥
MODEL_NAME = "gpt-4-turbo"  # 云端模型名称

# 处理参数
MAX_CHARS_FIRST = 3000  # 首次检测读取字符数
MAX_CHARS_SECOND = 5000  # 二次检测读取字符数
MAX_WORKERS = min(cpu_count(), 8)  # 基于CPU核心数动态调整
TIMEOUT = 300  # 超时时间(秒)
RETRY_LIMIT = 3  # 重试次数
REQUEST_INTERVAL = 0.1  # 请求间隔(秒)
BATCH_SIZE = 100  # 批次大小
PARALLEL_PROCESSING = True  # 启用并行处理
# ===================================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("md_classifier_cloud.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# 生成时间戳
timestamp = time.strftime("%Y%m%d_%H%M%S")

# 输出文件夹结构
GOOD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(INPUT_FOLDER)), f"good_{timestamp}")
TMP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(INPUT_FOLDER)), f"tmp_{timestamp}")
BAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(INPUT_FOLDER)), f"bad_{timestamp}")

os.makedirs(GOOD_FOLDER, exist_ok=True)
os.makedirs(TMP_FOLDER, exist_ok=True)
os.makedirs(BAD_FOLDER, exist_ok=True)

# 严格限定的海上风电领域关键词（2024年最新版）
STRICT_KEYWORDS = {
    # 核心技术
    "海上风电", "风机基础", "单桩基础", "导管架基础", "漂浮式基础", "张力腿平台",
    "海上升压站", "阵列电缆", "送出电缆", "动态电缆", "J型管", "跨接管",
    "防腐系统", "阴极保护", "冲刷防护", "风机吊装", "运维船", "人员转运系统",
    "海上变电站", "无功补偿", "谐波滤波", "黑启动", "SCADA系统", "中央监控",
    
    # 专业技术
    "尾流效应", "风资源评估", "LCOE", "平准化度电成本", "并网技术", 
    "电网适应性", "故障穿越", "海洋水文", "地质勘探", "冲刷分析",
    "风机载荷", "疲劳分析", "极限强度", "腐蚀速率", "防护涂层",
    
    # 英文术语
    "offshore wind", "monopile", "jacket foundation", "floating wind",
    "substation", "inter-array cable", "export cable", "dynamic cable",
    "corrosion protection", "cathodic protection", "scour protection",
    "WTG", "FMEA", "HAZID", "SIL", "LCOE", "CAPEX", "OPEX"
}

# 严格限定的相关领域关键词
RELATED_FIELDS = {
    # 工程领域
    "海洋工程", "海上施工", "海洋地质", "海上气象", "港口工程",
    "船舶工程", "防腐工程", "电力工程", "高电压技术",
    
    # 政策市场
    "可再生能源", "碳交易", "绿证", "电力市场", "竞价上网",
    
    # 英文术语
    "marine engineering", "offshore construction", "renewable energy",
    "carbon trading", "grid connection"
}

# 首次检测提示词（核心领域严格判定）
FIRST_PROMPT_TEMPLATE = """
### 海上风电核心领域严格分类
你是一名海上风电领域专家，请严格评估以下文本是否属于海上风电核心领域。

【核心特征】必须满足至少一项：
1. 包含海上风电特有技术术语（如：单桩基础、动态电缆、LCOE等）
2. 讨论海上风电专有技术、设备或工程问题
3. 包含海上风电项目具体案例（需明确项目名称或技术细节）

【排除条件】出现以下情况应判定为"否"：
1. 仅提及一般风能或陆地风电（未明确海上场景）
2. 仅涉及通用海洋工程（未明确风电应用）
3. 内容过于宽泛（如仅提到"可再生能源"但无具体技术内容）

核心关键词（参考）：
{core_keywords}

待评估文本（前{max_chars}字符）：
{content}

请严格判断后只回复"是"或"否"，不要解释：
"""

# 二次检测提示词（专业相关领域判定）
SECOND_PROMPT_TEMPLATE = """
### 海上风电专业相关领域评估
请评估以下文本是否与海上风电专业相关。

【通过标准】满足任一即可：
1. 包含海上风电相关技术讨论（即使不深入）
2. 提及海上风电项目、政策或市场分析
3. 涉及海上风电相关学科（如海洋工程中的风电应用）

【排除标准】出现以下情况应判定为"否"：
1. 完全不涉及技术、工程或专业内容
2. 内容与能源领域完全无关（如文学、日常等）

相关领域关键词（参考）：
{related_fields}

待评估文本（前{max_chars}字符）：
{content}

请判断后只回复"是"或"否"，不要解释：
"""

# 预编译正则表达式（性能优化）
KEYWORD_PATTERN = re.compile(r'|'.join(map(re.escape, STRICT_KEYWORDS)), re.IGNORECASE)
RELATED_PATTERN = re.compile(r'|'.join(map(re.escape, RELATED_FIELDS)), re.IGNORECASE)

class APIClient:
    """封装云端API调用"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        })
    
    def call_model(self, prompt: str) -> str:
        """调用云端模型API"""
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "你是一名严谨的海上风电领域专家。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 10,
            "response_format": {"type": "text"}
        }
        
        try:
            response = self.session.post(
                CLOUD_API_URL,
                json=payload,
                timeout=TIMEOUT
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {str(e)}")
            raise
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"API响应解析失败: {str(e)}")
            raise

def find_md_files(input_folder: str) -> List[str]:
    """递归查找所有Markdown文件"""
    md_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.md'):
                md_files.append(os.path.join(root, file))
    return sorted(md_files)

def check_filename_relevance(filename: str) -> str:
    """通过文件名初步分类"""
    base_name = os.path.basename(filename).lower()
    if KEYWORD_PATTERN.search(base_name):
        return "core"
    if RELATED_PATTERN.search(base_name):
        return "related"
    return "unrelated"

def classify_text(content: str, prompt_template: str, max_chars: int) -> bool:
    """
    使用云端模型进行文本分类
    返回：True(相关) / False(无关)
    """
    api_client = APIClient()
    prompt = prompt_template.format(
        core_keywords=", ".join(sorted(STRICT_KEYWORDS)),
        related_fields=", ".join(sorted(RELATED_FIELDS)),
        max_chars=max_chars,
        content=content[:max_chars]
    )
    
    for attempt in range(RETRY_LIMIT + 1):
        try:
            time.sleep(REQUEST_INTERVAL * (attempt + 1))
            response = api_client.call_model(prompt).lower()
            
            if "是" in response or "yes" in response:
                return True
            elif "否" in response or "no" in response:
                return False
            else:
                logger.warning(f"模型返回异常响应: {response[:100]}...")
                continue
                
        except Exception as e:
            logger.warning(f"分类尝试 {attempt + 1}/{RETRY_LIMIT} 失败: {str(e)}")
            if attempt == RETRY_LIMIT:
                logger.error("达到最大重试次数，默认保留文件")
                return True  # 失败时默认保留
    
    return True  # 默认保留

def process_first_pass(file_path: str) -> Tuple[str, Optional[str]]:
    """第一阶段处理：核心领域判定"""
    try:
        relative_path = os.path.relpath(file_path, INPUT_FOLDER)
        good_path = os.path.join(GOOD_FOLDER, relative_path)
        bad_path = os.path.join(BAD_FOLDER, relative_path)
        
        # 跳过已处理文件
        if os.path.exists(good_path) or os.path.exists(bad_path):
            return ("skip", None)
        
        # 空文件检测
        if os.path.getsize(file_path) == 0:
            os.makedirs(os.path.dirname(bad_path), exist_ok=True)
            shutil.move(file_path, bad_path)
            return ("empty", None)
        
        # 文件名检测
        filename_relevance = check_filename_relevance(file_path)
        if filename_relevance == "core":
            os.makedirs(os.path.dirname(good_path), exist_ok=True)
            shutil.move(file_path, good_path)
            return ("moved", None)
        
        # 需要内容检测的情况
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(MAX_CHARS_FIRST)
        
        if not content.strip():
            os.makedirs(os.path.dirname(bad_path), exist_ok=True)
            shutil.move(file_path, bad_path)
            return ("empty", None)
        
        # 核心领域分类
        is_core = classify_text(content, FIRST_PROMPT_TEMPLATE, MAX_CHARS_FIRST)
        
        if is_core:
            os.makedirs(os.path.dirname(good_path), exist_ok=True)
            shutil.move(file_path, good_path)
            return ("moved", None)
        else:
            tmp_path = os.path.join(TMP_FOLDER, relative_path)
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            shutil.move(file_path, tmp_path)
            return ("tmp", tmp_path)
            
    except Exception as e:
        logger.error(f"首次处理失败 {file_path}: {str(e)}")
        relative_path = os.path.relpath(file_path, INPUT_FOLDER)
        tmp_path = os.path.join(TMP_FOLDER, relative_path)
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        shutil.move(file_path, tmp_path)
        return ("tmp", tmp_path)

def process_second_pass(file_path: str) -> bool:
    """第二阶段处理：专业相关领域判定"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(MAX_CHARS_SECOND)
        
        if not content.strip():
            relative_path = os.path.relpath(file_path, TMP_FOLDER)
            bad_path = os.path.join(BAD_FOLDER, relative_path)
            os.makedirs(os.path.dirname(bad_path), exist_ok=True)
            shutil.move(file_path, bad_path)
            return False
        
        # 专业相关分类
        is_related = classify_text(content, SECOND_PROMPT_TEMPLATE, MAX_CHARS_SECOND)
        
        relative_path = os.path.relpath(file_path, TMP_FOLDER)
        if is_related:
            good_path = os.path.join(GOOD_FOLDER, relative_path)
            os.makedirs(os.path.dirname(good_path), exist_ok=True)
            shutil.move(file_path, good_path)
            return True
        else:
            bad_path = os.path.join(BAD_FOLDER, relative_path)
            os.makedirs(os.path.dirname(bad_path), exist_ok=True)
            shutil.move(file_path, bad_path)
            return False
            
    except Exception as e:
        logger.error(f"二次处理失败 {file_path}: {str(e)}")
        return False

def cleanup_empty_directories(root_folder: str):
    """清理空目录"""
    for root, dirs, _ in os.walk(root_folder, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except Exception as e:
                logger.warning(f"无法清理目录 {dir_path}: {str(e)}")

def batch_process(files: List[str], process_func, desc: str) -> Dict[str, int]:
    """批量处理文件"""
    stats = {"success": 0, "failed": 0, "skipped": 0}
    
    if not PARALLEL_PROCESSING:
        for file in tqdm(files, desc=desc):
            try:
                result = process_func(file)
                stats["success" if result else "failed"] += 1
            except Exception as e:
                logger.error(f"处理失败 {file}: {str(e)}")
                stats["failed"] += 1
        return stats
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_func, f): f for f in files}
        with tqdm(total=len(files), desc=desc) as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    stats["success" if result else "failed"] += 1
                except Exception as e:
                    logger.error(f"处理失败: {str(e)}")
                    stats["failed"] += 1
                finally:
                    pbar.update(1)
    
    return stats

def main():
    """主控制流程（完整云端版）"""
    # 初始化日志和目录
    logger.info("===== 海上风电文档分类器（云端严格版）=====")
    logger.info(f"输入目录: {INPUT_FOLDER}")
    logger.info(f"输出目录: Good={GOOD_FOLDER} | Temp={TMP_FOLDER} | Bad={BAD_FOLDER}")
    logger.info(f"使用模型: {MODEL_NAME} | 工作线程: {MAX_WORKERS} | 批次大小: {BATCH_SIZE}")

    # === 第一阶段：核心领域判定 ===
    logger.info("\n" + "="*40)
    logger.info("第一阶段：核心领域严格判定")
    logger.info("="*40)
    
    first_pass_files = find_md_files(INPUT_FOLDER)
    total_files = len(first_pass_files)
    
    if total_files == 0:
        logger.info("输入目录中未发现Markdown文件")
    else:
        logger.info(f"开始处理 {total_files} 个文件...")
        
        # 初始化统计
        first_stats = {
            "core_by_name": 0,    # 文件名直接通过
            "core_by_content": 0, # 内容检测通过
            "to_second_pass": 0,  # 待二次检测
            "empty": 0,           # 空文件
            "error": 0            # 错误文件
        }

        # 分批次处理
        total_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx in range(total_batches):
            start = batch_idx * BATCH_SIZE
            end = min((batch_idx + 1) * BATCH_SIZE, total_files)
            batch_files = first_pass_files[start:end]
            
            logger.info(f"\n>> 处理批次 {batch_idx+1}/{total_batches} ({len(batch_files)}个文件)")

            if PARALLEL_PROCESSING:
                # 并行处理
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {executor.submit(process_first_pass, f): f for f in batch_files}
                    batch_results = []
                    with tqdm(total=len(batch_files), desc="进度") as pbar:
                        for future in as_completed(futures):
                            try:
                                result = future.result()
                                batch_results.append(result)
                            except Exception as e:
                                logger.error(f"处理失败: {str(e)}")
                                batch_results.append(("error", None))
                            finally:
                                pbar.update(1)
            else:
                # 串行处理（调试用）
                batch_results = []
                for file in tqdm(batch_files, desc="进度"):
                    try:
                        batch_results.append(process_first_pass(file))
                    except Exception as e:
                        logger.error(f"处理失败 {file}: {str(e)}")
                        batch_results.append(("error", None))

            # 统计本批次结果
            for result, _ in batch_results:
                if result == "moved":
                    first_stats["core_by_name"] += 1
                elif result == "content_passed":
                    first_stats["core_by_content"] += 1
                elif result == "tmp":
                    first_stats["to_second_pass"] += 1
                elif result == "empty":
                    first_stats["empty"] += 1
                else:
                    first_stats["error"] += 1

            logger.info(
                f"本批次结果: "
                f"文件名通过={first_stats['core_by_name']} | "
                f"内容通过={first_stats['core_by_content']} | "
                f"待二次检测={first_stats['to_second_pass']} | "
                f"空文件={first_stats['empty']}"
            )

        # 第一阶段总结
        logger.info("\n" + "="*40)
        logger.info("第一阶段处理完成")
        logger.info(f"直接通过文件: {first_stats['core_by_name'] + first_stats['core_by_content']}")
        logger.info(f"待二次检测文件: {first_stats['to_second_pass']}")
        logger.info(f"已过滤空文件: {first_stats['empty']}")
        logger.info(f"错误文件: {first_stats['error']}")

    # === 第二阶段：专业相关领域判定 ===
    logger.info("\n" + "="*40)
    logger.info("第二阶段：专业相关领域判定")
    logger.info("="*40)
    
    second_pass_files = find_md_files(TMP_FOLDER)
    total_second_files = len(second_pass_files)
    
    if total_second_files == 0:
        logger.info("没有需要二次检测的文件")
    else:
        logger.info(f"开始二次检测 {total_second_files} 个文件...")
        
        # 初始化统计
        second_stats = {
            "passed": 0,    # 通过
            "rejected": 0,  # 拒绝
            "error": 0      # 错误
        }

        # 分批次处理
        total_second_batches = (total_second_files + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx in range(total_second_batches):
            start = batch_idx * BATCH_SIZE
            end = min((batch_idx + 1) * BATCH_SIZE, total_second_files)
            batch_files = second_pass_files[start:end]
            
            logger.info(f"\n>> 二次检测批次 {batch_idx+1}/{total_second_batches}")

            if PARALLEL_PROCESSING:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {executor.submit(process_second_pass, f): f for f in batch_files}
                    with tqdm(total=len(batch_files), desc="进度") as pbar:
                        for future in as_completed(futures):
                            try:
                                if future.result():
                                    second_stats["passed"] += 1
                                else:
                                    second_stats["rejected"] += 1
                            except Exception as e:
                                logger.error(f"处理失败: {str(e)}")
                                second_stats["error"] += 1
                            finally:
                                pbar.update(1)
            else:
                for file in tqdm(batch_files, desc="进度"):
                    try:
                        if process_second_pass(file):
                            second_stats["passed"] += 1
                        else:
                            second_stats["rejected"] += 1
                    except Exception as e:
                        logger.error(f"处理失败 {file}: {str(e)}")
                        second_stats["error"] += 1

            logger.info(
                f"本批次结果: "
                f"通过={second_stats['passed']} | "
                f"拒绝={second_stats['rejected']} | "
                f"错误={second_stats['error']}"
            )

        # 第二阶段总结
        logger.info("\n" + "="*40)
        logger.info("第二阶段处理完成")
        logger.info(f"最终通过文件: {second_stats['passed']}")
        logger.info(f"最终拒绝文件: {second_stats['rejected']}")
        logger.info(f"错误文件: {second_stats['error']}")

    # === 最终清理和统计 ===
    logger.info("\n" + "="*40)
    logger.info("执行清理操作")
    logger.info("="*40)
    
    cleanup_empty_directories(INPUT_FOLDER)
    cleanup_empty_directories(TMP_FOLDER)
    
    # 最终报告
    logger.info("\n" + "="*60)
    logger.info("处理完成！最终统计")
    logger.info(f"总处理文件数: {total_files}")
    logger.info(f"最终通过文件: {first_stats.get('core_by_name',0) + first_stats.get('core_by_content',0) + second_stats.get('passed',0)}")
    logger.info(f"最终拒绝文件: {first_stats.get('empty',0) + second_stats.get('rejected',0)}")
    logger.info(f"临时保留文件: {first_stats.get('to_second_pass',0) + second_stats.get('error',0)}")
    logger.info(f"通过文件存放位置: {GOOD_FOLDER}")
    logger.info(f"拒绝文件存放位置: {BAD_FOLDER}")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("程序被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序异常终止: {str(e)}", exc_info=True)
        sys.exit(1)
