"""
Listwise 排序模型的 RL 训练 Reward 函数

使用 Qwen3 72B 对排序后的网页结果进行打分 (满分20分)。

打分维度 (按重要性排序):
1. 相关性 (0-10分):  top5 / top10 与 query 的匹配程度  —— 最重要
2. 权威性 (0-4分):  头部结果来源站点的权威程度
3. 时效性 (0-3分):  网页发布时间与当前时间的匹配度
4. 多样性 (0-3分):  网页站点来源的分布多样性  —— 最次要

依赖: pip install openai
"""

import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class WebPage:
    """网页信息"""
    idx: int                             # 原始索引
    title: str
    snippet: str = ""
    url: str = ""
    domain: str = ""                     # 站点域名 (用于权威性)
    publish_time: str = ""               # 发布时间 (用于时效性, 如 "2025-03-15")


@dataclass
class RewardResult:
    """Reward 计算结果"""
    total_score: float                   # 总分 0-20
    relevance_score: float               # 相关性 0-10
    authority_score: float               # 权威性 0-4
    timeliness_score: float              # 时效性 0-3
    diversity_score: float               # 多样性 0-3
    reasoning: str                       # 打分理由概述


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

SCORING_SYSTEM_PROMPT = """你是一个搜索排序质量评估专家。你需要对一组已排序的搜索结果进行综合评分。请严格遵循评分标准，客观公正地打分。

## 评分维度

### 1. 相关性 (0-10分) —— 最重要
这是最重要的维度，关注排序后前5条(top5)和前10条(top10)结果与用户查询的相关程度：
- 9-10分: top5 全部高度相关，完美匹配 query 意图；top10 也几乎全部相关
- 7-8分:  top5 大部分相关，top10 中 1-2 条部分相关或主题偏离
- 5-6分:  top5 中约一半相关；top10 中混入了不少不相关结果
- 3-4分:  top10 中只有少数结果相关，大量无关内容排在靠前位置
- 0-2分:  top10 几乎都不相关，排序完全失败

评估要点：
- 网页摘要内容是否直接回答了 query 的问题
- 标题是否与 query 的核心意图一致
- query 有明确答案时，top1 是否为最直接的回答
- 注意区分"主题相关但内容不满足需求"和"完全无关"

### 2. 权威性 (0-4分)
关注前10条结果的来源站点是否权威：
- 4分: 头部结果来自知名权威站点 (如政府 .gov、学术 .edu、官方机构、行业公认权威媒体)
- 3分: 大部分来自较有公信力的站点，无低质来源
- 2分: 混合来源，有权威也有普通站点，整体可接受
- 1分: 多为不知名站点或个人来源
- 0分: 大量来自不可靠来源、农场内容、垃圾站点

权威站点参考：
- 政府/官方: 政府网站、官方公告平台
- 学术/机构: 高校、研究院、学术期刊
- 垂直权威: 医疗(卫健委、三甲医院)、法律(法院、律协)、科技(官方文档、知名技术媒体)
- 知名媒体: 新华社、人民日报、央视等正规新闻机构

### 3. 时效性 (0-3分)
结合当前时间和网页发布时间，判断头部结果的时效性：
- 3分: 所有头部结果的发布时间非常适合 query (如 query 涉及最新资讯，结果均为近期发布)
- 2分: 大部分结果时效性可接受，少数稍显陈旧
- 1分: 部分结果发布时间明显过时
- 0分: 头部结果几乎全部过时或发布时间缺失

注意：
- 并非所有 query 都要求高时效性 (如"唐诗三百首"不要求时效)
- 对于新闻、政策、价格、天气等 query，时效性要求高
- 对于百科、历史、原理类 query，时效性权重应降低
- 发布时间缺失时，结合内容判断是否仍在合理范围内

### 4. 多样性 (0-3分) —— 最次要
关注前10条结果的站点来源分布：
- 3分: 来自 7 个以上不同站点，且内容角度多样
- 2分: 来自 4-6 个不同站点，有一定多样性
- 1分: 来自 2-3 个站点，存在同站聚集
- 0分: 几乎都来自同一站点，同质化严重

注意：
- 如果 query 天然需要多角度信息 (如对比、评测、攻略)，多样性应更受重视
- 如果 query 有唯一权威来源 (如官方查询)，多样性不应作为主要考量

## 输出格式

严格按以下 JSON 格式输出，不要输出其他任何内容：
```json
{
  "relevance_score": 8.0,
  "authority_score": 3.0,
  "timeliness_score": 2.5,
  "diversity_score": 2.0,
  "total_score": 15.5,
  "reasoning": "相关性: xxx; 权威性: xxx; 时效性: xxx; 多样性: xxx"
}
```

- 所有分数类型为数字 (可带小数)
- total_score = relevance_score + authority_score + timeliness_score + diversity_score
- 总分区间 0-20
- reasoning 简明扼要，每项1-2句话
"""


def build_user_prompt(
    query: str,
    ranked_pages: List[WebPage],
    current_time: str,
    max_pages: int = 20,
) -> str:
    """构造 User Prompt"""
    lines = [
        f"## 用户查询\n\n{query}\n",
        f"## 当前时间\n\n{current_time}\n",
        "## 排序后的搜索结果\n",
    ]

    for rank, page in enumerate(ranked_pages[:max_pages]):
        position_tag = ""
        if rank == 0:
            position_tag = " ← Top1"
        elif rank < 5:
            position_tag = f" ← Top{rank + 1}"
        elif rank < 10:
            position_tag = f" ← Top{rank + 1}"
        elif rank == 9:
            position_tag = " ← Top10"

        lines.append(f"### [{rank + 1}]{position_tag}")
        lines.append(f"- 标题: {page.title}")
        if page.snippet:
            lines.append(f"- 摘要: {page.snippet[:300]}")
        if page.domain:
            lines.append(f"- 来源站点: {page.domain}")
        if page.publish_time:
            lines.append(f"- 发布时间: {page.publish_time}")
        lines.append("")

    if len(ranked_pages) > max_pages:
        lines.append(f"... 共 {len(ranked_pages)} 条结果，以上仅展示前 {max_pages} 条")

    lines.append("请按照评分标准，对以上排序结果进行综合打分。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Qwen3 72B 客户端
# ---------------------------------------------------------------------------

class QwenClient:
    """Qwen3 72B API 客户端 (兼容 OpenAI 接口)"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "qwen3-72b",
        temperature: float = 0.0,        # RL reward 需要确定性打分
        max_tokens: int = 1024,
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """发送请求, 含指数退避重试"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Qwen API 请求失败 (重试{self.max_retries}次): {last_error}")


# ---------------------------------------------------------------------------
# JSON 解析工具
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """从模型输出中提取 JSON"""
    text = text.strip()
    # 1. 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. 提取 ```json ... ``` 代码块
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3. 提取首个 {...} 块
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"无法解析 JSON: {text[:300]}")


# ---------------------------------------------------------------------------
# Reward 函数
# ---------------------------------------------------------------------------

class RankingReward:
    """
    Listwise 排序模型的 RL Reward 计算器

    用法:
        reward_fn = RankingReward(client)
        result = reward_fn.compute(ranked_indices, webpages, query, current_time)
        reward = result.total_score  # 用于 RL 训练的标量 reward
    """

    MAX_PAGES_IN_PROMPT = 20

    def __init__(self, client: QwenClient):
        self.client = client

    def compute(
        self,
        ranked_indices: List[int],    # 模型输出的排序后 idx 列表
        webpages: List[WebPage],      # 原始网页列表
        query: str,                   # 查询
        current_time: str = "",       # 当前时间 (空则用系统时间)
    ) -> RewardResult:
        """
        计算 reward

        参数:
            ranked_indices: 排序模型输出的网页索引列表, 如 [3, 0, 5, 1, 2, 4]
            webpages:       原始网页列表 (含元数据)
            query:          用户查询
            current_time:   当前时间字符串, 如 "2025-06-02"

        返回:
            RewardResult: 含总分和各维度子分
        """
        if not ranked_indices:
            return RewardResult(
                total_score=0.0, relevance_score=0.0,
                authority_score=0.0, timeliness_score=0.0,
                diversity_score=0.0, reasoning="排序列表为空",
            )

        # 按 ranked_indices 重排网页
        id_to_page = {p.idx: p for p in webpages}
        ranked_pages = []
        for idx in ranked_indices:
            page = id_to_page.get(idx)
            if page is None:
                raise ValueError(f"ranked_indices 中包含不存在的网页索引: {idx}")
            ranked_pages.append(page)

        if not current_time:
            current_time = time.strftime("%Y-%m-%d")

        # 构造 prompt 并请求模型打分
        user_prompt = build_user_prompt(
            query, ranked_pages, current_time, self.MAX_PAGES_IN_PROMPT,
        )
        raw = self.client.chat(SCORING_SYSTEM_PROMPT, user_prompt)
        data = _extract_json(raw)

        return self._parse_response(data)

    @staticmethod
    def _parse_response(data: dict) -> RewardResult:
        rel  = float(data.get("relevance_score", 0))
        auth = float(data.get("authority_score", 0))
        time_s = float(data.get("timeliness_score", 0))
        div  = float(data.get("diversity_score", 0))
        total = float(data.get("total_score", rel + auth + time_s + div))

        # clamp 到合法范围
        rel  = max(0.0, min(10.0, rel))
        auth = max(0.0, min(4.0, auth))
        time_s = max(0.0, min(3.0, time_s))
        div  = max(0.0, min(3.0, div))
        total = max(0.0, min(20.0, total))

        return RewardResult(
            total_score=round(total, 2),
            relevance_score=round(rel, 2),
            authority_score=round(auth, 2),
            timeliness_score=round(time_s, 2),
            diversity_score=round(div, 2),
            reasoning=str(data.get("reasoning", "")),
        )


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def make_ranking_reward_fn(
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    model: str = "qwen3-72b",
) -> RankingReward:
    """工厂函数: 创建一个即用的 RankingReward 实例"""
    client = QwenClient(base_url=base_url, api_key=api_key, model=model)
    return RankingReward(client)


# ---------------------------------------------------------------------------
# Mock 客户端 (离线测试)
# ---------------------------------------------------------------------------

class MockQwenClient(QwenClient):
    """模拟 Qwen 客户端, 用启发式规则近似打分，离线测试用"""

    def __init__(self):
        pass

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(self._heuristic_score(user_prompt), ensure_ascii=False)

    @staticmethod
    def _heuristic_score(user_prompt: str) -> dict:
        # 提取 query
        m = re.search(r'用户查询\s*\n+\s*(.+?)\n', user_prompt)
        query = m.group(1).strip() if m else ""

        # 解析每条结果的标题和摘要
        entries = re.split(r'\n(?=### \[\d+\])', user_prompt)

        scores_per_result = []
        domains_seen = set()
        query_chars = set(query)

        for entry in entries[1:]:  # 跳过 header
            # 相关性: 字面重叠
            overlap = len(query_chars & set(entry)) / max(1, len(query_chars))
            rel = min(1.0, overlap * 1.3 + 0.05)

            # 权威性: 检查域名关键词
            domain_match = re.search(r'来源站点:\s*(\S+)', entry)
            domain = domain_match.group(1) if domain_match else ""
            domains_seen.add(domain)
            auth_signals = [".gov", ".edu", "ac.cn", "新华社", "人民日报", "央视",
                           "卫计委", "中科院", "官方", "github.com", "wikipedia"]
            auth = 0.7 if any(s in domain for s in auth_signals) else 0.3

            # 时效性: 发布时间
            time_match = re.search(r'发布时间:\s*(\S+)', entry)
            has_time = bool(time_match)
            if has_time:
                ts = 0.8 if "2025" in time_match.group(1) or "2026" in time_match.group(1) else 0.4
            else:
                ts = 0.3

            scores_per_result.append({"rel": rel, "auth": auth, "ts": ts})

        n = max(1, len(scores_per_result))

        # 相关性: top5 / top10 加权
        top5_rel = sum(r["rel"] for r in scores_per_result[:5]) / min(5, n)
        top10_rel = sum(r["rel"] for r in scores_per_result[:10]) / min(10, n)
        relevance = top5_rel * 6.0 + top10_rel * 4.0
        relevance = min(10.0, relevance * 1.1)

        # 权威性: top10 平均
        top_auth = sum(r["auth"] for r in scores_per_result[:10]) / min(10, n)
        authority = top_auth * 4.0

        # 时效性
        top_ts = sum(r["ts"] for r in scores_per_result[:10]) / min(10, n)
        # query 有时效关键词则权重更高
        time_kw = ["最新", "今天", "实时", "最近", "新闻", "当前"]
        ts_weight = 1.5 if any(kw in query for kw in time_kw) else 1.0
        timeliness = top_ts * 3.0 * ts_weight
        timeliness = min(3.0, timeliness)

        # 多样性: 站点去重
        n_domains = len(domains_seen)
        if n_domains >= 7:
            diversity = 3.0
        elif n_domains >= 4:
            diversity = 2.0
        elif n_domains >= 2:
            diversity = 1.0
        else:
            diversity = 0.0

        total = round(relevance + authority + timeliness + diversity, 2)

        return {
            "relevance_score": round(relevance, 2),
            "authority_score": round(authority, 2),
            "timeliness_score": round(timeliness, 2),
            "diversity_score": round(diversity, 2),
            "total_score": total,
            "reasoning": (
                f"相关性: top5={top5_rel:.2f} top10={top10_rel:.2f} → {relevance:.1f}; "
                f"权威性: top10平均={top_auth:.2f} → {authority:.1f}; "
                f"时效性: top10平均={top_ts:.2f} → {timeliness:.1f}; "
                f"多样性: {n_domains}个不同站点 → {diversity:.1f}"
            ),
        }


# ---------------------------------------------------------------------------
# 示例 & 测试
# ---------------------------------------------------------------------------

def main():
    query = "2025年诺贝尔物理学奖获得者是谁"

    webpages = [
        WebPage(idx=0, title="2025诺贝尔物理学奖揭晓",
                snippet="瑞典皇家科学院宣布，2025年诺贝尔物理学奖授予...",
                domain="xinhuanet.com", publish_time="2025-10-08"),
        WebPage(idx=1, title="诺贝尔奖历年获得者名单",
                snippet="诺贝尔物理学奖自1901年颁发以来...",
                domain="wikipedia.org", publish_time="2025-01-15"),
        WebPage(idx=2, title="物理学最新研究进展",
                snippet="2025年物理学领域取得多项突破...",
                domain="cass.cn", publish_time="2025-09-20"),
        WebPage(idx=3, title="如何学好高中物理",
                snippet="高中物理学习方法分享，从力学到电磁学...",
                domain="zhihu.com", publish_time="2024-06-01"),
        WebPage(idx=4, title="诺贝尔奖趣闻",
                snippet="关于诺贝尔奖的一些有趣故事...",
                domain="sohu.com", publish_time="2025-10-10"),
        WebPage(idx=5, title="物理学入门基础知识",
                snippet="物理学是研究物质运动规律的科学...",
                domain="baidu.com", publish_time="2023-03-10"),
        WebPage(idx=6, title="2025诺贝尔奖全部获奖名单",
                snippet="2025年诺贝尔各奖项获奖者完整名单...",
                domain="people.com.cn", publish_time="2025-10-08"),
        WebPage(idx=7, title="诺贝尔物理学奖幕后故事",
                snippet="揭秘2025年诺贝尔物理学奖得主的研究历程...",
                domain="thepaper.cn", publish_time="2025-10-09"),
        WebPage(idx=8, title="中小学物理实验教程",
                snippet="趣味物理实验，适合中小学生动手操作...",
                domain="pep.com.cn", publish_time="2022-08-01"),
        WebPage(idx=9, title="什么是诺贝尔奖",
                snippet="诺贝尔奖是根据瑞典化学家诺贝尔遗嘱设立的奖项...",
                domain="wikipedia.org", publish_time="2025-05-20"),
    ]

    # ---- 好的排序: 相关结果靠前 ----
    good_ranking = [0, 6, 7, 1, 4, 2, 9, 3, 5, 8]

    # ---- 差的排序: 无关结果靠前 ----
    bad_ranking = [3, 5, 8, 9, 4, 1, 2, 6, 7, 0]

    print("=" * 70)
    print("测试 RankingReward (使用 Mock 客户端)")
    print("=" * 70)

    mock = MockQwenClient()
    reward_fn = RankingReward(mock)

    for label, ranking in [("好的排序", good_ranking), ("差的排序", bad_ranking)]:
        result = reward_fn.compute(ranking, webpages, query, current_time="2025-10-12")
        print(f"\n--- {label} ---")
        print(f"  排序: {ranking}")
        print(f"  总分:       {result.total_score:5.1f} / 20")
        print(f"  相关性:     {result.relevance_score:5.1f} / 10")
        print(f"  权威性:     {result.authority_score:5.1f} / 4")
        print(f"  时效性:     {result.timeliness_score:5.1f} / 3")
        print(f"  多样性:     {result.diversity_score:5.1f} / 3")
        print(f"  理由: {result.reasoning[:120]}...")


if __name__ == "__main__":
    main()
