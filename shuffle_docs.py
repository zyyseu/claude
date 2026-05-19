import random
from typing import List


def shuffle_docs(strings: List[str]) -> List[str]:
    """
    输入: 大小为 m 的字符串数组，每个字符串格式为:
        prefix\n\n#所有网页标题\n doc_line0 \n ... doc_linen \n\npostfix

    将 prefix/postfix 和 doc_lines 随机重组后，返回大小为 m 的新数组。
    """
    m = len(strings)
    if m == 0:
        return []

    prefixes = []
    postfixes = []
    all_doc_lines = []          # 所有 doc_line 的大池子，共 m*n 条
    doc_counts = []             # 每条原始字符串有多少个 doc_line

    for s in strings:
        parts = s.split("\n\n#所有网页标题\n", 1)
        head = parts[0]          # prefix
        tail = parts[1] if len(parts) > 1 else ""

        # 拆分尾部: | doc_line0 \n ... doc_linen \n\npostfix
        lines_part, postfix = tail.rsplit("\n\n", 1)
        doc_lines = [line.strip() for line in lines_part.strip().split("\n")]

        prefixes.append(head)
        postfixes.append(postfix)
        all_doc_lines.extend(doc_lines)
        doc_counts.append(len(doc_lines))

    n = doc_counts[0] if doc_counts else 0
    result: List[str] = []

    for i in range(m):
        p = random.choice(prefixes)
        q = random.choice(postfixes)
        sampled_docs = random.sample(all_doc_lines, n)

        doc_block = "\n".join(sampled_docs)
        new_s = f"{p}\n\n#所有网页标题\n{doc_block}\n\n{q}"
        result.append(new_s)

    return result
