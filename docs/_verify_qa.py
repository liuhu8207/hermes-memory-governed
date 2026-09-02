# -*- coding: utf-8 -*-
"""临时验证脚本：复核 QA（严过关）提出的 3 条关键缺陷。验证后即删除。"""
import sys, sqlite3, time, tempfile, threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugin.memory_governed._config import load_governed_config
from plugin.memory_governed._sync import WriteQueue, L3Writer
from plugin.memory_governed._recall import RecallEngine

home = Path(tempfile.mkdtemp())
(home / "memory" / "l3").mkdir(parents=True, exist_ok=True)
cfg = load_governed_config(str(home))

print("=== A: rowid_map 键 vs fact 键（QA: source_rowid 恒为 None）===")
w = L3Writer(cfg)
msgs = [
    {"role": "user", "content": "我今天去了北京。天气很不错。我们决定用 Postgres 而不是 MySQL。"},
    {"role": "assistant", "content": "好的。我记住了你们的技术选型。数据库迁移下周一完成。"},
]
rmap = w.write(msgs, "sess1") or {}
print("  rowid_map (key=整条消息):")
for k, v in rmap.items():
    print(f"    rowid={v}  key={k[:30]!r}...")
q = WriteQueue(cfg)
facts = q._extract_atomic_facts(msgs)
print(f"  facts (key=句子片段), 共 {len(facts)} 条:")
for f in facts:
    print(f"    {f['content'][:30]!r}")
hits = sum(1 for f in facts if rmap.get(f["content"]) or rmap.get(f["content"][:60]))
print(f"  >>> source_rowid 可解析: {hits}/{len(facts)}")

print()
print("=== B: 纯 CJK 多词查询（QA: 空格被拼接 → 0 结果）===")
db = cfg.l3_db_path
conn = sqlite3.connect(db)
conn.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
             "session_id TEXT, role TEXT, content TEXT, timestamp REAL, metadata TEXT, hash TEXT)")
now = time.time()
for i, c in enumerate(["今天天气不错，北京的空气质量很好",
                       "我们讨论了数据库选型",
                       "明天的会议改到下午三点"]):
    conn.execute("INSERT INTO messages (session_id,role,content,timestamp,hash) VALUES (?,?,?,?,?)",
                 ("s", "user", c, now, str(i)))
conn.commit()
conn.close()
re_ = RecallEngine(cfg)
for qy in ["天气 北京", "weather 天气 北京", "天气", "北京", "会议 下午"]:
    r = re_._search_l3(qy)
    print(f"  query={qy!r:24} -> {len(r)} 条")

print()
print("=== C: 线程 identity 差分法复核（QA 指出 active_count 不可靠）===")
from plugin.memory_governed import GovernedMemoryProvider

p = GovernedMemoryProvider()
p.initialize("s1", hermes_home=str(home))
before = {t.ident for t in threading.enumerate()}
p._recall.parallel_recall = lambda x: (time.sleep(0.6), [])[1]
for i in range(20):
    p.queue_prefetch(f"q{i}")
time.sleep(0.05)
after = {t.ident for t in threading.enumerate()}
print(f"  identity 差分新增线程 = {len(after - before)} (20 次调用)")
print(f"  threading.active_count() = {threading.active_count()}")
p.shutdown()

print()
print("=== D: WriteQueue.__init__ 是否定义 _embed_fn（QA P1）===")
qq = WriteQueue(cfg)
print(f"  hasattr(_embed_fn) = {hasattr(qq, '_embed_fn')}")
print(f"  hasattr(_embed_model) = {hasattr(qq, '_embed_model')}")
print(f"  -> _init_l2 早退时 _index_l2:145 访问 _embed_fn 会抛 AttributeError")
