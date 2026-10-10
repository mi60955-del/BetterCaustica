#!/usr/bin/env python3
"""抓取 Minecraft 挂起时的线程栈（退出无响应时的第一现场）。

用法（游戏卡住、还没关进程时运行）：
    python tools/dump_threads.py

自动找到游戏进程（命令行含 KnotClient），依次尝试 jstack / jcmd 抓全线程栈，
落盘到 logs/threaddump-<时间>.txt。进程被杀之后就来不及了——卡住时立刻跑。
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PS = "powershell"
OUT_DIR = Path(__file__).resolve().parent.parent / "logs"


def find_tool(name):
    """按 PATH、JAVA_HOME/bin 的顺序找 JDK 命令行工具，找不到返回 None。"""
    exe = shutil.which(name)
    if exe:
        return Path(exe)
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        binary = f"{name}.exe" if os.name == "nt" else name
        candidate = Path(java_home) / "bin" / binary
        if candidate.exists():
            return candidate
    return None


def find_game_pid():
    """返回游戏进程 PID；按命令行里的 KnotClient 匹配，避免误伤 Gradle daemon。"""
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='javaw.exe' or Name='java.exe'\" "
        "| Where-Object { $_.CommandLine -match 'KnotClient' } "
        "| Select-Object -ExpandProperty ProcessId"
    )
    r = subprocess.run([PS, "-NoProfile", "-Command", script],
                       capture_output=True, text=True, errors="replace")
    pids = [int(x) for x in r.stdout.split() if x.strip().isdigit()]
    return pids


def run_tool(exe, pid):
    if exe is None or not exe.exists():
        return None
    r = subprocess.run([str(exe), str(pid)], capture_output=True, text=True,
                       errors="replace", timeout=60)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 0 and r.stdout.strip():
        return out
    return f"[{exe.name} rc={r.returncode}]\n{out}" if out else None


def main():
    jstack = find_tool("jstack")
    jcmd = find_tool("jcmd")
    if jstack is None and jcmd is None:
        print("没找到 jstack/jcmd：把 JDK 的 bin 放进 PATH，或设置 JAVA_HOME。")
        return 1
    pids = find_game_pid()
    if not pids:
        print("没找到游戏进程（命令行含 KnotClient 的 java/javaw）。"
              "如果游戏已经被关掉/杀掉，第一现场就丢了。")
        return 1
    if len(pids) > 1:
        print(f"多个候选进程: {pids}，全抓。")

    OUT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for pid in pids:
        out = OUT_DIR / f"threaddump-{stamp}-{pid}.txt"
        parts = [f"# thread dump @ {time.strftime('%Y-%m-%d %H:%M:%S')} pid={pid}\n"]
        dump = run_tool(jstack, pid)
        if not dump or "Unable to attach" in dump:
            alt = run_tool(jcmd, pid)
            if alt:
                dump = f"--- jstack 失败，改用 jcmd Thread.print ---\n{alt}"
        parts.append(dump or "（jstack 与 jcmd 都没有产出——进程可能已僵死到无法附加，"
                             "立即改用任务管理器右键该进程 > 创建转储文件）")
        out.write_text("\n".join(parts), encoding="utf-8")
        print(f"已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
