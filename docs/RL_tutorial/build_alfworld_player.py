"""Build the reusable, offline ALFWorld player from the checked-in trajectories.

The scene geometry is schematic. Actions and observations are copied verbatim
(except chat serialization tokens). Translations/scene annotations live in JS.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
SAMPLES = ROOT.parent.parent / 'scripts/rl_tutorial/sample_data'


def player_assets():
    data = {}
    for key, name in [('success', 'traj_success_late.json'), ('fail', 'traj_fail_early.json')]:
        record = json.loads((SAMPLES / name).read_text())
        data[key] = {
            'source': name, 'eid': record['eid'], 'reward': record['reward'],
            'success': record['success'], 'instruction': record['instruction'],
            'steps': [dict(action=s['action'], observation=re.split(r'<\|im_end\|>', s['observation'])[0].strip()) for s in record['steps']],
        }
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    css = (ASSETS / 'alfworld-player.css').read_text()
    js = (ASSETS / 'alfworld-player.js').read_text()
    return '<style>' + css + '</style><script>window.ALFWORLD_TRAJECTORIES=' + payload + ';</script><script>' + js + '</script>'


def build_standalone():
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>小机器人找蜡烛 · ALFWorld 任务动画</title><style>body{margin:0;background:#edf3f2;padding:24px;font-family:system-ui}main{max-width:1080px;margin:auto}a{color:#087f82}header{font-size:13px;color:#617585;display:flex;justify-content:space-between;gap:12px} @media(max-width:600px){body{padding:10px}header{padding:5px}}</style></head><body><main><header><span>RL TUTORIAL / 可交互示例</span><a href="./ch2_rollout与experience.html">回到第 2 章</a></header><div class="alf-player" data-mode="intro"></div><noscript>请启用 JavaScript 播放动画；完整文字轨迹见第二章。</noscript></main>'''
    (ROOT / 'alfworld_robot.html').write_text(page + player_assets() + '</body></html>', encoding='utf-8')


if __name__ == '__main__':
    build_standalone()
