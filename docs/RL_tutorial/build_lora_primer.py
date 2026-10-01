"""Offline LoRA diagrams: SVG fallbacks for Markdown, responsive HTML for the site."""
from html import escape
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'


def frame(number, title, body, foot, extra=''):
    return f'''<figure class="lora-figure" {extra}>
<div class="lora-head"><div class="lora-kicker">LORA · 图解 {number}</div><div class="lora-title">{title}</div></div>
<div class="lora-body">{body}</div><figcaption class="lora-foot">{foot}</figcaption></figure>'''


def motivation():
    tasks = ('客服', '代码', 'ALFWorld')
    full = ''.join(f'<div class="lora-task-model"><strong>{task}</strong><span>完整模型权重</span></div>' for task in tasks)
    adapters = ''.join(f'<div class="lora-task-adapter"><strong>{task}</strong><span>adapter</span></div>' for task in tasks)
    return frame('01', '新增一个任务，保存一份小 adapter', f'''
<div class="lora-task-comparison">
 <div class="lora-task-option"><div class="lora-option-title">全参数微调</div><div class="lora-task-list">{full}</div><div class="lora-task-caption">每个任务维护一份完整权重</div></div>
 <div class="lora-task-option shared"><div class="lora-option-title">LoRA 微调</div><div class="lora-adapter-list">{adapters}</div><div class="lora-sharing-arrows" aria-hidden="true">↑　　↑　　↑</div><div class="lora-base-model"><strong>一份共享的基础模型</strong><span>按任务配合不同 adapter 使用</span></div><div class="lora-task-caption">小 adapter 易于保存、分发与切换</div></div>
</div>''', '示意同一基础模型的三个任务版本。adapter 需配套基础模型；图块大小不代表显存比例，同时服务多少任务取决于后端和资源。')


def flow():
    return frame('02', '原权重不动，只学一条小分支', '''
<div class="lora-full"><span>全参数</span><span class="lora-math">x →</span><span class="lora-weight">W 全部可训练</span><span class="lora-math">→ y</span><span class="lora-note">优化器直接更新原权重</span></div>
<div class="lora-split"><div class="lora-input">LoRA：同一个输入 x</div>
<div class="lora-branches">
 <div class="lora-path"><span class="lora-arrow" aria-hidden="true">↓</span><span class="lora-pill frozen">冻结 · 不更新</span><strong>基础权重 W</strong><span class="lora-math">Wx</span><span class="lora-note">保留原有计算</span><span class="lora-arrow" aria-hidden="true">↓</span></div>
 <div class="lora-path adapter"><span class="lora-arrow" aria-hidden="true">↓</span><span class="lora-pill">可训练 · 只更新 A、B</span><div class="lora-pair"><span class="lora-small-matrix">A</span><span class="lora-arrow" aria-hidden="true">→</span><span class="lora-small-matrix">B</span></div><span class="lora-math">s · B(Ax)</span><span class="lora-note">学习任务所需的增量</span><span class="lora-arrow" aria-hidden="true">↓</span></div>
</div><div class="lora-sum"><span class="lora-plus" aria-label="两路逐元素相加">+</span><span class="lora-output lora-math">y = Wx + s · B(Ax)</span></div></div>''',
    '图示一层线性变换。s 是缩放系数，常用 α/r。冻结指不更新 W；基础模型仍参与前向与梯度传播。')


def rank():
    buttons = ''.join(f'<button type="button" data-rank="{r}" aria-pressed="{str(r == 32).lower()}">{r}</button>' for r in (8, 16, 32, 64, 128))
    return frame('03', 'rank 越小，需要学习的参数越少', f'''
<div class="lora-rank-buttons" role="group" aria-label="选择示意层的 LoRA rank"><span class="lora-rank-label">切换 rank</span>{buttons}</div>
<div class="lora-shapes" aria-hidden="true">
 <div><div class="lora-shape-slot"><div class="lora-matrix"></div></div><div class="lora-shape-label">完整增量 ΔW<br>4096 × 4096</div></div>
 <span class="lora-arrow">→</span>
 <div><div class="lora-shape-slot"><div class="lora-matrix thin b"></div><span>×</span><div class="lora-matrix thin a"></div></div><div class="lora-shape-label">用 B × A 表示增量<br>4096 × <span data-rank-value>32</span> · <span data-rank-value>32</span> × 4096</div></div>
</div>
<div class="lora-counts"><div class="lora-count"><span>直接训练一个大矩阵</span><span class="lora-number">16,777,216</span><span class="lora-formula">4096 × 4096 个参数</span></div><div class="lora-count trainable"><span>只训练 A 和 B</span><output class="lora-number" data-parameter-count aria-label="LoRA 可训练参数量">262,144</output><span class="lora-formula">4096 × <span data-rank-value>32</span> + <span data-rank-value>32</span> × 4096</span></div></div>
<p class="lora-ratio" aria-live="polite">本层可训练参数为原来的 <strong data-parameter-ratio>1.56%</strong>。</p>''',
    '用 4096 × 4096 的单层作算术示例；形状非等比例。它不是 Qwen3-1.7B 全模型的统计，也不预测显存、速度或效果。', 'data-lora-rank-demo')


def sharing():
    return frame('04', 'TuFT：多个任务，共享训练与采样后端', '''
<img class="lora-overview" src="./assets/tuft_multi_tenant_overview.png" alt="TuFT 主图：多个客户端向共享 GPU 后端发送前向反向、优化器更新、存档和采样请求" width="4000" height="2250">
<div class="lora-counts"><div class="lora-count"><span class="lora-pill frozen">共享</span><p><strong>冻结的基础权重<br>GPU 计算资源与服务</strong></p></div><div class="lora-count trainable"><span class="lora-pill">各任务独立</span><p><strong>LoRA adapter<br>优化器状态与 checkpoint</strong></p></div></div>
<div class="lora-shared-result">多个客户端提交请求，TuFT 统一调度；<br>同一基础模型上的任务可以共用基础权重，分别训练自己的 adapter。</div>''',
    '主图来源：<a href="https://github.com/agentscope-ai/TuFT">TuFT 项目</a>。不同基础模型可以由平台管理，但不共享同一份权重；并发容量与执行方式取决于后端配置和资源。')


FIGURES = {'ch6_lora_tasks.svg': motivation, 'ch6_lora_flow.svg': flow, 'ch6_lora_rank.svg': rank, 'assets/tuft_multi_tenant_overview.png': sharing}


def enhance_lora_figures(html):
    for name, render in FIGURES.items():
        pattern = r'<p><img alt="[^"]*" src="\./' + re.escape(name) + r'"\s*/?></p>'
        html = re.sub(pattern, lambda _: render(), html)
    return html


def primer_assets():
    return '<style>' + (ASSETS / 'lora-primer.css').read_text() + '</style><script>' + (ASSETS / 'lora-primer.js').read_text() + '</script>'


def build_fallbacks():
    """Plain vector figures stay readable on GitHub without styles or scripts."""
    def text(x, y, label, size=22, fill='#243648', weight=400):
        return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}">{escape(label)}</text>'
    def box(x, y, w, h, label, fill='#eff9f5', stroke='#b6dcd1'):
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}"/>' + text(x+18, y+h/2+8, label, 24)
    def save(name, title, content, height):
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 {height}" role="img" aria-label="{escape(title)}" font-family="system-ui, PingFang SC, Microsoft YaHei, sans-serif"><title>{escape(title)}</title><rect width="800" height="{height}" rx="18" fill="white"/>' + text(30, 49, title, 28, weight=700) + content + '</svg>\n'
        (ROOT/name).write_text(svg, encoding='utf-8')
    task_models = ''.join(box(30, 144+i*92, 350, 72, task+'：完整模型', '#f4f0fc', '#dbd0ed') for i, task in enumerate(('客服', '代码', 'ALFWorld')))
    task_adapters = ''.join(box(420, 144+i*67, 350, 52, task+' adapter') for i, task in enumerate(('客服', '代码', 'ALFWorld')))
    save('ch6_lora_tasks.svg', '同一个底座，多个任务版本',
         text(30, 108, '全参数：每个任务一份完整权重', 21) +
         text(420, 108, 'LoRA：每个任务一份小 adapter', 21) +
         task_models + task_adapters + text(587, 372, '↑', 30) +
         box(420, 389, 350, 68, '一份共享的基础模型', '#f3f6f8', '#cbd5e1') +
         text(30, 499, '小 adapter 易于保存、分发与切换，需配套基础模型。', 22) +
         text(30, 543, '图块大小不代表显存比例；并发容量取决于后端和资源。', 20), 580)
    save('ch6_lora_flow.svg', 'LoRA：冻结原权重，训练低秩分支',
         box(30,80,740,70,'全参数：x → W（全部可训练）→ y','#f4f0fc','#dbd0ed') +
         text(250,196,'同一个输入 x 分成两路') +
         text(195,240,'↓',30) + text(573,240,'↓',30) +
         box(30,260,350,88,'W 冻结，不更新','#f3f6f8','#cbd5e1') + box(420,260,350,88,'A → B 可训练') +
         text(145,391,'Wx',28) + text(516,391,'s · B(Ax)',28) +
         text(195,439,'↓',30) + text(573,439,'↓',30) +
         box(145,461,510,76,'相加：y = Wx + s · B(Ax)') +
         text(30,582,'只更新 A、B；W 仍参与计算。s 通常取 α/r。',21), 620)
    save('ch6_lora_rank.svg','rank：用两个小矩阵表示权重增量',
         text(30,98,'单层算术示例：4096 × 4096，rank = 32',22) +
         box(30,129,740,75,'直接训练：4096 × 4096 = 16,777,216 个参数','#f3f6f8','#cbd5e1') +
         text(370,251,'↓',34) +
         box(30,280,350,86,'B：4096 × 32') + box(420,280,350,86,'A：32 × 4096') +
         text(385,333,'×',25) +
         text(30,418,'LoRA：4096 × 32 + 32 × 4096 = 262,144',23) +
         text(30,463,'只需训练原矩阵参数量的 1.56%',27,'#087f82',700) +
         text(30,508,'这是一层的示例，不是本章模型约 2% 的全局比例。',21) +
         text(30,545,'网页中可以切换 rank，观察参数量变化。',21), 580)


if __name__ == '__main__':
    build_fallbacks()
