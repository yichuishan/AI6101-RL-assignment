"""Produce the strictly two-page report from measured experiment artifacts."""
from pathlib import Path
import json
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'deliverables'
RESULTS = ROOT / 'results'
summary = json.loads((RESULTS / 'summary.json').read_text())
arrays = np.load(RESULTS / 'experiment_arrays.npz')
success = arrays['success'].mean(0)
returns = arrays['mean_return'].mean(0)
sd = arrays['success'].std(0, ddof=1)
config = summary['config']
baseline = summary['baseline']
rollout = summary['rollout']

pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVu-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu-Bold')
W, H = A4
M = 38
WIDTH = W - 2*M
path = OUT / 'AI6101_Report_YourName_YourMatriculationNumber.pdf'
c = canvas.Canvas(str(path), pagesize=A4)
c.setTitle('Q-learning for CliffBoxPushing: Learning and Action-noise Robustness')
c.setAuthor('YourName — YourMatriculationNumber')
styles = {
    'body': ParagraphStyle('body', fontName='DejaVu', fontSize=8.8, leading=12.0,
                           textColor=HexColor('#233341'), spaceAfter=5),
    'small': ParagraphStyle('small', fontName='DejaVu', fontSize=7.6, leading=10.3,
                            textColor=HexColor('#41515d')),
    'head': ParagraphStyle('head', fontName='DejaVu-Bold', fontSize=10.5, leading=14,
                           textColor=HexColor('#1a4f75')),
}

def block(text, y, kind='body', gap=6):
    p = Paragraph(text, styles[kind])
    _, height = p.wrap(WIDTH, H)
    if y - height < 36:
        raise ValueError(f'Report overflow at {text[:70]}: y={y}, height={height}')
    p.drawOn(c, M, y-height)
    return y-height-gap

def image(name, y, width=WIDTH, gap=5):
    img = RESULTS / name
    with Image.open(img) as im:
        height = width * im.height / im.width
    if y-height < 36:
        raise ValueError(f'Image overflow: {name}, bottom={y-height}')
    c.drawImage(str(img), M+(WIDTH-width)/2, y-height, width=width, height=height)
    return y-height-gap

def header(page, title):
    c.setFillColor(HexColor('#13334c'))
    c.setFont('DejaVu-Bold', 15)
    c.drawString(M, H-42, title)
    c.setFont('DejaVu', 8)
    c.setFillColor(HexColor('#5a6976'))
    c.drawString(M, H-58, 'AI6101 · YourName · YourMatriculationNumber')
    c.setStrokeColor(HexColor('#b8cad8'))
    c.line(M, H-67, W-M, H-67)
    c.setFont('DejaVu', 7.5)
    c.drawString(M, 22, 'Unchanged supplied environment · 3 independent training seeds')
    c.drawRightString(W-M, 22, f'{page} / 2')
    return H-80

y = header(1, 'Q-learning for CliffBoxPushing')
y = block('1. Task, algorithm and controlled experiment', y, 'head')
y = block('The agent must push a box through the fixed 6×14 map to (4,13). '
          'The state is (agent row, agent column, box row, box column); four actions '
          'give 7,056 Cartesian states and 28,224 Q entries. We keep the supplied map, '
          'reward and step() unchanged. Episodes terminate at a cliff, the goal or 100 steps. '
          'Reward is −1 − Manhattan(A,B) − Manhattan(B,G), with −1000 for a cliff '
          'and +1000 for reaching the goal.', y)
y = block('<b>Q-learning:</b> Q(s,a) ← Q(s,a) + α[r + γ(1−d) max<sub>b</sub> Q(s′,b) − Q(s,a)]. '
          'Here d marks termination, so terminal transitions do not bootstrap. '
          'Initialize Q to zero; use α=0.1, γ=0.99 and fixed ε=0.1 with uniform greedy tie-breaking. '
          'With probability p, replace the intended action uniformly with one of the other three. '
          'Update Q for the <b>intended</b> action, learning the noisy decision channel.', y)
y = block(f'Train {config["episodes"]:,} episodes per policy for p<sub>train</sub>∈{{0,0.1,0.2}} '
          'and seeds {7,17,27}: nine policies. Evaluate each for 300 episodes per '
          'p<sub>test</sub> with ε=0 and separate RNGs (8,100 evaluation episodes). '
          '<b>Success means box at goal</b>, not return &gt;0. We use no planning, reward '
          'engineering or extra state features.', y)
y = block('2. Learning performance', y, 'head', gap=2)
y = image('learning_curves.png', y)
clean_train = arrays['training_success'][:, 0, -2000:].mean()
y = block(f'<b>Figure 1.</b> 500-episode moving means; bands are ±1 sample SD across seeds. '
          f'Clean-trained greedy success is {100*success[0,0]:.1f}% '
          f'(SD {100*sd[0,0]:.1f} percentage points), versus '
          f'{100*np.mean(baseline["success"]):.1f}% for 1,000 random episodes '
          f'(mean return {np.mean(baseline["returns"]):.1f}). '
          f'The last 2,000 clean-training episodes succeed {100*clean_train:.1f}% of the time: '
          'ε exploration remains active during training, so training and greedy evaluation differ.', y, 'small')
y = block('3. A learned clean solution', y, 'head', gap=2)
y = image('greedy_trajectory.png', y)
y = block(f'<b>Figure 2.</b> Preselected seed 7, clean-trained greedy rollout: '
          f'{len(rollout["actions"])} actions, undiscounted return '
          f'{sum(rollout["rewards"]):.0f}, box at goal={rollout["success"]}. '
          'Blue/orange arrows show the agent/box routes. The agent must reposition to '
          'push around cliffs; distance to the goal alone is insufficient.', y, 'small')
c.showPage()

y = header(2, 'Robustness and learned policy')
y = block('4. Action-noise robustness', y, 'head', gap=2)
y = image('robustness_heatmaps.png', y, width=480)
y = block('<b>Figure 3.</b> Each cell uses 300 evaluation episodes per seed (900 total). '
          'Entries are means ± sample SD across the three training seeds; SD is not a '
          'confidence interval. Rows are training noise, columns test noise. No ε exploration '
          'or learning is used in evaluation.', y, 'small')
delta1 = 100*(success[1,1]-success[0,1])
delta2 = 100*(success[1,2]-success[0,2])
direction = 'improves' if delta1 > 0 else ('reduces' if delta1 < 0 else 'does not change')
y = block(f'For clean training, success falls from {100*success[0,0]:.1f}% at test noise 0 '
          f'to {100*success[0,1]:.1f}% at 0.1 and {100*success[0,2]:.1f}% at 0.2. '
          'A single corrupted action near a cliff can terminate a long trajectory. '
          f'Training at p=0.1 {direction} success at test p=0.1 by '
          f'{abs(delta1):.1f} percentage points; the change at test p=0.2 is '
          f'{delta2:+.1f} points. '
          'This is a descriptive result under a fixed training budget, not evidence that '
          'noise training universally helps. Larger training noise also increases unsafe '
          'transitions and makes useful successful experience rarer. '
          '<b>Neither noisy-training setting reached the goal in its 90,000 training '
          'episodes across seeds; all their clean greedy evaluations timed out.</b> '
          'This points to unsuccessful goal discovery under these settings, not a '
          'successfully learned robust strategy.', y)
y = block('5. Values, policy and scope', y, 'head', gap=2)
y = image('value_policy_slices.png', y, width=480)
y = block('<b>Figure 4.</b> Final clean-trained seed-7 table. Color is V(s)=max<sub>a</sub>Q(s,a) '
          '(not divided by state count); arrows are unique greedy actions. Each panel '
          'fixes the box and varies the agent position. Gray means unvisited and a dot '
          'means a tie. The notebook also plots the entire joint-state table and lists the '
          '20 highest-valued visited states. Unvisited actions can retain optimistic zeros.', y, 'small')
y = block('<b>Limitations.</b> Three seeds provide limited precision. The four-coordinate '
          'state omits time remaining, so the stationary table approximates the 100-step '
          'finite-horizon task. Constant α and ε do not prove convergence or route optimality. '
          'We retain failed runs and do not attempt the optional UCB/sparse-reward bonuses.', y, 'small')
y = block('<b>References.</b> Watkins &amp; Dayan (1992), “Q-learning”, <i>Machine Learning</i> '
          '8:279–292. Sutton &amp; Barto (2018), <i>Reinforcement Learning: An Introduction</i>, '
          '2nd ed., §6.5. AI6101 assignment brief and supplied RLAgent_2026Fall.ipynb.', y, 'small', gap=0)
c.save()
print(path)
print(f'Final page content bottom: {y:.1f} pt')
