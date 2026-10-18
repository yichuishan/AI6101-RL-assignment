"""Build a strictly two-page PDF and shared editable-document content."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from PIL import Image
from rl_assignment.plotting import decorate_grid, ARROWS

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results';OUT=ROOT/'deliverables'
summary=json.loads((RESULTS/'summary.json').read_text())
arrays=np.load(RESULTS/'experiment_arrays.npz')
success=arrays['success'].mean(0)
explore=json.loads((RESULTS/'bonus_exploration.json').read_text())
sparse=json.loads((RESULTS/'bonus_sparse.json').read_text())
exp_arrays=np.load(RESULTS/'bonus_exploration.npz')
sparse_arrays=np.load(RESULTS/'bonus_sparse.npz')

# Two readable joint-state slices for the space-limited PDF; all four and the
# complete 84-by-84 joint-state table remain in the notebook.
model=np.load(RESULTS/'policy_train_0.0_seed_7.npz')
q,visits=model['q'],model['visits']
v=q.max(2)
known=v[visits>0]
norm=colors.Normalize(known.min(),known.max())
cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#ededed')
fig,axes=plt.subplots(1,2,figsize=(10.5,2.55),layout='constrained')
for ax,box in zip(axes,[(1,1),(2,13)]):
    bi=box[0]*14+box[1]
    matrix=np.where(visits[:,bi]>0,v[:,bi],np.nan).reshape(6,14)
    im=ax.imshow(matrix,cmap=cmap,norm=norm);decorate_grid(ax,box)
    for ai in range(84):
        if visits[ai,bi] and divmod(ai,14)!=box:
            best=np.flatnonzero(q[ai,bi]==q[ai,bi].max())
            label=ARROWS[int(best[0])+1] if len(best)==1 else '·'
            ax.text(ai%14,ai//14,label,ha='center',va='center',fontsize=10,
                    color='white' if norm(v[ai,bi])>.65 else '#172533',zorder=4)
    ax.set_title(f'Box fixed at {box}; agent position varies',fontsize=9)
fig.colorbar(im,ax=axes,label='V(s) = max Q(s,a)',shrink=.8)
fig.savefig(RESULTS/'report_policy_slices.png',dpi=210,bbox_inches='tight');plt.close(fig)

pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVu-Bold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold')
W,H=A4;M=38;WIDTH=W-2*M
path=OUT/'AI6101_Report_YourName_YourMatriculationNumber.pdf'
c=canvas.Canvas(str(path),pagesize=A4)
c.setTitle('CliffBoxPushing: Q-learning, Exploration and Sparse Rewards')
c.setAuthor('YourName — YourMatriculationNumber')
styles={
 'body':ParagraphStyle('body',fontName='DejaVu',fontSize=8.3,leading=11.1,textColor=HexColor('#233341')),
 'small':ParagraphStyle('small',fontName='DejaVu',fontSize=7.3,leading=9.7,textColor=HexColor('#41515d')),
 'head':ParagraphStyle('head',fontName='DejaVu-Bold',fontSize=10.2,leading=13.5,textColor=HexColor('#1a4f75'))}
events=[]

def block(text,y,kind='body',gap=5):
    p=Paragraph(text,styles[kind]);_,height=p.wrap(WIDTH,H)
    if y-height<34:raise ValueError(f'Text overflow: {text[:60]}, bottom={y-height:.1f}')
    p.drawOn(c,M,y-height)
    events.append(dict(type='block',text=text,kind=kind,gap=gap))
    return y-height-gap

def image(name,y,width=WIDTH,gap=4):
    with Image.open(RESULTS/name) as im:height=width*im.height/im.width
    if y-height<34:raise ValueError(f'Figure overflow: {name}, bottom={y-height:.1f}')
    c.drawImage(str(RESULTS/name),M+(WIDTH-width)/2,y-height,width=width,height=height)
    events.append(dict(type='image',name=name,width=width,gap=gap))
    return y-height-gap

def table(rows,y,widths):
    cells=[[Paragraph(str(cell),styles['small']) for cell in row] for row in rows]
    t=Table(cells,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#eaf0f5')),
       ('LINEBELOW',(0,0),(-1,0),.5,HexColor('#9fb4c5')),
       ('LINEBELOW',(0,-1),(-1,-1),.4,HexColor('#bccbd6')),
       ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),3),
       ('BOTTOMPADDING',(0,0),(-1,-1),3)]))
    _,height=t.wrap(WIDTH,H)
    if y-height<34:raise ValueError(f'Table overflow: {y-height}')
    t.drawOn(c,M,y-height)
    events.append(dict(type='table',rows=rows,widths=widths))
    return y-height-5

def header(page,title):
    c.setFont('DejaVu-Bold',14.5);c.setFillColor(HexColor('#13334c'))
    c.drawString(M,H-40,title)
    c.setFont('DejaVu',8);c.setFillColor(HexColor('#5a6976'))
    c.drawString(M,H-56,'AI6101 · YourName · YourMatriculationNumber')
    c.setStrokeColor(HexColor('#b8cad8'));c.line(M,H-64,W-M,H-64)
    c.setFont('DejaVu',7)
    c.drawString(M,21,'Fixed map and 100-step limit · 3 seeds · Mandatory study + two bonus experiments')
    c.drawRightString(W-M,21,f'{page} / 2')
    events.append(dict(type='header',title=title))
    return H-75

y=header(1,'CliffBoxPushing: learning and robustness')
y=block('1. Q-learning and the mandatory experiment',y,'head')
y=block('The fixed 6×14 world uses state (agent row, column, box row, column), four actions, '
        '7,056 Cartesian states and 28,224 Q entries. The original reward is '
        '−1−Manhattan(A,B)−Manhattan(B,G), with −1000 for a cliff and +1000 at the goal. '
        'Cliff, goal or 100 steps ends the episode. The original environment is unchanged.',y)
y=block('<b>Update:</b> Q(s,a) ← Q(s,a)+α[r+γ(1−d)max<sub>b</sub>Q(s′,b)−Q(s,a)], '
        'with α=0.1, γ=0.99, ε=0.1 and zero terminal bootstrap. Train 30,000 episodes '
        'per noise level {0,0.1,0.2}, seeds {7,17,27}. Noise replaces the intended action '
        'uniformly with one of the other three; update the intended action. Evaluate '
        '300 episodes per policy/test noise with ε=0: 8,100 episodes. Success means box at goal.',y)
y=image('learning_curves.png',y,width=500)
y=block('<b>Figure 1.</b> Original-reward training: 500-episode rolling means, bands ±1 sample SD '
        'across seeds. Clean greedy success is 100%; random actions achieve 0% in 1,000 episodes. '
        'The seed-7 learned clean route takes 39 steps, return 577. Training retains ε exploration.',y,'small')
y=block('2. Cross-noise generalization and final policy',y,'head',gap=2)
y=image('robustness_heatmaps.png',y,width=455)
y=block('<b>Figure 2.</b> Rows=train noise, columns=test noise; mean ± SD across three seeds '
        '(900 episodes/cell). Clean-trained success drops to 48.2% and 18.6% at test noise 0.1/0.2. '
        'Both noisy-training settings have zero goal visits in 90,000 training episodes each '
        'and zero evaluation successes. Their clean tests all time out: noise training did '
        'not improve robustness within this budget.',y,'small')
y=image('report_policy_slices.png',y,width=500)
y=block('<b>Figure 3.</b> Clean-trained seed-7 V(s)=max<sub>a</sub>Q(s,a), with greedy arrows '
        'conditioned on both positions. Gray=unvisited, dot=tie, B=box, G=goal. The notebook '
        'includes four slices and the complete joint-state table. Untried actions may retain optimistic zeros.',y,'small')
c.showPage();events.append(dict(type='pagebreak'))

y=header(2,'Exploration and pure sparse-reward bonuses')
y=block('3. Bonus 1: exploration comparison',y,'head')
y=block('Original rewards and primitive actions, no action noise; identical α, γ, 30,000-episode '
        'budget and three seeds. Reuse the mandatory clean ε-greedy runs. UCB tries unvisited '
        'actions first, then maximizes Q(s,a)+20√[ln(N(s)+1)/N(s,a)]. The bonus affects '
        'selection only. The greedy baseline trains with ε=0; every learned policy is evaluated '
        'without exploration for 300 episodes/seed.',y)
exp_rows=[['Training method','First goal episode: seeds 7 / 17 / 27','Eval success','Mean return']]
for j,method in enumerate(explore['methods']):
    rows=[r for r in explore['rows'] if r['method']==method]
    label={'epsilon_greedy':'ε-greedy (ε=0.1)','ucb':'UCB (c=20)','greedy':'Greedy (ε=0)'}[method]
    exp_rows.append([label,' / '.join(str(r['first_success']) for r in rows),
                     f"{100*exp_arrays['evaluation'][:,j,0].mean():.1f}%",
                     f"{exp_arrays['evaluation'][:,j,1].mean():.1f}"])
y=table(exp_rows,y,[120,205,90,WIDTH-415])
y=block('UCB discovers a goal earlier than ε-greedy (median 5,106 vs 8,038 episodes), '
        'but greedy is earlier still (4,735). Negative step rewards make untried zero Q-values '
        'optimistic, and random ties provide exploration even with ε=0. All reach 100% final '
        'clean success; these results do not establish a universal UCB advantage.',y,'small')
y=block('4. Bonus 2: only goal reward, hierarchical Q-learning',y,'head')
y=block('<b>Reward is exactly 1000 at the original goal, otherwise 0</b>, including cliffs and '
        'timeout. No dense pretraining, intermediate reward or demonstration is used. '
        'Hand-designed options use local BFS to position the agent safely, then make one safe '
        'push. This map/safety prior does not plan the full box route or use goal distance. '
        'The high-level push choice is learned. Every primitive action executes the original '
        'step() and counts toward 100; dead ends continue safely until the original timeout.',y)
y=block('For an option lasting k steps, the target is '
        'Σ<sub>i=0</sub><super>k−1</super> γ<super>i</super>r<sub>i</sub> '
        '+(1−d)γ<super>k</super>max<sub>o</sub>Q(s′,o). The replay method performs 10 backward '
        'passes over actual successful trajectories. All four ablations train 5,000 episodes/seed '
        'and evaluate 300 episodes/seed; random options remain random in evaluation.',y)
sp_rows=[['Sparse training method','First goal: seeds 7 / 17 / 27','Eval success','Mean steps']]
for j,method in enumerate(sparse['methods']):
    rows=[r for r in sparse['rows'] if r['method']==method]
    label={'primitive_epsilon':'Primitive ε-greedy','random_options':'Random safe options',
           'option_q':'Option Q, no replay','option_q_replay':'Option Q + replay ×10'}[method]
    first=' / '.join('none' if r['first_success'] is None else str(r['first_success']) for r in rows)
    sp_rows.append([label,first,f"{100*sparse_arrays['evaluation'][:,j,0].mean():.1f}%",
                    f"{sparse_arrays['evaluation'][:,j,2].mean():.1f}"])
y=table(sp_rows,y,[135,190,90,WIDTH-415])
y=block('Only the replay learner reaches 100% final success (900/900; sparse return 1000). '
        'No-replay learns the same first goals but fails final tests: replay improves propagation, '
        'not first discovery. Equal episode caps do not imply equal interactions or compute; '
        'the notebook records both, including extra replay updates.',y,'small')
y=image('bonus_learning_curves.png',y,width=500)
y=block('<b>Figure 4.</b> Goal-reaching training rates: exploration comparison (500-episode '
        'window) and sparse ablations (100-episode window); mean ± SD across three seeds.',y,'small')
y=block('<b>Scope.</b> Three seeds limit precision. Constant α/ε and the four-coordinate '
        'state (without time remaining) do not prove convergence or optimality. Sparse skills '
        'supply substantial map knowledge; the ablations explicitly isolate their contribution. '
        '<b>References:</b> Watkins &amp; Dayan (1992), Q-learning; Auer et al. (2002), '
        'Finite-time Analysis of the Multiarmed Bandit Problem; Sutton, Precup &amp; Singh '
        '(1999), Between MDPs and semi-MDPs; supplied AI6101 brief/environment.',y,'small',gap=0)
c.save()
(RESULTS/'report_content.json').write_text(json.dumps(dict(events=events,styles={
    name:dict(font_size=style.fontSize,leading=style.leading) for name,style in styles.items()}),indent=2))
print(path);print(f'Page 2 content bottom: {y:.1f} pt')
