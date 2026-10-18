"""Create the two-file submission ZIP and the complete reproducibility archive."""
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parent
out=root/'deliverables'
stem='YourName_YourMatriculationNumber'
submission=out/f'AI6101_Project_{stem}.zip'
with zipfile.ZipFile(submission,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in [f'AI6101_Code_{stem}.ipynb',f'AI6101_Report_{stem}.pdf']:
        archive.write(out/name,name)
    assert archive.testzip() is None
files=[root/'README.md',root/'.gitignore',root/'requirements.txt',root/'requirements-report.txt']
files+=list(root.glob('build_*.py'))+[root/'package_deliverables.py']
for folder in ['rl_assignment','tests','results','deliverables']:
    files.extend(p for p in (root/folder).rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and p.suffix!='.zip')
complete=out/'AI6101_Complete_Materials.zip'
with zipfile.ZipFile(complete,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(files):archive.write(path,path.relative_to(root))
    assert archive.testzip() is None
print(submission);print(complete)
