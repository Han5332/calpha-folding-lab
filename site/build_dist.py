"""Assemble the whole website (scroll tour + lab) into one folder.

usage: python3 site/build_dist.py <trajectory.json> <out_dir> [--ref HEAD]

out_dir/index.html   the tour (complete page)          out_dir/page.html  the tour without the document skeleton,
out_dir/tour.html    the tour again, for links back                           for publishing as a claude.ai Artifact
out_dir/lab.html     the lab, from the given git ref, its site tabs pointed at tour.html
out_dir/*.js         the lab's scripts from the same ref

The lab is taken from a commit (default HEAD), never from uncommitted work in the checkout.
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAB_FILES = ['app.js', 'benchmark.js', 'engine.js', 'evo-examples.js', 'evo-feat.js', 'evo-model.js', 'evo-msa.js',
             'evo-worker.js', 'evo.js', 'kb.js', 'natives.js', 'viewer3d.js', 'worker.js']

def git_show(ref, path):
    return subprocess.run(['git', '-C', ROOT, 'show', f'{ref}:{path}'], check=True, capture_output=True).stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('trajectory'); ap.add_argument('out'); ap.add_argument('--ref', default='HEAD')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    py = sys.executable
    src = os.path.join(HERE, 'src.html')
    # the tour: complete page (index + tour) and the skeleton-less copy for an Artifact, all linking to lab.html
    tmp = os.path.join(a.out, '_src.html')
    open(tmp, 'w').write(open(src).read().replace('LAB_URL', 'lab.html'))
    subprocess.run([py, os.path.join(HERE, 'build_site.py'), a.trajectory, tmp, os.path.join(a.out, 'index.html'), os.path.join(a.out, 'page.html')], check=True)
    os.remove(tmp)
    open(os.path.join(a.out, 'tour.html'), 'w').write(open(os.path.join(a.out, 'index.html')).read())
    # the lab, from a commit: cache-busting query strings dropped (each file is published under its plain name)
    lab = git_show(a.ref, 'index.html').decode()
    lab = re.sub(r'(<script src="[\w.-]+\.js)\?[\w]+"', r'\1"', lab)
    # the lab's site tabs point at the repo's tour (site/index.html) and at itself (index.html): use the published names
    if 'class="navtabs"' not in lab:
        sys.exit('the lab has no site tabs (needs a commit with the shared top bar)')
    lab = lab.replace('href="site/index.html', 'href="tour.html').replace('href="index.html"', 'href="lab.html"')
    if not lab.lstrip().lower().startswith('<!doctype'):
        # the lab page was written for an Artifact wrapper; as a second page it needs its own document skeleton
        j = lab.index('</style>') + len('</style>')
        base = '<style>html,body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>\n'
        lab = '<!doctype html>\n<html lang="en">\n<head>\n' + base + lab[:j] + '\n</head>\n<body>\n' + lab[j:] + '\n</body>\n</html>\n'
    open(os.path.join(a.out, 'lab.html'), 'w').write(lab)
    for f in LAB_FILES:
        data = git_show(a.ref, f)
        if f == 'app.js':
            data = re.sub(rb"new Worker\('([\w.-]+\.js)\?\w+'\)", rb"new Worker('\1')", data)
        open(os.path.join(a.out, f), 'wb').write(data)
    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f'built {a.out}: {len(os.listdir(a.out))} files, {total / 1e6:.1f} MB, lab from {a.ref}')


if __name__ == '__main__':
    main()
