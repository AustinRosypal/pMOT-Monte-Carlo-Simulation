"""Verify 12 rendered figures, standalone HTML, and a clean notebook execution."""

import argparse
from base64 import b64decode
from hashlib import sha256
from html.parser import HTMLParser

import nbformat
from nbclient import NotebookClient

from validate_pmot_2level_loading import OUTPUT, ROOT, write_json


class Standalone(HTMLParser):
    def __init__(self):
        super().__init__()
        self.figures = []
        self.tables = 0

    def handle_starttag(self, tag, attrs):
        props = dict(attrs)
        if tag == 'table':
            self.tables += 1
        if tag == 'img':
            src = props.get('src', '')
            if not src.startswith('data:image/png;base64,'):
                raise ValueError('HTML requires external image assets')
            self.figures.append(b64decode(src.split(',', 1)[1], validate=True))
        if ('src' in props and tag != 'img') or tag == 'link':
            raise ValueError('Unexpected external asset')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--impact-stratified', action='store_true')
    args = parser.parse_args()
    run = OUTPUT.name
    nb_file = ROOT / 'notebooks/pMOT_2Level' / f'{run}.ipynb'
    html_file = ROOT / 'visualizations/pMOT_2Level' / f'{run}.html'
    png_root = ROOT / 'outputs/figures/pMOT_2Level' / run
    html = Standalone()
    html.feed(html_file.read_text(encoding='utf-8'))
    files = sorted(png_root.glob('*.png'))
    figures = {sha256(f.read_bytes()).hexdigest() for f in files}
    embedded = {sha256(img).hexdigest() for img in html.figures}
    if len(files) != 12 or len(figures) != 12 or len(html.figures) != 12 or html.tables != 13 or embedded != figures:
        raise ValueError('HTML does not include all 12 plots and their companion tables')

    def notebook_images(book):
        return [sha256(b64decode(output['data']['image/png'])).hexdigest()
                for cell in book.cells if cell.cell_type == 'code' for output in cell.outputs
                if 'image/png' in output.get('data', {})]

    book = nbformat.read(nb_file, as_version=4)
    nbformat.validate(book)
    saved = notebook_images(book)
    if len(saved) != 12 or set(saved) != figures:
        raise ValueError('Saved notebook figures differ from the exported report')
    executed = NotebookClient(book, timeout=120, resources={'metadata': {'path': str(ROOT)}}).execute()
    replayed = notebook_images(executed)
    if len(replayed) != 12 or set(replayed) != figures:
        raise ValueError('Notebook did not reproduce the audited figures on a clean kernel')
    destination = (OUTPUT / 'impact_stratified' if args.impact_stratified else OUTPUT) / 'report_verification.json'
    write_json(destination, {'passed': True, 'figures': 12, 'tables_in_html': 13,
                              'artifact_sha256': {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
                                                  for p in [nb_file, html_file, *files]}})
    print('Verified all 12 figures, standalone HTML, and clean notebook execution')


if __name__ == '__main__':
    main()
