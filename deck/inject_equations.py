"""Replace the deck's equation placeholders (text boxes reading ⟦EQ:name:pt⟧) with native PowerPoint equations.

Each placeholder shape becomes <mc:AlternateContent>: the Choice (Office 2010+) holds the same text box with an
OMML equation (<a14:m><m:oMathPara>), the Fallback holds it with the plain-text form, as PowerPoint itself writes it.
Usage: inject_equations.py deck.pptx   (rewritten in place)
"""
import json
from pathlib import Path
import re
import sys
import zipfile
from copy import deepcopy

from lxml import etree

NS = dict(p='http://schemas.openxmlformats.org/presentationml/2006/main', a='http://schemas.openxmlformats.org/drawingml/2006/main',
          m='http://schemas.openxmlformats.org/officeDocument/2006/math', mc='http://schemas.openxmlformats.org/markup-compatibility/2006',
          a14='http://schemas.microsoft.com/office/drawing/2010/main')
TOKEN = re.compile(r'⟦EQ:([A-Za-z0-9_]+):([0-9.]+)⟧')


def text_of(sp):
    return ''.join(sp.xpath('.//a:t/text()', namespaces=NS))


def set_paragraph(sp, children):
    body = sp.find('p:txBody', NS)
    for p in body.findall('a:p', NS):
        body.remove(p)
    para = etree.SubElement(body, f'{{{NS["a"]}}}p')
    for child in children:
        para.append(child)
    return para


def main(path):
    equations = json.loads(Path(__file__).with_name('eq_omml.json').read_text(encoding='utf-8'))
    src = zipfile.ZipFile(path)
    items = {n: src.read(n) for n in src.namelist()}
    src.close()
    count = 0
    for name in [n for n in items if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)]:
        root = etree.fromstring(items[name])
        changed = False
        for sp in root.xpath('//p:sp', namespaces=NS):
            match = TOKEN.search(text_of(sp))
            if not match:
                continue
            eq, pt = match.group(1), float(match.group(2))
            size = str(int(round(pt * 100)))
            run_pr = sp.find('.//a:r/a:rPr', NS)
            # Choice: native equation.
            choice_sp = deepcopy(sp)
            math = etree.Element(f'{{{NS["a14"]}}}m', nsmap={'a14': NS['a14']})
            para_math = etree.SubElement(math, f'{{{NS["m"]}}}oMathPara', nsmap={'m': NS['m']})
            jc = etree.SubElement(etree.SubElement(para_math, f'{{{NS["m"]}}}oMathParaPr'), f'{{{NS["m"]}}}jc')
            jc.set(f'{{{NS["m"]}}}val', 'left')
            para_math.append(etree.fromstring(equations[eq]['omml'].replace('{SZ}', size)))
            end = etree.Element(f'{{{NS["a"]}}}endParaRPr', lang='en-US', sz=size, dirty='0')
            set_paragraph(choice_sp, [math, end])
            # Fallback: the same box with the plain-text form.
            fallback_sp = deepcopy(sp)
            run = etree.Element(f'{{{NS["a"]}}}r')
            rpr = deepcopy(run_pr) if run_pr is not None else etree.Element(f'{{{NS["a"]}}}rPr', lang='en-US', sz=size)
            run.append(rpr)
            etree.SubElement(run, f'{{{NS["a"]}}}t').text = equations[eq]['linear']
            set_paragraph(fallback_sp, [run])
            alt = etree.Element(f'{{{NS["mc"]}}}AlternateContent', nsmap={'mc': NS['mc']})
            choice = etree.SubElement(alt, f'{{{NS["mc"]}}}Choice', nsmap={'a14': NS['a14']}, Requires='a14')
            choice.append(choice_sp)
            etree.SubElement(alt, f'{{{NS["mc"]}}}Fallback').append(fallback_sp)
            sp.getparent().replace(sp, alt)
            count += 1
            changed = True
        if changed:
            items[name] = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as out:
        for n, data in items.items():
            out.writestr(n, data)
    leftover = sum(len(TOKEN.findall(items[n].decode('utf-8', 'ignore'))) for n in items if n.startswith('ppt/slides/slide') and n.endswith('.xml'))
    print(f'{count} equations injected; placeholders left outside equations: {leftover - count if leftover > count else 0}')


if __name__ == '__main__':
    main(sys.argv[1])
