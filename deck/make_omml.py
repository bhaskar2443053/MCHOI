"""Convert the deck's LaTeX equations (equations.py) to PowerPoint equations (OMML) plus a plain-text fallback.

LaTeX -> MathML (latex2mathml) -> OMML (Office's MML2OMML.XSL). Then, to match what PowerPoint itself writes:
summands are moved into their n-ary operator, variables use Unicode math-italic letters, and every run gets an
<a:rPr> with the Cambria Math font and a size token {SZ} filled in when the equation is placed on a slide.
Writes eq_omml.json: {name: {"omml": "<m:oMath ...>", "linear": "plain text"}}.
"""
from copy import deepcopy
import json
import os
import re
from pathlib import Path

from latex2mathml.converter import convert
from lxml import etree

from equations import EQUATIONS

M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
# MathML -> OMML stylesheet that ships with Microsoft Office (override with MML2OMML_XSL).
XSL = etree.XSLT(etree.parse(os.environ.get('MML2OMML_XSL', r'C:/Program Files/Microsoft Office/root/Office16/MML2OMML.XSL')))
q = lambda tag: f'{{{M}}}{tag}'


def math_italic(text):
    out = []
    for ch in text:
        o = ord(ch)
        if ch == 'h':
            out.append('\u210E')
        elif 'a' <= ch <= 'z':
            out.append(chr(0x1D44E + o - ord('a')))
        elif 'A' <= ch <= 'Z':
            out.append(chr(0x1D434 + o - ord('A')))
        elif 0x03B1 <= o <= 0x03C9 and ch != 'ς':
            out.append(chr(0x1D6FC + o - 0x03B1))
        elif ch == 'ϑ':
            out.append('\U0001D717')
        elif ch == 'ϕ':
            out.append('\U0001D719')
        else:
            out.append(ch)
    return ''.join(out)


def fill_nary(root):
    """MML2OMML leaves the n-ary operand empty; move the following terms (up to a separator) into it."""
    while True:
        target = next((n for n in root.iter(q('nary')) if (e := n.find(q('e'))) is not None and len(e) == 0), None)
        if target is None:
            return
        e = target.find(q('e'))
        sibling = target.getnext()
        while sibling is not None:
            text = ''.join(sibling.itertext()) if sibling.tag == q('r') else ''
            if sibling.tag == q('r') and text.startswith((',', '=', '\u2003')):
                break
            nxt = sibling.getnext()
            e.append(sibling)
            sibling = nxt


def merge_separators(root):
    """MML2OMML turns '|a - b|' or '(a - b)' into a delimiter whose separator is '-'; PowerPoint draws such
    separators as stretchy glyphs. Keep only ',' separators; otherwise put the operators back as ordinary runs."""
    for d in list(root.iter(q('d'))):
        pr = d.find(q('dPr'))
        sep = pr.find(q('sepChr')) if pr is not None else None
        if sep is None or sep.get(q('val')) == ',':
            continue
        char = sep.get(q('val'))
        pr.remove(sep)
        parts = d.findall(q('e'))
        if len(parts) < 2:
            continue
        first = parts[0]
        for e in parts[1:]:
            if char:
                run = etree.SubElement(first, q('r'))
                etree.SubElement(run, q('t')).text = char
            for child in list(e):
                first.append(child)
            d.remove(e)


def style_runs(root):
    for r in root.iter(q('r')):
        rpr = r.find(q('rPr'))
        plain = rpr is not None and (rpr.find(q('nor')) is not None or
                                     (rpr.find(q('sty')) is not None and rpr.find(q('sty')).get(q('val')) == 'p'))
        t = r.find(q('t'))
        if t is not None and t.text and not plain:
            t.text = math_italic(t.text)
        if t is not None and t.text and (' ' in t.text):
            t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        for w in [c for c in r if not c.tag.startswith(f'{{{M}}}')]:
            r.remove(w)
        arpr = etree.Element(f'{{{A}}}rPr', nsmap={'a': A})
        arpr.set('lang', 'en-US')
        arpr.set('sz', '{SZ}')
        arpr.set('i', '0' if plain else '1')
        latin = etree.SubElement(arpr, f'{{{A}}}latin')
        latin.set('typeface', 'Cambria Math'); latin.set('panose', '02040503050406030204'); latin.set('pitchFamily', '18'); latin.set('charset', '0')
        r.insert(1 if rpr is not None else 0, arpr)
    # PowerPoint drops a supplementary-plane letter (math italic/bold) that follows a BMP character in the same run,
    # so split runs at every change between the two planes.
    for r in list(root.iter(q('r'))):
        t = r.find(q('t'))
        if t is None or not t.text:
            continue
        parts, current = [], ''
        for ch in t.text:
            if current and (ord(ch) > 0xFFFF) != (ord(current[-1]) > 0xFFFF):
                parts.append(current)
                current = ''
            current += ch
        parts.append(current)
        t.text = parts[0]
        anchor = r
        for part in parts[1:]:
            clone = deepcopy(r)
            clone.find(q('t')).text = part
            anchor.addnext(clone)
            anchor = clone
    for t in root.iter(q('t')):
        if t.text and t.text != t.text.strip():
            t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')


def linear(node):
    """Readable plain-text form for viewers without equation support."""
    tag = etree.QName(node).localname
    kids = lambda name: node.find(q(name))
    lin = lambda n: '' if n is None else ''.join(linear(c) for c in n)
    wrap = lambda s: s if len(s) <= 1 else '(' + s + ')'
    if tag == 'r':
        return ''.join(t.text or '' for t in node.iter(q('t')))
    if tag == 'sSub':
        return lin(kids('e')) + '_' + wrap(lin(kids('sub')))
    if tag == 'sSup':
        return lin(kids('e')) + '^' + wrap(lin(kids('sup')))
    if tag == 'sSubSup':
        return lin(kids('e')) + '_' + wrap(lin(kids('sub'))) + '^' + wrap(lin(kids('sup')))
    if tag == 'f':
        return '(' + lin(kids('num')) + ')/(' + lin(kids('den')) + ')'
    if tag == 'rad':
        return '√(' + lin(kids('e')) + ')'
    if tag == 'nary':
        pr = kids('naryPr'); ch = pr.find(q('chr')).get(q('val')) if pr is not None and pr.find(q('chr')) is not None else '∫'
        sub, sup = lin(kids('sub')), lin(kids('sup'))
        return ch + ('_' + wrap(sub) if sub else '') + ('^' + wrap(sup) if sup else '') + ' ' + lin(kids('e'))
    if tag == 'd':
        pr = kids('dPr')
        get = lambda name, default: (pr.find(q(name)).get(q('val')) if pr is not None and pr.find(q(name)) is not None else default)
        sep = get('sepChr', ',')
        return get('begChr', '(') + sep.join(lin(e) for e in node.findall(q('e'))) + get('endChr', ')')
    if tag == 'acc':
        pr = kids('accPr'); ch = pr.find(q('chr')).get(q('val')) if pr is not None and pr.find(q('chr')) is not None else '\u0302'
        return lin(kids('e')) + ch
    if tag in ('limLow', 'limUpp'):
        return lin(kids('e')) + ('_' if tag == 'limLow' else '^') + wrap(lin(kids('lim')))
    if tag.endswith('Pr'):
        return ''
    return ''.join(linear(c) for c in node)


def prepare(tex):
    """Spacing commands become explicit Unicode spaces (MML2OMML drops MathML spacing)."""
    tex = tex.strip('$').replace('\\!', '')
    tex = re.sub(r'\\(min|max|sin|cos|det)(?![A-Za-z])', r'\\mathrm{\1}', tex)  # upright function names
    tex = tex.replace('arg\\,min', 'arg}\\text{ }\\mathrm{min').replace('arg\\,max', 'arg}\\text{ }\\mathrm{max')
    for cmd, space in (('\\qquad', '  '), ('\\quad', ' '), ('\\;', ' '), ('\\,', ' ')):
        tex = tex.replace(cmd, '\\text{' + space + '}')
    return tex


def main():
    result = {}
    for name, tex in EQUATIONS.items():
        mml = convert(prepare(tex))
        omml = XSL(etree.fromstring(mml.encode())).getroot()
        merge_separators(omml)
        fill_nary(omml)
        text = linear(omml)
        style_runs(omml)
        result[name] = dict(omml=etree.tostring(omml, encoding='unicode'), linear=re.sub(r'\s+', ' ', text).strip())
    Path(__file__).with_name('eq_omml.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    for name, v in result.items():
        print(f'{name:4s} {v["linear"]}')


if __name__ == '__main__':
    main()
