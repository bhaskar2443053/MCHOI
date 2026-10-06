// Write the deck's theme colours and name into ppt/theme/theme1.xml. pptxgenjs writes the two theme fonts but always
// Office's default palette, so scheme colours (pres.SchemeColor) would otherwise resolve to Office blue and orange.
const fs = require('fs');

const JSZip = require(require.resolve('jszip', { paths: [require.resolve('pptxgenjs')] }));
const SLOTS = ['dk1', 'lt1', 'dk2', 'lt2', 'accent1', 'accent2', 'accent3', 'accent4', 'accent5', 'accent6', 'hlink', 'folHlink'];
const escape = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

async function applyTheme(deck, theme) {
  for (const k of SLOTS) {
    if (!/^[0-9A-Fa-f]{6}$/.test(theme.colors[k])) throw new Error(`theme colour ${k} must be six hex digits`);
  }
  const zip = await JSZip.loadAsync(fs.readFileSync(deck));
  const part = 'ppt/theme/theme1.xml';
  const name = escape(theme.name);
  const scheme = `<a:clrScheme name="${name}">` + SLOTS.map((k) => `<a:${k}><a:srgbClr val="${theme.colors[k].toUpperCase()}"/></a:${k}>`).join('') + '</a:clrScheme>';
  const xml = (await zip.file(part).async('string'))
    .replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, () => scheme)
    .replace(/(<a:(?:theme|fontScheme)\b[^>]*?\bname=")[^"]*"/g, (_, head) => `${head}${name}"`);
  if (!xml.includes(scheme)) throw new Error(`${part} has no colour scheme to replace`);
  zip.file(part, xml);
  // A scheme colour passed to a hex-only option is written as <a:srgbClr val="bg2"/>, which PowerPoint rejects.
  for (const file of Object.keys(zip.files).filter((f) => f.endsWith('.xml'))) {
    const bad = (await zip.file(file).async('string')).match(/<a:srgbClr val="((?![0-9A-Fa-f]{6}")[^"]*)"/);
    if (bad) throw new Error(`${file}: <a:srgbClr val="${bad[1]}"> is not a hex colour (scheme colour in a hex-only option)`);
  }
  fs.writeFileSync(deck, await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }));
}

module.exports = { applyTheme };
