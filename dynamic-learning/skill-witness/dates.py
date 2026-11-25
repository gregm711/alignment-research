"""Collect chronological Moment.js localized short dates from hash-pinned npm releases."""
import argparse
import base64
import calendar
from datetime import date
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen

CACHE = Path.home() / '.cache' / 'alignment-research' / 'skill-witness' / 'moment'
RELEASES = {
    '2.5.0': 'dN31ouoiP0Nz6rWQrxNnmSVVaaV/luTHI1rkn/wZakXnA7XjncWLuqHh1Lvqs54sAGFMrGVK9iJZbeRhP7x3ug==',
    '2.8.4': 'PZTKkUyBuAAZvh20DslqxFYSTVsB7J5gdzJ9WS67XynvWsXPkDSqtkVMJrLQCGB2+tGwvDFbVhe6l6BK/ER86A==',
    '2.10.6': 'Tqzz/uADMAPuqWA7iE0eMCJHhDqGuRpiwUSokCT3144wCsjeTHwzbgJY62GAvt3ax8bI8tDugD2oaMQXvJ9tBA==',
    '2.17.1': 'FmIqcMMT72OcLqks+4iWx30Tv8gwIZBJvimtnGjT/9c0GeTe28xNxNRrzJQ4xxp/2aD43b8FgImx3ATko9s/qw==',
    '2.20.1': 'Yh9y73JRljxW5QxN08Fner68eFLxM5ynNOAw2LbIB1YAGeQzZT8QFSUvkAz609Zf+IHhhaUxqZK8dG3W/+HEvg==',
    '2.23.0': '3IE39bHVqFbWWaPOMHZF98Q9c3LDKGTmypMiTM2QygGXXElkFWIH7GxfmlwmY2vwa+wmNsoYZmG2iusf1ZjJoA==',
    '2.24.0': 'bV7f+6l2QigeBBZSM/6yTNq4P2fNpSWj/0e7jQcy87A8e7o2nAfP/34/2ky5Vw4B9S446EtIhodAzkFCcR4dQg==',
    '2.29.1': 'kHmoybcPV8Sqy59DwNDY3Jefr64lK/by/da0ViFcuA4DH0vQg5Q6Ze5VimxkfQNSC+Mls/Kx53s7TjP1RhFEDQ==',
    '2.29.4': '5LC9SOxjSc2HF6vO2CyuTDNivEdoz2IvyJJGj6X8DJ0eFyfszE0QiEd+iXmBvUP3WHxSjFH/vIsA0EN00cgr8w==',
}
LOCALES = ('en en-gb en-au fr fr-ca de de-at de-ch es pt-br ru uk pl fi el ar '
           'ar-sa fa he hi bn ta id th ja ko zh-cn zh-tw af sw').split()
JS = """
const data = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const vm = require('vm');
const context = vm.createContext({});
vm.runInContext(data.source, context, {timeout: 3000});
const moment = context.moment;
if (!moment || moment.version !== data.version) throw Error('bundle version mismatch');
const locale = moment.locale || moment.lang;
const result = {};
for (const id of data.locales) {
  locale.call(moment, 'en');
  locale.call(moment, id);
  if (locale.call(moment) !== id) { result[id] = null; continue; }
  result[id] = {};
  for (const [key, {y,m,d}] of Object.entries(data.inputs)) {
    const value = moment.utc([Number(y), Number(m)-1, Number(d)]);
    if (!value.isValid() || value.year() !== Number(y) ||
        value.month() !== Number(m)-1 || value.date() !== Number(d)) throw Error('invalid test date');
    result[id][key] = value.format('L');
  }
}
process.stdout.write(JSON.stringify(result));
"""


def collect():
    inputs = {}
    for year in (1999, 2000, 2011, 2024):
        for month in (1, 2, 4, 11, 12):
            for day in sorted({1, 4, 10, 13, month, calendar.monthrange(year, month)[1]}):
                key = date(year, month, day).isoformat()
                inputs[key] = {'y': str(year), 'm': str(month), 'd': str(day)}
    producers = {locale: [] for locale in LOCALES}
    sources = {}
    CACHE.mkdir(parents=True, exist_ok=True)
    for version, digest in RELEASES.items():
        path = CACHE / f'moment-{version}.tgz'
        url = 'https://registry.npmjs.org/moment/-/' + path.name
        if path.exists():
            raw = path.read_bytes()
        else:
            with urlopen(url, timeout=45) as response:
                raw = response.read()
        if base64.b64encode(hashlib.sha512(raw).digest()).decode() != digest:
            raise ValueError(f'npm archive integrity mismatch for {version}')
        if not path.exists():
            with tempfile.NamedTemporaryFile(dir=CACHE, delete=False) as partial:
                partial.write(raw)
            Path(partial.name).replace(path)
        filename = 'moment-with-langs.js' if version == '2.5.0' else 'moment-with-locales.js'
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            source = archive.extractfile('package/min/' + filename).read().decode('utf-8')
        request = dict(source=source, version=version, inputs=inputs, locales=LOCALES)
        stdout = subprocess.run(['node', '-e', JS], input=json.dumps(request),
                                capture_output=True, text=True, timeout=30)
        if stdout.returncode:
            raise RuntimeError(f'{version}: {stdout.stderr[-2000:]}')
        panel = json.loads(stdout.stdout)
        assert panel.keys() == producers.keys() and panel['en'] is not None
        for locale, values in panel.items():
            if values is not None:
                assert values.keys() == inputs.keys() and all(isinstance(v, str) and v for v in values.values())
            producers[locale].append(values)
        sources[version] = {'url': url, 'integrity': 'sha512-' + digest, 'bundle': filename}
    return dict(project='moment-js', releases=list(RELEASES), producers=producers, inputs=inputs,
                sources=dict(releases=sources, node=subprocess.check_output(['node','-v'], text=True).strip()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    dest = parser.parse_args().output
    panel = collect()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=dest.parent, delete=False) as out:
        json.dump(panel, out, ensure_ascii=False, separators=(',', ':'))
        out.write('\n')
    Path(out.name).replace(dest)
    active = [sum(v is not None for v in row) for row in zip(*panel['producers'].values())]
    print(f"{len(panel['inputs'])} inputs, available locales by release {active}")
