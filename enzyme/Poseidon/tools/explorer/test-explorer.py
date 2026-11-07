#!/usr/bin/env python3
"""End-to-end checks of a Poseidon Compiler Explorer through its REST API.

  test-explorer.py [--url http://localhost:10240] [--quick] [--jobs N] [--out DIR]

Exits non-zero when a check fails. --quick skips the checks that wait out a
timeout and the concurrency check.
"""
import argparse, concurrent.futures, glob, json, os, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, 'examples', 'c++')
ANSI = re.compile(r'\x1b\[[0-9;]*m')

HEADER = '''#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
POSEIDON_OPTIMIZE void kernel(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i) y[i] = (1.0 - std::cos(x[i])) / (x[i] * x[i]);
}
'''
SITE = '''static double x[64], y[64];
  for (int i = 0; i < 64; ++i) x[i] = 1e-6 * (i + 1);
  kernel(x, y, 64);
  printf("%g\\n", y[0]);'''
NO_SITE = HEADER.replace('POSEIDON_OPTIMIZE ', '') + \
    'int main() { double x[1] = {1e-6}, y[1]; kernel(x, y, 1); printf("%g\\n", y[0]); }\n'
SYNTAX = HEADER + 'int main() { return undeclared; }\n'
CRASH = HEADER + 'int main() {\n  ' + SITE + '\n  abort();\n}\n'
SPIN = HEADER + 'int main() {\n  ' + SITE + '\n  for (volatile int k = 0;; ++k) {}\n}\n'
SITE_ONLY = HEADER + 'int main() {\n  ' + SITE + '\n}\n'
BUDGET = '-O2 -poseidon-tau=0 -poseidon-comp-cost-budget=-3453952'


def rows(out):
    return [l.split() for l in out.splitlines()[1:]]


EXPECT = {
    'cancellation': (r'^Applying solution', lambda o: all(r[2] == '0.5' for r in rows(o))),
    'lulesh_compression': (r'^Applying solution for .* \(/\.f64 \(-\.f64',
                           lambda o: all(abs(float(r[2]) / float(r[3]) - 1) < 1e-15 for r in rows(o))),
    'precision': (r'^Applying solution for (CS: All FP64\(0%\) \+ FP32\(100%\)|.*-> \(\*\.f32)',
                  lambda o: all(0 < float(r[3]) < 1e-6 for r in rows(o))),
    'refused': (r'no rewrite applied', lambda o: all(float(r[3]) == 0 for r in rows(o))),
    'quaternion': (r'^Applying solution', lambda o: all(r[1] != r[2] for r in rows(o))),
    'planck': (r'^Applying solution for .*expm1', lambda o: abs(float(rows(o)[0][2]) - 0.9999999995) < 1e-15),
    'targets': (r'no rewrite applied for preprocess__Z8checksum',
                lambda o: all(0 < float(r[2]) < 1e-4 and float(r[3]) == 0 for r in rows(o))),
    'per_call': (r'^Applying solution for .*literal 1/2 binary64',
                 lambda o: all(r[1] == '0.5' and r[3] != r[4] and abs(float(r[3]) / float(r[4]) - 1) < 1e-6
                               for r in rows(o))),
    'quadratic': (r'^Applying solution for .*\(if\.f64',
                  lambda o: abs(float(rows(o)[-1][2]) - float(rows(o)[-1][3])) <
                  abs(float(rows(o)[-1][1]) - float(rows(o)[-1][3]))),
}


class Explorer:
    def __init__(self, url, out):
        self.url, self.out = url.rstrip('/'), out

    def get(self, path):
        req = urllib.request.Request(self.url + path, headers={'Accept': 'application/json'})
        return json.loads(urllib.request.urlopen(req, timeout=60).read())

    def compile(self, cid, source, args, execute=False, tag=None, views=None, tools=(), binary=False):
        body = {'source': source, 'lang': 'c++', 'options': {
            'userArguments': args, 'compilerOptions': views or {}, 'libraries': [],
            'tools': [{'id': t, 'args': []} for t in tools],
            'filters': {'execute': execute, 'intel': True, 'demangle': True, 'labels': True,
                        'directives': True, 'commentOnly': True, 'binary': binary}}}
        req = urllib.request.Request(f'{self.url}/api/compiler/{cid}/compile', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
        t0 = time.time()
        res = json.loads(urllib.request.urlopen(req, timeout=900).read())
        res['_seconds'] = time.time() - t0
        if self.out and tag:
            with open(os.path.join(self.out, tag + '.json'), 'w') as f:
                json.dump(res, f, indent=1)
        return res


IR = {'filterDebugInfo': True, 'filterIRMetadata': True, 'filterAttributes': True, 'filterComments': True,
      'noDiscardValueNames': True, 'demangle': True}
PIPELINE = {'filterDebugInfo': True, 'filterIRMetadata': False, 'fullModule': False, 'noDiscardValueNames': True,
            'demangle': True, 'libraryFunctions': True}
VIEWS = {'produceIr': IR, 'produceOptInfo': True, 'produceStackUsageInfo': True, 'produceAst': True,
         'producePp': {'filter-headers': True}, 'produceCfg': {'asm': True, 'ir': True}, 'produceOptPipeline': PIPELINE}
CAPS = ['supportsOptOutput', 'supportsIrView', 'optPipeline', 'supportsCfg', 'supportsAstView', 'supportsPpView',
        'supportsStackUsageOutput', 'supportsBinary', 'supportsExecute']
TOOLS = ['llvm-mca', 'nm']
EXPLAIN = ['poseidon-report', 'poseidon-source', 'poseidon-diff', 'poseidon-fpcore']
CHATTER = re.compile(r'regime-split candidate|dropping (array )?candidate|Herbie core .*status=[a-z]+-(start|target)')
BUSY = 'poseidon-ce: note: the explorer is busy; compiled as written, try again in a minute'
DECISION = re.compile(r'^(?:Applying solution for .*|\[poseidon\] .*(?:: tau=.*|no rewrite applied.*)|No solution found.*)$', re.M)


def views_ok(r, poseidon):
    got = {}
    ir = r.get('irOutput') or {}
    got['ir'] = any(l['text'].startswith('define ') for l in ir.get('asm', []))
    got['ir cfg'] = bool(ir.get('cfg'))
    got['cfg'] = bool(r.get('cfg'))
    got['opt remarks'] = bool(r.get('optOutput'))
    got['stack usage'] = bool(r.get('stackUsageOutput'))
    got['ast'] = any('FunctionDecl' in l['text'] for l in r.get('astOutput') or [])
    got['preprocessor'] = 'main' in (r.get('ppOutput') or {}).get('output', '')
    pipe = r.get('optPipelineOutput') or {}
    passes = [p for fn in (pipe.get('results') or {}).values() for p in fn]
    got['opt pipeline'] = not pipe.get('error') and len(passes) > 50
    if poseidon:
        got['poseidon pass'] = any('OptimizePass' in p['name'] and p['irChanged'] for p in passes)
    tools = {t['id']: t for t in r.get('tools') or []}
    mca = tools.get('llvm-mca', {})
    got['llvm-mca'] = mca.get('code') == 0 and 'Iterations:' in text(mca.get('stdout'))
    if poseidon:
        got['poseidon remarks'] = any(x['Pass'] == 'poseidon' for x in r.get('optOutput') or [])
        notes = [l['tag'] for l in r.get('stderr') or [] if l.get('tag') and '=poseidon]' in l['tag']['text']]
        got['remark notes'] = bool(notes) and all(t['severity'] == 1 for t in notes)
        want = {'poseidon-report': 'Poseidon site', 'poseidon-source': 'int main', 'poseidon-diff': 'example.cpp',
                'poseidon-fpcore': '(FPCore'}
        for k, w in want.items():
            t = tools.get(k, {})
            lines = (t.get('stdout') or []) + (t.get('stderr') or [])
            body = text(lines)
            if k == 'poseidon-diff' and 'left the source as written' in body:
                w = 'as written'
            got[k] = (t.get('code') == 0 and w in body and not re.search(r'/tmp/|compiler-explorer', body)
                      and not any(l.get('tag') and l['tag']['severity'] > 1 for l in lines))
        got['report markers'] = any(l.get('tag') for l in tools.get('poseidon-report', {}).get('stdout') or [])
    return [k for k, v in got.items() if not v], len(got)


def headline(d):
    lines = d.strip().splitlines() or ['(empty)']
    return ([l for l in lines if 'poseidon-ce:' in l or 'error' in l] or lines)[0][:160]


def text(lines):
    return ANSI.sub('', '\n'.join(l.get('text', '') for l in lines or []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default=os.environ.get('POSEIDON_EXPLORER_URL', 'http://localhost:10240'))
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out')
    ap.add_argument('--jobs', type=int, default=6)
    a = ap.parse_args()
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    ce = Explorer(a.url, a.out)
    failures = []

    def check(name, ok, detail, seconds=None):
        tag = f'{seconds:6.1f}s' if seconds is not None else '       '
        print(f'{"PASS" if ok else "FAIL"} {tag} {name}: {detail}', flush=True)
        if not ok:
            failures.append(name)

    comps = ce.get('/api/compilers/c++?fields=id,name,version,fullVersion,tools,' + ','.join(CAPS))
    pos = [c for c in comps if c['id'].endswith('-poseidon')]
    check('probe', bool(pos), ', '.join(f'{c["id"]} = {c["name"]}' for c in comps))
    if not pos:
        return 1
    cid = pos[0]['id']
    stock = next((c['id'] for c in comps if c['id'] == cid[:-len('-poseidon')]), None)
    print('       ' + pos[0].get('fullVersion', '').strip().replace('\n', '\n       '))
    for c in comps:
        want = TOOLS + (EXPLAIN if c['id'] == cid else [])
        missing = [k for k in CAPS if not c.get(k)] + [t for t in want if t not in (c.get('tools') or {})]
        extra = [t for t in EXPLAIN if c['id'] != cid and t in (c.get('tools') or {})]
        check(f'views and tools of {c["id"]}', not missing and not extra,
              f'missing {missing}, unexpected {extra}' if missing or extra else ', '.join(CAPS + want))

    for path in sorted(glob.glob(os.path.join(EXAMPLES, 'Poseidon_*.cpp'))):
        stem = os.path.basename(path)[:-4]
        r = ce.compile(cid, open(path).read(), '-O2', execute=True, tag=stem)
        e = r.get('execResult') or {}
        diag, out = text(r.get('stderr')), text(e.get('stdout'))
        decision = re.findall(r'^(?:Applying solution for .*|\[poseidon\] no rewrite applied.*|No solution found.*)$',
                              diag, re.M)
        want = EXPECT.get(stem[len('Poseidon_'):])
        ok = r.get('code') == 0 and e.get('code') == 0 and 'Poseidon' in out
        ok = ok and (want is None or (re.search(want[0], diag, re.M) and want[1](out)))
        check(f'example {stem}', bool(ok), '; '.join(d[:110] for d in decision) or diag[-300:], r['_seconds'])
        print('       ' + out.strip().replace('\n', '\n       '))
        noise = [l for l in diag.splitlines() if CHATTER.search(l) or '/tmp/' in l]
        check(f'compiler output {stem}', not noise, f'{len(diag.splitlines())} lines, no solver chatter or paths'
              if not noise else noise[0][:120])
        v = ce.compile(cid, open(path).read(), '-O2', tag=stem + '_views', views=VIEWS, tools=['llvm-mca'] + EXPLAIN)
        missing, n = views_ok(v, True)
        same = text(v.get('asm')) == text(r.get('asm')) and DECISION.findall(text(v.get('stderr'))) == DECISION.findall(diag)
        check(f'views {stem}', v.get('code') == 0 and not missing and same,
              f'{n} views, same assembly and decisions' if not missing and same else
              f'missing {missing}' if missing else 'assembly or decisions differ from the plain compile', v['_seconds'])
        b = ce.compile(cid, open(path).read(), '-O2', tag=stem + '_binary', tools=['nm'], binary=True)
        nm = next((t for t in b.get('tools') or [] if t['id'] == 'nm'), {})
        ok = b.get('code') == 0 and len(b.get('asm') or []) > 10 and nm.get('code') == 0
        ok = ok and re.search(r' T main$', text(nm.get('stdout')), re.M) is not None
        check(f'binary {stem}', ok, 'objdump, nm', b['_seconds'])

    src = open(os.path.join(EXAMPLES, 'Poseidon_cancellation.cpp')).read()
    if stock:
        v = ce.compile(stock, src, '-O2', tag='stock_views', views=VIEWS, tools=['llvm-mca'])
        missing, n = views_ok(v, False)
        check(f'views {stock}', v.get('code') == 0 and not missing, f'{n} views' if not missing else f'missing {missing}',
              v['_seconds'])
    for args in ('-O2', '-O2 -Rpass-missed=poseidon'):
        r = ce.compile(cid, src, args, tag='remarks' + str(len(args)))
        tags = [l['tag'] for l in r.get('stderr') or [] if l.get('tag')]
        notes = [t for t in tags if t['severity'] == 1 and t['text'].startswith('note: ')]
        check(f'remarks are notes ({args})', r.get('code') == 0 and bool(notes) and len(notes) == len(tags),
              f'{len(notes)} of {len(tags)} markers are notes: {notes[-1]["text"][:80] if notes else ""}', r['_seconds'])
    templates = ce.get('/api/siteTemplates').get('templates') or []
    ref = next((t['reference'] for t in templates if 'h:diff' in t['reference']), '')
    check('diff preset', f"compiler:'{cid}'" in ref and (stock is None or f"compiler:'{stock}'" in ref),
          next((t['name'] for t in templates if t['reference'] == ref), 'no template with a diff pane'))
    for args, want in [('-O2 -poseidon-tau=1e-3', 'tau=1.000000e-03'),
                       ('-O2 -mllvm -poseidon-tau=1e-12', 'tau=1.000000e-12')]:
        r = ce.compile(cid, src, args, tag='tau' + want[-3:])
        check(f'flag {args}', r.get('code') == 0 and want in text(r.get('stderr')), want, r['_seconds'])
    r = ce.compile(cid, src, BUDGET, execute=True, tag='budget')
    d, out = text(r.get('stderr')), text((r.get('execResult') or {}).get('stdout'))
    check(f'flag {BUDGET}', r.get('code') == 0 and re.search(r'^Applying solution for .*\(fma', d, re.M)
          is not None and rows(out)[-1][2] != '0.5', 'the fma series point of the Pareto frontier', r['_seconds'])
    quad = open(os.path.join(EXAMPLES, 'Poseidon_quadratic.cpp')).read()
    r = ce.compile(cid, quad, '-O2', tag='quadratic_source', tools=['poseidon-source'])
    t = next((x for x in r.get('tools') or [] if x['id'] == 'poseidon-source'), {})
    check('regime split as a ternary', r.get('code') == 0 and ' ? ' in text(t.get('stdout')),
          'poseidon-source shows the branch as ?:', r['_seconds'])
    r = ce.compile(cid, src, '-O2 -poseidon-enable-herbie=0', tag='noherbie')
    d = text(r.get('stderr'))
    check('flag -poseidon-enable-herbie=0', r.get('code') == 0 and 'Herbie' not in d and 'tau=' in d,
          'no Herbie lines', r['_seconds'])
    r = ce.compile(cid, src, '-O2 -poseidon-print', tag='print')
    d = text(r.get('stderr'))
    check('flag -poseidon-print', r.get('code') == 0 and 'Herbie output:' in d, f'{len(d.splitlines())} lines',
          r['_seconds'])

    for name, source, want in [('no site', NO_SITE, 'no site was profiled'),
                               ('syntax error', SYNTAX, "use of undeclared identifier 'undeclared'"),
                               ('profiling run crashes', CRASH, 'profiling run')]:
        r = ce.compile(cid, source, '-O2', tag=name.replace(' ', '_'))
        d = text(r.get('stderr'))
        check(name, r.get('code') != 0 and want in d, headline(d), r['_seconds'])

    r1 = ce.compile(cid, SITE_ONLY, '-O2 -poseidon-tau=1e-4', tag='cache1')
    r2 = ce.compile(cid, SITE_ONLY, '-O2 -poseidon-tau=1e-4', tag='cache2')
    check('cache', r1.get('code') == 0 and r2.get('retreivedFromCache') is True,
          f'first {r1["_seconds"]:.1f}s, repeat {r2["_seconds"]:.2f}s (retreivedFromCache={r2.get("retreivedFromCache")})')

    if a.quick:
        return 1 if failures else 0

    r = ce.compile(cid, SPIN, '-O2', tag='spin')
    d = text(r.get('stderr'))
    check('profiling run never exits', r.get('code') != 0 and 'profiling run' in d,
          headline(d), r['_seconds'])
    fresh = lambda k: SITE_ONLY.replace('(x[i] * x[i])', f'(x[i] * x[i]) * {1 + (time.time_ns() % 999983 + k) / 1e7!r}')
    quat = open(os.path.join(EXAMPLES, 'Poseidon_quaternion.cpp')).read()
    quat = quat.replace('std::sin(0.5 * angle)', f'std::sin({0.5 + time.time_ns() % 999983 / 1e9!r} * angle)')
    r = ce.compile(cid, quat, '-O2 -poseidon-herbie-timeout=1', tag='herbie_timeout')
    d = text(r.get('stderr'))
    check('Herbie timeout', r.get('code') == 0 and 'status=timeout' in d, 'status=timeout reported, compile succeeds',
          r['_seconds'])

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        rs = list(pool.map(lambda k: ce.compile(cid, fresh(k + 1), '-O2', True, f'concurrent_{k}'), range(a.jobs)))
    wall = time.time() - t0
    good = [x for x in rs if x.get('code') == 0 and (x.get('execResult') or {}).get('code') == 0]
    busy = [x for x in good if BUSY in text(x.get('stderr'))]
    searched = [x for x in good if DECISION.search(text(x.get('stderr')))]
    check(f'{a.jobs} concurrent compile+execute, each a new Herbie search', len(good) == len(rs) and busy and searched
          and len(busy) + len(searched) == len(rs),
          f'{len(good)}/{len(rs)} ok: {len(searched)} searched, {len(busy)} compiled as written because the explorer '
          f'was busy (slowest {max((x["_seconds"] for x in busy), default=0):.1f}s); wall {wall:.1f}s, '
          f'slowest {max(x["_seconds"] for x in rs):.1f}s')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
