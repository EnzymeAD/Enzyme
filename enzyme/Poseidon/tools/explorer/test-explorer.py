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
template <typename R, typename... T> R __poseidon_fp_optimize(void *, T...);
extern int enzyme_dup;
__attribute__((noinline)) void kernel(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i) y[i] = (1.0 - std::cos(x[i])) / (x[i] * x[i]);
}
'''
SITE = '''static double x[64], y[64], dx[64], dy[64];
  for (int i = 0; i < 64; ++i) { x[i] = 1e-6 * (i + 1); dy[i] = 1.0; }
  __poseidon_fp_optimize<void>((void *)kernel, enzyme_dup, x, dx, enzyme_dup, y, dy, 64);
  printf("%g\\n", y[0]);'''
NO_SITE = HEADER + 'int main() { double x[1] = {1e-6}, y[1]; kernel(x, y, 1); printf("%g\\n", y[0]); }\n'
SYNTAX = HEADER + 'int main() { return undeclared; }\n'
CRASH = HEADER + 'int main() {\n  ' + SITE + '\n  abort();\n}\n'
SPIN = HEADER + 'int main() {\n  ' + SITE + '\n  for (volatile int k = 0;; ++k) {}\n}\n'
SITE_ONLY = HEADER + 'int main() {\n  ' + SITE + '\n}\n'


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
}


class Explorer:
    def __init__(self, url, out):
        self.url, self.out = url.rstrip('/'), out

    def get(self, path):
        req = urllib.request.Request(self.url + path, headers={'Accept': 'application/json'})
        return json.loads(urllib.request.urlopen(req, timeout=60).read())

    def compile(self, cid, source, args, execute=False, tag=None):
        body = {'source': source, 'lang': 'c++', 'options': {
            'userArguments': args, 'compilerOptions': {}, 'tools': [], 'libraries': [],
            'filters': {'execute': execute, 'intel': True, 'demangle': True, 'labels': True,
                        'directives': True, 'commentOnly': True}}}
        req = urllib.request.Request(f'{self.url}/api/compiler/{cid}/compile', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
        t0 = time.time()
        res = json.loads(urllib.request.urlopen(req, timeout=900).read())
        res['_seconds'] = time.time() - t0
        if self.out and tag:
            with open(os.path.join(self.out, tag + '.json'), 'w') as f:
                json.dump(res, f, indent=1)
        return res


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

    comps = ce.get('/api/compilers/c++?fields=id,name,version,fullVersion')
    pos = [c for c in comps if c['id'].endswith('-poseidon')]
    check('probe', bool(pos), ', '.join(f'{c["id"]} = {c["name"]}' for c in comps))
    if not pos:
        return 1
    cid = pos[0]['id']
    print('       ' + pos[0].get('fullVersion', '').strip().replace('\n', '\n       '))

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

    src = open(os.path.join(EXAMPLES, 'Poseidon_cancellation.cpp')).read()
    for args, want in [('-O2 -poseidon-tau=1e-3', 'tau=1.000000e-03'),
                       ('-O2 -mllvm -poseidon-tau=1e-12', 'tau=1.000000e-12')]:
        r = ce.compile(cid, src, args, tag='tau' + want[-3:])
        check(f'flag {args}', r.get('code') == 0 and want in text(r.get('stderr')), want, r['_seconds'])
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
    r = ce.compile(cid, fresh(0), '-O2 -poseidon-herbie-timeout=1', tag='herbie_timeout')
    d = text(r.get('stderr'))
    check('Herbie timeout', r.get('code') == 0 and 'status=timeout' in d, 'status=timeout reported, compile succeeds',
          r['_seconds'])

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        rs = list(pool.map(lambda k: ce.compile(cid, fresh(k + 1), '-O2', True, f'concurrent_{k}'), range(a.jobs)))
    wall = time.time() - t0
    good = [x for x in rs if x.get('code') == 0 and (x.get('execResult') or {}).get('code') == 0]
    check(f'{a.jobs} concurrent compile+execute, each a new Herbie search', len(good) == len(rs),
          f'{len(good)}/{len(rs)} ok, wall {wall:.1f}s, slowest {max(x["_seconds"] for x in rs):.1f}s')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
