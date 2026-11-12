"""usage: pick_chains.py <dl dir> <existing.txt> <builtin.txt> <out dir>
Picks new training chains: one per RCSB 30%-identity cluster that holds no chain already used for training, testing or
the lab's built-in proteins, and that has an alignment in OpenProteinSet. Writes chains_full.txt (40-300 residues) and
chains_long.txt (301-1500): name, OpenProteinSet folder, length, resolution, sequence."""
import json, os, random, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
dl, existing_f, builtin_f, out = sys.argv[1:5]

def gql(query):
    r = urllib.request.Request('https://data.rcsb.org/graphql', json.dumps({'query': query}).encode(), {'Content-Type': 'application/json'})
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(r, timeout=120))['data']
        except Exception as e:
            err = e
    raise err

def batches(xs, n):
    xs = list(xs)
    return [xs[i:i + n] for i in range(0, len(xs), n)]

ops = set(open(os.path.join(dl, 'ops_chains.txt')).read().split())
rep = {}                                  # chain -> a chain with the same sequence that has a folder
for line in open(os.path.join(dl, 'dups.txt')):
    g = line.split()
    have = [c for c in g if c in ops]
    if have:
        for c in g: rep[c] = have[0]
for c in ops: rep.setdefault(c, c)
ops_entries = {c.split('_')[0].upper() for c in rep}

cluster_of, clusters = {}, []
for i, line in enumerate(open(os.path.join(dl, 'clusters30.txt'))):
    m = line.split(); clusters.append(m)
    for e in m: cluster_of[e] = i

existing = [l.strip() for l in open(existing_f) if l.strip()]
builtin = {l.strip().upper() for l in open(builtin_f) if l.strip()}
want = {}
for name in existing:
    e, _, c = name.partition('_'); want.setdefault(e.upper(), set()).add(c)
for e in builtin: want[e] = None           # every protein in a built-in entry is off limits
def entry_entities(ids):
    q = '{ entries(entry_ids: %s) { rcsb_id polymer_entities { rcsb_id rcsb_polymer_entity_container_identifiers { auth_asym_ids } } } }' % json.dumps(ids)
    return gql(q)['entries'] or []
forbidden, unmapped = set(), 0
with ThreadPoolExecutor(4) as ex:
    for res in ex.map(entry_entities, batches(sorted(want), 300)):
        for en in res:
            chains, ents = want[en['rcsb_id']], en['polymer_entities'] or []
            hit = [p['rcsb_id'] for p in ents if chains is None or chains & set(p['rcsb_polymer_entity_container_identifiers']['auth_asym_ids'] or [])]
            if not hit: hit, unmapped = [p['rcsb_id'] for p in ents], unmapped + 1
            forbidden.update(hit)
bad = {cluster_of[e] for e in forbidden if e in cluster_of}
print(f'{len(existing)} existing chains + {len(builtin)} built-in entries -> {len(forbidden)} entities, {len(bad)} clusters off limits ({unmapped} entries not matched by chain, all their proteins excluded)', flush=True)

ask = {}
for i, m in enumerate(clusters):
    if i in bad: continue
    for e in [e for e in m if e.split('_')[0] in ops_entries][:3]: ask[e] = i
print(f'{len(set(ask.values()))} clusters have a member in OpenProteinSet; asking RCSB about {len(ask)} entities', flush=True)
def entities(ids):
    q = '''{ polymer_entities(entity_ids: %s) { rcsb_id entity_poly { rcsb_sample_sequence_length pdbx_seq_one_letter_code_can rcsb_entity_polymer_type }
      rcsb_polymer_entity_container_identifiers { auth_asym_ids } entry { rcsb_entry_info { resolution_combined experimental_method } } } }''' % json.dumps(ids)
    return gql(q)['polymer_entities'] or []
best, used = {}, set()
with ThreadPoolExecutor(4) as ex:
    for res in ex.map(entities, batches(sorted(ask), 400)):
        for p in res:
            ep, info = p['entity_poly'], p['entry']['rcsb_entry_info']
            seq, res_ = (ep['pdbx_seq_one_letter_code_can'] or '').replace('\n', ''), (info['resolution_combined'] or [99])[0]
            if ep['rcsb_entity_polymer_type'] != 'Protein' or not 40 <= len(seq) <= 1500 or info['experimental_method'] not in ('X-ray', 'EM') or res_ > 3.0:
                continue
            if sum(ch not in 'ARNDCQEGHILKMFPSTWYV' for ch in seq) > 0.05 * len(seq): continue
            entry = p['rcsb_id'].split('_')[0]
            for c in p['rcsb_polymer_entity_container_identifiers']['auth_asym_ids'] or []:
                key = f'{entry.lower()}_{c}'
                if key in rep and c.isalnum():
                    cand = (res_, f'{entry}_{c}', rep[key], len(seq), seq)
                    k = ask[p['rcsb_id']]
                    if k not in best or cand < best[k]: best[k] = cand
                    break
rows = []
for res_, name, folder, n, seq in sorted(best.values(), key=lambda t: t[1]):
    if name.upper() in used: continue      # names must stay distinct on a case-insensitive disk
    used.add(name.upper()); rows.append((name, folder, n, res_, seq))
random.Random(7).shuffle(rows)
full, long_ = [r for r in rows if r[2] <= 300], [r for r in rows if r[2] > 300]
print(f'{len(rows)} candidate chains: {len(full)} of 40-300 residues, {len(long_)} of 301-1500')
for fn, rs in (('chains_full.txt', full), ('chains_long.txt', long_)):
    open(os.path.join(out, fn), 'w').write(''.join(f'{a} {b} {c} {d} {e}\n' for a, b, c, d, e in rs))
