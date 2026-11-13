import {execFileSync} from 'node:child_process';

// Parse GitHub's sanitized body_html, never Markdown or candidate source.
// The static program receives untrusted HTML only through stdin; no shell/eval.
export const htmlParser = String.raw`
import json, sys, unicodedata
from html.parser import HTMLParser

names = ['Summary', 'Human contribution', 'Agent contribution', 'External sources', 'Validation']
prompts = {
    'describe the change and why it is needed.',
    'list the commands and environments used to test it.',
    'describe the problem and resulting behavior.',
    'identify the accountable human and their design, coding, review and validation work.',
    'list each agent/tool/model when known, tasks, autonomy, and human verification. state none if unused.',
    'list code/reference urls, exact revisions, licenses, affected files and how sources were used. state none if applicable.',
    'list exact commands, environments, results, and meaningful untested cases.',
}
voids = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

def clean(value):
    return ' '.join(''.join(c for c in value if c.isspace() or unicodedata.category(c)[0] != 'C').split())

class Sections(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections = {}
        self.active = None
        self.heading = None
        self.ignored = []

    def handle_starttag(self, tag, attrs):
        if self.ignored:
            if tag not in voids:
                self.ignored.append(tag)
            return
        if tag in {'script', 'style', 'template', 'head'} or any(k == 'hidden' for k, v in attrs):
            if tag not in voids:
                self.ignored.append(tag)
            return
        if tag == 'h2':
            self.active = None
            self.heading = []
        elif tag == 'h1':
            self.active = None
        elif tag in {'p', 'div', 'br', 'li', 'pre', 'blockquote', 'tr', 'td', 'th', 'h3', 'h4', 'h5', 'h6'}:
            self.handle_data(' ')

    def handle_endtag(self, tag):
        if self.ignored:
            if tag in self.ignored:
                del self.ignored[len(self.ignored) - 1 - self.ignored[::-1].index(tag):]
            return
        if tag == 'h2' and self.heading is not None:
            self.active = clean(''.join(self.heading)).casefold()
            self.sections.setdefault(self.active, [])
            self.heading = None
        elif tag in {'p', 'div', 'li', 'pre', 'blockquote', 'tr', 'td', 'th', 'h3', 'h4', 'h5', 'h6'}:
            self.handle_data(' ')

    def handle_data(self, data):
        if self.ignored:
            return
        if self.heading is not None:
            self.heading.append(data)
        elif self.active is not None:
            self.sections[self.active].append(data)

parser = Sections()
parser.feed(sys.stdin.read())
parser.close()
missing = []
for name in names:
    value = clean(''.join(parser.sections.get(name.casefold(), [])))
    folded = value.casefold()
    if not value or folded.rstrip('.') in {'todo', 'tbd', 'n/a'} or folded in prompts or (folded.startswith('[replace') and folded.rstrip('.').endswith(']')):
        missing.append(name)
print(json.dumps(missing))
`;

export function missingSections(html) {
  if (typeof html !== 'string' || Buffer.byteLength(html) > 1024 * 1024) {
    throw Error('Missing or oversized GitHub-rendered disclosure body.');
  }
  return JSON.parse(execFileSync('python3', ['-I', '-c', htmlParser], {
    input: html, encoding: 'utf8', timeout: 10000, maxBuffer: 65536,
  }));
}

export function incompletePulls(pulls, head) {
  const unique = new Map(pulls.filter(p => p.state === 'open' && p.head?.sha === head).map(p => [p.number, p]));
  return [...unique.values()].map(p => ({number: p.number, missing: missingSections(p.body_html === null ? '' : p.body_html)}))
    .filter(p => p.missing.length).sort((a, b) => a.number - b.number);
}

export async function openPulls(github, owner, repo) {
  const rows = [];
  for (let page = 1; page <= 10; page++) {
    // Commit association omits open companions once a SHA is on main.
    const {data, headers} = await github.rest.pulls.list({owner, repo, state: 'open', per_page: 100, page,
      headers: {accept: 'application/vnd.github.full+json'}});
    rows.push(...data);
    if (!headers.link?.includes('rel="next"')) return rows;
  }
  throw Error('Too many open PRs; disclosure validation requires maintainer inspection.');
}

export async function checkDisclosure(github, context, core) {
  const {owner, repo} = context.repo;
  const heads = new Set(), failures = [];
  const target_url = `${context.serverUrl}/${owner}/${repo}/actions/runs/${context.runId}`;
  const write = (sha, state, description) => github.rest.repos.createCommitStatus({owner, repo, sha, state,
    context: 'Contribution disclosure', description, target_url});
  const pending = async sha => {
    if (!/^[0-9a-f]{40}$/.test(sha || '') || heads.has(sha)) return;
    heads.add(sha);
    try { await write(sha, 'pending', 'Revalidating current PR disclosures'); }
    catch (error) { failures.push(error); }
  };
  // Invalidate every event/current head before lookups can fail. A moved PR
  // also revalidates companions remaining at its previous SHA. Unrelated heads
  // are not rewritten on every event, preserving GitHub's per-SHA status quota.
  await pending(context.payload?.pull_request?.head?.sha);
  await pending(context.payload?.before);
  if (failures.length) throw failures[0];
  const {data: current} = await github.rest.pulls.get({owner, repo, pull_number: context.issue.number,
    headers: {accept: 'application/vnd.github.full+json'}});
  await pending(current.head?.sha);
  if (failures.length) throw failures[0];
  const related = await openPulls(github, owner, repo);
  const pulls = [...related.filter(p => p.number !== current.number), current];
  for (const sha of heads) {
    try {
      const incomplete = incompletePulls(pulls, sha);
      if (sha === current.head?.sha && incomplete.length) {
        core.setFailed(incomplete.map(p => '#' + p.number + ': ' + p.missing.join(', ')).join('\n'));
      }
      await write(sha, incomplete.length ? 'failure' : 'success', incomplete.length
        ? ('Complete disclosures in PRs: ' + incomplete.map(p => '#' + p.number).join(', ')).slice(0, 140)
        : 'All open PRs at this head disclose authorship, sources and validation');
    } catch (error) { failures.push(error); }
  }
  if (failures.length) throw failures[0];
}

export function workflow() {
  const script = "const {execFileSync} = require('node:child_process');\nconst htmlParser = " + JSON.stringify(htmlParser) + ';\n' +
    [missingSections, incompletePulls, openPulls, checkDisclosure].map(fn => fn.toString()).join('\n') + '\nawait checkDisclosure(github,context,core);';
  return `# Generated from the first-party contribution-policy.mjs; no candidate checkout.
name: Contribution policy
on:
  pull_request_target:
    types: [opened, edited, synchronize, reopened, ready_for_review, closed]
permissions:
  contents: read
  pull-requests: read
  statuses: write
concurrency:
  group: contribution-policy-\${{ github.repository }}
  cancel-in-progress: false
  queue: max
jobs:
  disclosure:
    name: Validate contribution disclosure
    runs-on: ubuntu-24.04
    timeout-minutes: 5
    steps:
      - uses: actions/github-script@ed597411d8f924073f98dfc5c65a23a2325f34cd # v8
        with:
          script: |
${script.split('\n').map(line => '            ' + line).join('\n')}
`;
}
