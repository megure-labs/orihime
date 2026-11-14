// GitHub-only routing. Never executes, builds, checks out, or publishes private code.
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

export const PAIRS = Object.freeze({'megure-labs/orihime':'megure-labs/orihime-dev','orikata-bio/hikoboshi':'orikata-bio/hikoboshi-dev'});
const OWNER = 'caseysm';
const SHA = /^[0-9a-f]{40}$/;
const protectedPath = /^(?:\.github(?:\/|$)|\.agents(?:\/|$)|\.claude(?:\/|$)|AGENTS\.md$|CLAUDE\.md$|CONTRIBUTING\.md$|\.git(?:\/|$))/i;
export function developmentBody(repo,pr,changes) {
  const metadata={source:repo,pr:pr.number,sha:pr.head.sha,base:pr.base.sha,changes};
  const original=(pr.body||'').slice(0,8000);
  const fence='`'.repeat(Math.max(2,...(original.match(/`+/g)||[]).map(s=>s.length))+1);
  const body=`## Summary
Port an owner-approved external contribution.

## Human contribution
${OWNER} approved source commit ${pr.head.sha}; original submission account: ${pr.user.login}.

## Agent contribution
TODO

## External sources
${pr.html_url} at ${pr.head.sha}. Preserve its license and notices.

## Validation
Product CI and exact-head maintainer approval are required here before promotion.

## Import provenance
GitHub App copied immutable blobs without executing contributor code. Complete the Agent contribution section with the original contribution's actual human/agent facts before merging; do not infer None. Full records stay in this private PR.

<!-- contribution-router-v1 ${JSON.stringify(metadata)} -->

### Original public submission (reference text; first 8000 characters)
${fence}markdown
${original}
${fence}
`;
  if(body.length>60000)throw Error('Intake metadata exceeds the PR body bound; manual port required');
  return body;
}
export function releaseBody(source,sha) {
  return `## Summary
Validated release candidate for ${source.html_url}.

## Validation
The exact approved source ${sha} passed private development product CI and required checks. Public release checks and exact-head maintainer approval remain required.

## Sources and credits
Original public submission: ${source.html_url}. Original contributor: ${source.user.login}. Preserve source/license notices and author credit. Full internal contribution records remain private.
`;
}
export function validateSource(repo, pr) {
  if (!PAIRS[repo] || pr.base?.repo?.full_name !== repo || pr.base.ref !== 'main' || pr.state !== 'open') throw Error('Source must be an open main PR in an explicit release/dev pair');
  if (!SHA.test(pr.head?.sha || '') || !SHA.test(pr.base?.sha || '')) throw Error('Source revision must be immutable');
}
export function ownerApproved(reviews, sha) {
  // A later dismissal/change request invalidates a former approval.
  const latest = reviews.filter(r => r.user?.login === OWNER && ['APPROVED', 'CHANGES_REQUESTED', 'DISMISSED'].includes(r.state)).sort((a,b) => a.id-b.id).at(-1);
  return latest?.state === 'APPROVED' && latest.commit_id === sha;
}
export function planImport(files, publicBase, publicHead, devBase) {
  if (!files.length || files.length > 100) throw Error('Expected 1–100 explicitly changed files');
  const maps = [publicBase, publicHead, devBase].map(entries => new Map(entries.map(e => [e.path,e])));
  const changes=[];
  for (const file of files) {
    const path=file.filename;
    if (!path || path.startsWith('/') || path.split('/').some(p => !p || p==='.' || p==='..' || p.toLowerCase()==='.git') || /[\u0000-\u001f\u007f]/.test(path) || protectedPath.test(path)) throw Error('Privileged or unsafe path requires a manual port: '+path);
    if (!['added','modified','removed'].includes(file.status)) throw Error('Renames and other complex changes require a manual port');
    const [before,after,dev]=maps.map(m => m.get(path));
    for (const entry of [before,after,dev].filter(Boolean)) {
      if (entry.type!=='blob' || !['100644','100755'].includes(entry.mode)) throw Error('Links/submodules require a manual port');
      if ((entry.size ?? 0)>1048576) throw Error('Large files require a manual port');
    }
    if ((dev?.sha ?? null)!==(before?.sha ?? null) || (dev?.mode ?? null)!==(before?.mode ?? null)) throw Error('Development source diverged; manual conflict resolution required: '+path);
    if (file.status==='removed' ? (!before || after) : !after) throw Error('Source file status/tree mismatch');
    changes.push({path,before:before?.sha ?? null,after:after?.sha ?? null,mode:after?.mode ?? before.mode});
  }
  return changes;
}
export function validationPassed(checks, statuses, required=[]) {
  if (checks.some(c => c.status!=='completed' || !['success','neutral','skipped'].includes(c.conclusion))) return false;
  if (statuses.some(s => s.state!=='success')) return false;
  const completed=new Set([...checks.filter(c=>c.conclusion==='success').map(c=>c.name),...statuses.filter(s=>s.state==='success').map(s=>s.context)]);
  if (!required.every(name=>completed.has(name))) return false;
  // Metadata checks and bot review alone cannot establish product validation.
  return [...completed].some(name=>!/(?:contribution disclosure|validate contribution disclosure|coderabbit|greptile)/i.test(name));
}

class GitHub {
  constructor(token){if(!token)throw Error('Scoped router GitHub App token is not configured');this.token=token;}
  async request(path,method='GET',body){
    const response=await fetch('https://api.github.com/'+path,{method,headers:{Authorization:'Bearer '+this.token,Accept:'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'},body:body===undefined?undefined:JSON.stringify(body)});
    if(!response.ok)throw Error(`GitHub ${method} ${path.split('?')[0]} returned ${response.status}`);
    return response.status===204?null:response.json();
  }
  async pages(path){const all=[];for(let page=1;page<=100;page++){const rows=await this.request(path+(path.includes('?')?'&':'?')+'per_page=100&page='+page);if(!Array.isArray(rows))throw Error('Invalid paginated response');all.push(...rows);if(rows.length<100)return all;}throw Error('Pagination limit exceeded');}
  async tree(repo,sha){const result=await this.request(`repos/${repo}/git/trees/${sha}?recursive=1`);if(result.truncated)throw Error('Truncated tree requires a manual port');return result.tree;}
}

export async function route(api,repo,number,actor){
  if(actor!==OWNER)throw Error('Only the repository owner may request privileged routing');
  const pr=await api.request(`repos/${repo}/pulls/${number}`);validateSource(repo,pr);
  if(!ownerApproved(await api.pages(`repos/${repo}/pulls/${number}/reviews`),pr.head.sha))throw Error('Owner approval is required for this exact source SHA');
  const dest=PAIRS[repo];const destInfo=await api.request(`repos/${dest}`);
  if(!destInfo.private || destInfo.default_branch!=='main')throw Error('Destination must be private with a main default');
  const devRef=await api.request(`repos/${dest}/git/ref/heads/main`);
  const [files,before,after,dev]=await Promise.all([api.pages(`repos/${repo}/pulls/${number}/files`),api.tree(repo,pr.base.sha),api.tree(repo,pr.head.sha),api.tree(dest,devRef.object.sha)]);
  const changes=planImport(files,before,after,dev);
  const branch=`contribution/${number}-${pr.head.sha.slice(0,16)}`;
  const prior=await api.pages(`repos/${dest}/pulls?state=all&head=${dest.split('/')[0]}:${branch}`);
  if(prior.length)return {state:'already-routed',number:prior[0].number};
  // Re-read approval and head immediately before privileged writes.
  const fresh=await api.request(`repos/${repo}/pulls/${number}`);
  if(fresh.head.sha!==pr.head.sha || fresh.base.sha!==pr.base.sha || fresh.state!=='open' || !ownerApproved(await api.pages(`repos/${repo}/pulls/${number}/reviews`),pr.head.sha))throw Error('Source changed or approval withdrawn during routing');
  const body=developmentBody(repo,pr,changes);
  const entries=[];
  for(const change of changes){
    if(change.after===null){entries.push({path:change.path,mode:change.mode,type:'blob',sha:null});continue;}
    const blob=await api.request(`repos/${repo}/git/blobs/${change.after}`);
    if(blob.size>1048576 || blob.encoding!=='base64')throw Error('Unsupported source blob');
    const copied=await api.request(`repos/${dest}/git/blobs`,'POST',{encoding:'base64',content:blob.content});
    if(copied.sha!==change.after)throw Error('Blob identity changed during copy');
    entries.push({path:change.path,mode:change.mode,type:'blob',sha:copied.sha});
  }
  const baseCommit=await api.request(`repos/${dest}/git/commits/${devRef.object.sha}`);
  const tree=await api.request(`repos/${dest}/git/trees`,'POST',{base_tree:baseCommit.tree.sha,tree:entries});
  const commit=await api.request(`repos/${dest}/git/commits`,'POST',{message:`Port approved contribution ${repo}#${number}\n\nSource-head: ${pr.head.sha}\nSource-base: ${pr.base.sha}\nOriginal-author: ${pr.user.login}`,tree:tree.sha,parents:[devRef.object.sha]});
  await api.request(`repos/${dest}/git/refs`,'POST',{ref:'refs/heads/'+branch,sha:commit.sha});
  const created=await api.request(`repos/${dest}/pulls`,'POST',{title:`Port ${repo}#${number}: ${pr.title}`.slice(0,240),head:branch,base:'main',body});
  await api.request(`repos/${repo}/statuses/${pr.head.sha}`,'POST',{state:'pending',context:'Development validation',description:'Owner-approved contribution is awaiting development validation'});
  return {state:'routed',number:created.number};
}

export async function promote(api,repo,number,appSlug){
  const pr=await api.request(`repos/${repo}/pulls/${number}`);
  if(!appSlug || pr.user?.login!==appSlug+'[bot]' || !pr.merged || pr.base.ref!=='main')throw Error('Only a merged router-authored development PR can be promoted');
  const match=/<!-- contribution-router-v1 (.+) -->/.exec(pr.body||'');
  if(!match)throw Error('Missing immutable source binding');
  const m=JSON.parse(match[1]);
  if(PAIRS[m.source]!==repo || !SHA.test(m.sha)||!SHA.test(m.base))throw Error('Invalid paired-repository binding');
  const source=await api.request(`repos/${m.source}/pulls/${m.pr}`);validateSource(m.source,source);
  if(source.head.sha!==m.sha || source.base.sha!==m.base)throw Error('Public source or base changed; reroute and validate the new revision');
  if(!ownerApproved(await api.pages(`repos/${repo}/pulls/${number}/reviews`),pr.head.sha) || !ownerApproved(await api.pages(`repos/${m.source}/pulls/${m.pr}/reviews`),m.sha))throw Error('Current owner approvals are required on both exact revisions');
  const commits=await api.pages(`repos/${repo}/pulls/${number}/commits`);
  const first=commits[0];
  if(!first?.commit?.message.startsWith(`Port approved contribution ${m.source}#${m.pr}\n\nSource-head: ${m.sha}\nSource-base: ${m.base}\n`))throw Error('Mutable PR metadata does not match the immutable imported commit');
  const changes=planImport(await api.pages(`repos/${m.source}/pulls/${m.pr}/files`),await api.tree(m.source,m.base),await api.tree(m.source,m.sha),await api.tree(repo,first.parents[0].sha));
  const checkResponse=await api.request(`repos/${repo}/commits/${pr.head.sha}/check-runs?per_page=100`);
  if(checkResponse.total_count>100)throw Error('Too many checks; manual verification required');
  const statuses=(await api.request(`repos/${repo}/commits/${pr.head.sha}/status?per_page=100`)).statuses;
  const rules=await api.request(`repos/${repo}/rules/branches/main`);
  const required=rules.filter(r=>r.type==='required_status_checks').flatMap(r=>r.parameters.required_status_checks.map(c=>c.context));
  if(!validationPassed(checkResponse.check_runs,statuses,required))throw Error('Completed product CI and every required check must pass');
  const finalTree=new Map((await api.tree(repo,pr.merge_commit_sha)).map(e=>[e.path,e]));
  const approvedTree=new Map((await api.tree(repo,pr.head.sha)).map(e=>[e.path,e]));
  for(const c of changes){
    const expectedMode=c.after===null?null:c.mode;
    if([finalTree,approvedTree].some(tree=>(tree.get(c.path)?.sha??null)!==c.after || (tree.get(c.path)?.mode??null)!==expectedMode))throw Error('Development fixes differ from public source; a maintainer must update the public PR before promotion');
  }
  // Export only the original public tree; no private repository content leaves.
  const branch=`validated/${m.pr}-${m.sha.slice(0,16)}`;
  const existing=await api.pages(`repos/${m.source}/pulls?state=all&head=${m.source.split('/')[0]}:${branch}`);
  if(existing.length)return {state:'already-promoted',number:existing[0].number};
  await api.request(`repos/${m.source}/git/refs`,'POST',{ref:'refs/heads/'+branch,sha:m.sha});
  const release=await api.request(`repos/${m.source}/pulls`,'POST',{title:`Validated contribution #${m.pr}: ${source.title}`.slice(0,240),head:branch,base:'main',body:releaseBody(source,m.sha)});
  await api.request(`repos/${m.source}/statuses/${m.sha}`,'POST',{state:'success',context:'Development validation',description:'Exact approved contribution passed development validation'});
  return {state:'release-pr-opened',number:release.number};
}

if(process.argv[1] && import.meta.url===pathToFileURL(process.argv[1]).href){
  const event=JSON.parse(await readFile(process.env.GITHUB_EVENT_PATH,'utf8'));
  const api=new GitHub(process.env.CONTRIBUTION_ROUTER_TOKEN);
  const repo=process.env.GITHUB_REPOSITORY;
  const result=process.env.ROUTER_MODE==='promote'
    ?await promote(api,repo,event.pull_request.number,process.env.CONTRIBUTION_ROUTER_APP_SLUG)
    :await route(api,repo,event.pull_request.number,event.sender.login);
  console.log(JSON.stringify(result));
}
