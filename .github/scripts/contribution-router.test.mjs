import test from 'node:test';
import assert from 'node:assert/strict';
import {PAIRS,PRODUCT_CHECKS,planImport,ownerApproved,validationPassed,validateSource,route,promote,developmentBody,releaseBody} from './contribution-router.mjs';
const EXPECTED_PAIRS = Object.freeze({"megure-labs/orihime":"megure-labs/orihime-dev"});
const PUBLIC=Object.keys(PAIRS)[0];
const old={path:'src/example.rs',sha:'a'.repeat(40),mode:'100644',type:'blob',size:10};
const next={...old,sha:'b'.repeat(40)};
const file={filename:old.path,status:'modified'};
test('copies a change only when public and private base blobs match',()=>{
  assert.equal(planImport([file],[old],[next],[old])[0].after,next.sha);
  assert.throws(()=>planImport([file],[old],[next],[next]),/diverged/);
});
test('added and removed files preserve expected existence',()=>{
  assert.equal(planImport([{...file,status:'added'}],[],[next],[]).length,1);
  assert.equal(planImport([{...file,status:'removed'}],[old],[],[old])[0].after,null);
  assert.throws(()=>planImport([{...file,status:'removed'}],[old],[next],[old]),/mismatch/);
});
for(const filename of ['../secret','a/../secret','a/.git/config','/absolute','.github/workflows/run.yml','.claude/hooks.json','AGENTS.md','a\u0000b','docs/provenance-policy.md','PROVENANCE_POLICY.md','CLEAN_ROOM.md','LICENSE'])test('rejects unsafe path '+JSON.stringify(filename),()=>assert.throws(()=>planImport([{...file,filename}],[],[],[]),/unsafe/));
test('rejects renames, links, large inputs and truncated batches',()=>{
  assert.throws(()=>planImport([{...file,status:'renamed'}],[old],[next],[old]),/manual/);
  assert.throws(()=>planImport([file],[old],[{...next,mode:'120000'}],[old]),/Links/);
  assert.throws(()=>planImport([file],[old],[{...next,size:2e6}],[old]),/Large/);
  assert.throws(()=>planImport(Array(101).fill(file),[],[],[]),/1–100/);
});
test('only latest owner decision on exact SHA permits import',()=>{
  const approved={id:1,user:{login:'caseysm'},state:'APPROVED',commit_id:next.sha};
  assert.equal(ownerApproved([approved],next.sha),true);
  assert.equal(ownerApproved([approved],old.sha),false);
  assert.equal(ownerApproved([{...approved,user:{login:'outsider'}}],next.sha),false);
  assert.equal(ownerApproved([approved,{...approved,id:2,state:'DISMISSED'}],next.sha),false);
});
test('product validation requires completed real CI and every required check',()=>{
  const ci={name:'Scientific CPU tests',status:'completed',conclusion:'success',app:{id:15368}};
  assert.equal(validationPassed([ci],[],['Scientific CPU tests'],['Scientific CPU tests']),true);
  assert.equal(validationPassed([ci],[],[],[]),false);
  assert.equal(validationPassed([{...ci,name:'Validate provenance'}],[],[],['Scientific CPU tests']),false);
  assert.equal(validationPassed([ci],[],[],['Scientific CPU tests','Missing matrix entry']),false);
  assert.equal(validationPassed([ci],[],['Missing check']),false);
  assert.equal(validationPassed([{...ci,status:'in_progress'}],[]),false);
  assert.equal(validationPassed([{...ci,conclusion:'failure'}],[]),false);
  assert.equal(validationPassed([{...ci,name:'CodeRabbit'}],[]),false);
  assert.equal(validationPassed([{...ci,name:'Contribution disclosure'}],[]),false);
  assert.equal(validationPassed([ci],[{context:'Other gate',state:'pending'}]),false);
});
test('source scope rejects guessed pairs and non-main targets',()=>{
  const pr={state:'open',head:{sha:next.sha},base:{sha:old.sha,ref:'main',repo:{full_name:PUBLIC}}};
  assert.doesNotThrow(()=>validateSource(PUBLIC,pr));
  assert.throws(()=>validateSource('example/openfold',pr));
  assert.throws(()=>validateSource(PUBLIC,{...pr,base:{...pr.base,ref:'release'}}));
});
test('unauthorized routing performs no API call or write',async()=>{
  const api={request:()=>{throw Error('API should not be called')}};
  await assert.rejects(route(api,'example/orihime',1,'outsider'),/Only the repository owner/);
});


test('repository pairs are explicit rather than inferred from the environment',()=>{
 assert.deepEqual({...PAIRS},{...EXPECTED_PAIRS});
 assert.equal(PAIRS['unpaired/source'],undefined);
});
test('private intake requires actual agent attribution and quotes untrusted submission text',()=>{
 const pr={number:1,head:{sha:next.sha},base:{sha:old.sha},user:{login:'contributor'},html_url:'https://github.com/example/public/pull/1',body:'## Agent contribution\nInvented facts\n```'};
 const body=developmentBody(PUBLIC,pr,[]);
 assert.match(body,/## Agent contribution\nTODO\n/);assert.match(body,/original submission account: contributor/);
 assert.match(body,/````markdown\n## Agent contribution\nInvented facts/);assert.match(body,/contribution-router-v1/);
});
test('public promotion publishes only source identity, credits and validation',()=>{
 const source={html_url:'https://github.com/example/public/pull/1',user:{login:'contributor'},body:'PRIVATE-HUMAN-AGENT-RECORD'};
 const body=releaseBody(source,next.sha);
 assert.match(body,/## Summary/);assert.match(body,/## Validation/);assert.match(body,/Original contributor: contributor/);
 assert.match(body,new RegExp(next.sha));assert.doesNotMatch(body,/## Human contribution|## Agent contribution|PRIVATE-HUMAN-AGENT-RECORD/);
});

function hostedFixture() {
 const dev=PAIRS[PUBLIC],base='c'.repeat(40),head='d'.repeat(40),merged='e'.repeat(40),imported='f'.repeat(40),writes=[],refs=new Map();
 const source={number:7,title:'Example fix',state:'open',head:{sha:next.sha},base:{sha:old.sha,ref:'main',repo:{full_name:PUBLIC}},user:{login:'external-contributor'},html_url:`https://github.com/${PUBLIC}/pull/7`,body:'PUBLIC-REFERENCE-ONLY'};
 const approval=sha=>[{id:1,user:{login:'caseysm'},state:'APPROVED',commit_id:sha}];
 let privatePR,publicPR,mode='100644',extra=false,commitRecord;
 const api={
  async tree(repo,sha){
   if(repo===PUBLIC && sha===old.sha)return [old];
   if(repo===PUBLIC && sha===next.sha)return [next];
   if(repo===dev && sha===base)return [old];
   if(repo===dev && sha===imported)return [next];
   if(repo===dev && [head,merged].includes(sha))return [{...next,mode},...(extra?[{...old,path:'src/dependency.rs'}]:[])];
   throw Error('Unexpected tree read');
  },
  async pages(path){
   if(path===`repos/${PUBLIC}/pulls/7/reviews`)return approval(next.sha);
   if(path===`repos/${dev}/pulls/9/reviews`)return approval(head);
   if(path===`repos/${PUBLIC}/pulls/7/files`)return [file];
   if(path===`repos/${dev}/commits/${head}/statuses`)return [{id:2,context:'Contribution disclosure',state:'success',creator:{login:'github-actions[bot]'}},{id:1,context:'Contribution disclosure',state:'pending',creator:{login:'github-actions[bot]'}}];
   if(path.startsWith(`repos/${dev}/pulls?`))return privatePR?[privatePR]:[];
   if(path.startsWith(`repos/${PUBLIC}/pulls?`))return publicPR?[publicPR]:[];
   if(path===`repos/${dev}/pulls/9/commits`)return [{sha:imported,parents:[{sha:base}],commit:{message:`Port approved contribution ${PUBLIC}#7\n\nSource-head: ${next.sha}\nSource-base: ${old.sha}\nOriginal-author: external-contributor`}}];
   throw Error('Unexpected paginated read: '+path);
  },
  async request(path,method='GET',body){
   if(method!=='GET'){
    writes.push({path,method,body});
    if(path===`repos/${dev}/git/blobs`)return {sha:next.sha};
    if(path===`repos/${dev}/git/trees`)return {sha:'f'.repeat(40)};
    if(path===`repos/${dev}/git/commits`){commitRecord={message:body.message,tree:{sha:body.tree},parents:body.parents.map(sha=>({sha}))};return {sha:head};}
    if(path===`repos/${dev}/pulls`){privatePR={...body,number:9,user:{login:'router[bot]'},state:'open',merged:false,base:{ref:'main'},head:{sha:head,ref:body.head},merge_commit_sha:merged};return {number:9};}
    if(path===`repos/${PUBLIC}/pulls`){publicPR={...body,number:10,user:{login:'router[bot]'},base:{ref:'main'},head:{ref:body.head,sha:next.sha}};return {number:10};}
    if(path.endsWith('/git/refs')){refs.set(path.replace('/git/refs','/git/ref/heads/'+body.ref.replace('refs/heads/','')),body.sha);return {};}
    if(path.includes('/statuses/'))return {};
    throw Error('Unexpected mutation: '+path);
   }
   if(path===`repos/${PUBLIC}/pulls/7`)return source;
   if(path===`repos/${dev}`)return {private:true,default_branch:'main'};
   if(path===`repos/${dev}/git/ref/heads/main`)return {object:{sha:base}};
   if(path.includes('/git/ref/heads/')){if(refs.has(path))return {object:{sha:refs.get(path)}};throw Object.assign(Error('Not found'),{status:404});}
   if(path===`repos/${PUBLIC}/git/blobs/${next.sha}`)return {size:10,encoding:'base64',content:'ZXhhbXBsZQ=='};
   if(path===`repos/${dev}/git/commits/${base}`)return {tree:{sha:base}};
   if(path===`repos/${dev}/git/commits/${head}`)return commitRecord;
   if(path===`repos/${dev}/pulls/9`)return privatePR;
   if(path===`repos/${dev}/commits/${head}/check-runs?per_page=100`)return {total_count:PRODUCT_CHECKS[dev].length+1,check_runs:[...PRODUCT_CHECKS[dev].map(name=>({name,status:'completed',conclusion:'success',app:{id:15368}})),{name:'Validate contribution disclosure',status:'completed',conclusion:'success',app:{id:15368}}]};
   if(path===`repos/${dev}/rules/branches/main`)return [{type:'required_status_checks',parameters:{required_status_checks:[{context:'Contribution disclosure',integration_id:15368}]}}];
   throw Error('Unexpected read: '+path);
  }
 };
 return {api,dev,source,writes,head,refs,setMode(value){mode=value;},setExtra(){extra=true;},completeDisclosure(){privatePR.body=privatePR.body.replace('## Agent contribution\nTODO','## Agent contribution\nPRIVATE-AGENT-RECORD');privatePR.merged=true;privatePR.state='closed';}};
}
test('approved import and validated promotion keep full records private through actual API orchestration',async()=>{
 const f=hostedFixture();assert.deepEqual(await route(f.api,PUBLIC,7,'caseysm','router'),{state:'routed',number:9});
 const intake=f.writes.find(w=>w.path===`repos/${f.dev}/pulls`);
 assert.match(intake.body.body,/## Agent contribution\nTODO/);assert.match(intake.body.body,new RegExp(f.source.head.sha));
 f.completeDisclosure();assert.deepEqual(await promote(f.api,f.dev,9,'router'),{state:'release-pr-opened',number:10});
 const published=f.writes.find(w=>w.path===`repos/${PUBLIC}/pulls`);
 assert.match(published.body.body,/external-contributor/);assert.doesNotMatch(published.body.body,/PRIVATE-AGENT-RECORD|PUBLIC-REFERENCE-ONLY|## Agent contribution/);
 assert.equal(f.writes.find(w=>w.path===`repos/${PUBLIC}/git/refs`).body.sha,next.sha);
 assert.equal(f.writes.filter(w=>w.path.startsWith(`repos/${PUBLIC}/git/blobs`)).length,0);
});
test('promotion rejects private mode drift before any public write',async()=>{
 const f=hostedFixture();await route(f.api,PUBLIC,7,'caseysm','router');f.completeDisclosure();f.setMode('100755');
 const count=f.writes.length;await assert.rejects(promote(f.api,f.dev,9,'router'),/Development review or merge changed/);
 assert.equal(f.writes.length,count);
});
test('private fixes outside the public patch cannot be omitted by promotion',async()=>{
 const f=hostedFixture();await route(f.api,PUBLIC,7,'caseysm','router');f.completeDisclosure();f.setExtra();
 const count=f.writes.length;await assert.rejects(promote(f.api,f.dev,9,'router'),/Development review or merge changed/);assert.equal(f.writes.length,count);
});
function failOnce(api,path){
 const request=api.request.bind(api);let failed=false;
 api.request=async(p,method='GET',body)=>{if(!failed&&p===path&&method==='POST'){failed=true;throw Object.assign(Error('Transient API failure'),{status:503});}return request(p,method,body);};
}
test('import resumes an existing verified branch after private PR creation fails',async()=>{
 const f=hostedFixture();failOnce(f.api,`repos/${f.dev}/pulls`);
 await assert.rejects(route(f.api,PUBLIC,7,'caseysm','router'),/Transient/);
 assert.deepEqual(await route(f.api,PUBLIC,7,'caseysm','router'),{state:'routed',number:9});
 assert.equal(f.writes.filter(w=>w.path===`repos/${f.dev}/git/refs`).length,1);
});
test('import retries repair a pending status without creating another PR',async()=>{
 const f=hostedFixture();failOnce(f.api,`repos/${PUBLIC}/statuses/${next.sha}`);
 await assert.rejects(route(f.api,PUBLIC,7,'caseysm','router'),/Transient/);
 assert.deepEqual(await route(f.api,PUBLIC,7,'caseysm','router'),{state:'already-routed',number:9});
 assert.equal(f.writes.filter(w=>w.path===`repos/${f.dev}/pulls`).length,1);
 assert.equal(f.writes.at(-1).body.state,'pending');
});
test('promotion resumes its public branch after release PR creation fails',async()=>{
 const f=hostedFixture();await route(f.api,PUBLIC,7,'caseysm','router');f.completeDisclosure();failOnce(f.api,`repos/${PUBLIC}/pulls`);
 await assert.rejects(promote(f.api,f.dev,9,'router'),/Transient/);
 assert.deepEqual(await promote(f.api,f.dev,9,'router'),{state:'release-pr-opened',number:10});
 assert.equal(f.writes.filter(w=>w.path===`repos/${PUBLIC}/git/refs`).length,1);
});
test('promotion retries repair success status without creating another release PR',async()=>{
 const f=hostedFixture();await route(f.api,PUBLIC,7,'caseysm','router');f.completeDisclosure();failOnce(f.api,`repos/${PUBLIC}/statuses/${next.sha}`);
 await assert.rejects(promote(f.api,f.dev,9,'router'),/Transient/);
 assert.deepEqual(await promote(f.api,f.dev,9,'router'),{state:'already-promoted',number:10});
 assert.equal(f.writes.filter(w=>w.path===`repos/${PUBLIC}/pulls`).length,1);assert.equal(f.writes.at(-1).body.state,'success');
});
test('a conflicting public retry branch is never overwritten',async()=>{
 const f=hostedFixture();await route(f.api,PUBLIC,7,'caseysm','router');f.completeDisclosure();
 f.refs.set(`repos/${PUBLIC}/git/ref/heads/validated/7-${next.sha.slice(0,16)}`,old.sha);
 const count=f.writes.length;await assert.rejects(promote(f.api,f.dev,9,'router'),/conflicts/);assert.equal(f.writes.length,count);
});
test('product validation rejects fabricated statuses and check runs from another app',()=>{
 const ci={name:'Product tests',status:'completed',conclusion:'success',app:{id:15368}};
 assert.equal(validationPassed([ci],[],[],[ci.name]),true);
 assert.equal(validationPassed([{...ci,app:{id:999}}],[],[],[ci.name]),false);
 assert.equal(validationPassed([],[{id:1,context:ci.name,state:'success',creator:{login:'github-actions[bot]'}}],[],[ci.name]),false);
});
test('required integration identity and latest status are preserved',()=>{
 const ci={name:'Product tests',status:'completed',conclusion:'success',app:{id:15368}};
 const required=[{context:'Contribution disclosure',integration_id:15368}];
 const pending={id:1,context:'Contribution disclosure',state:'pending',creator:{login:'github-actions[bot]'}};
 const passed={...pending,id:2,state:'success'};
 assert.equal(validationPassed([ci],[pending,passed],required,[ci.name]),true);
 assert.equal(validationPassed([ci],[{...passed,creator:{login:'another-app[bot]'}}],required,[ci.name]),false);
 assert.equal(validationPassed([ci],[passed],required.map(r=>({...r,integration_id:999})),[ci.name]),false);
});
