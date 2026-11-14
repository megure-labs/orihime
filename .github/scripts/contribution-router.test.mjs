import test from 'node:test';
import assert from 'node:assert/strict';
import {PAIRS,planImport,ownerApproved,validationPassed,validateSource,route,promote,developmentBody,releaseBody} from './contribution-router.mjs';
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
for(const filename of ['../secret','a/../secret','a/.git/config','/absolute','.github/workflows/run.yml','.claude/hooks.json','AGENTS.md','a\u0000b'])test('rejects unsafe path '+JSON.stringify(filename),()=>assert.throws(()=>planImport([{...file,filename}],[],[],[]),/unsafe/));
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
  const ci={name:'Scientific CPU tests',status:'completed',conclusion:'success'};
  assert.equal(validationPassed([ci],[],['Scientific CPU tests']),true);
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
 assert.deepEqual({...PAIRS},{'megure-labs/orihime':'megure-labs/orihime-dev','orikata-bio/hikoboshi':'orikata-bio/hikoboshi-dev'});
 assert.equal(PAIRS['megure-labs/hikoboshi'],undefined);assert.equal(PAIRS['orikata-bio/orihime'],undefined);
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
 const dev=PAIRS[PUBLIC],base='c'.repeat(40),head='d'.repeat(40),merged='e'.repeat(40),writes=[];
 const source={number:7,title:'Example fix',state:'open',head:{sha:next.sha},base:{sha:old.sha,ref:'main',repo:{full_name:PUBLIC}},user:{login:'external-contributor'},html_url:`https://github.com/${PUBLIC}/pull/7`,body:'PUBLIC-REFERENCE-ONLY'};
 const approval=sha=>[{id:1,user:{login:'caseysm'},state:'APPROVED',commit_id:sha}];
 let privatePR,mode='100644';
 const api={
  async tree(repo,sha){
   if(repo===PUBLIC && sha===old.sha)return [old];
   if(repo===PUBLIC && sha===next.sha)return [next];
   if(repo===dev && sha===base)return [old];
   if(repo===dev && [head,merged].includes(sha))return [{...next,mode}];
   throw Error('Unexpected tree read');
  },
  async pages(path){
   if(path===`repos/${PUBLIC}/pulls/7/reviews`)return approval(next.sha);
   if(path===`repos/${dev}/pulls/9/reviews`)return approval(head);
   if(path===`repos/${PUBLIC}/pulls/7/files`)return [file];
   if(path.startsWith(`repos/${dev}/pulls?`) || path.startsWith(`repos/${PUBLIC}/pulls?`))return [];
   if(path===`repos/${dev}/pulls/9/commits`)return [{parents:[{sha:base}],commit:{message:`Port approved contribution ${PUBLIC}#7\n\nSource-head: ${next.sha}\nSource-base: ${old.sha}\nOriginal-author: external-contributor`}}];
   throw Error('Unexpected paginated read: '+path);
  },
  async request(path,method='GET',body){
   if(method!=='GET'){
    writes.push({path,method,body});
    if(path===`repos/${dev}/git/blobs`)return {sha:next.sha};
    if(path===`repos/${dev}/git/trees`)return {sha:'f'.repeat(40)};
    if(path===`repos/${dev}/git/commits`)return {sha:head};
    if(path===`repos/${dev}/pulls`){privatePR={...body,number:9,user:{login:'router[bot]'},merged:true,base:{ref:'main'},head:{sha:head},merge_commit_sha:merged};return {number:9};}
    if(path===`repos/${PUBLIC}/pulls`)return {number:10};
    if(path.endsWith('/git/refs') || path.includes('/statuses/'))return {};
    throw Error('Unexpected mutation: '+path);
   }
   if(path===`repos/${PUBLIC}/pulls/7`)return source;
   if(path===`repos/${dev}`)return {private:true,default_branch:'main'};
   if(path===`repos/${dev}/git/ref/heads/main`)return {object:{sha:base}};
   if(path===`repos/${PUBLIC}/git/blobs/${next.sha}`)return {size:10,encoding:'base64',content:'ZXhhbXBsZQ=='};
   if(path===`repos/${dev}/git/commits/${base}`)return {tree:{sha:base}};
   if(path===`repos/${dev}/pulls/9`)return privatePR;
   if(path===`repos/${dev}/commits/${head}/check-runs?per_page=100`)return {total_count:2,check_runs:[{name:'Product tests',status:'completed',conclusion:'success'},{name:'Validate contribution disclosure',status:'completed',conclusion:'success'}]};
   if(path===`repos/${dev}/commits/${head}/status?per_page=100`)return {statuses:[]};
   if(path===`repos/${dev}/rules/branches/main`)return [{type:'required_status_checks',parameters:{required_status_checks:[{context:'Product tests'},{context:'Validate contribution disclosure'}]}}];
   throw Error('Unexpected read: '+path);
  }
 };
 return {api,dev,source,writes,head,setMode(value){mode=value;},completeDisclosure(){privatePR.body=privatePR.body.replace('## Agent contribution\nTODO','## Agent contribution\nPRIVATE-AGENT-RECORD');}};
}
test('approved import and validated promotion keep full records private through actual API orchestration',async()=>{
 const f=hostedFixture();assert.deepEqual(await route(f.api,PUBLIC,7,'caseysm'),{state:'routed',number:9});
 const intake=f.writes.find(w=>w.path===`repos/${f.dev}/pulls`);
 assert.match(intake.body.body,/## Agent contribution\nTODO/);assert.match(intake.body.body,new RegExp(f.source.head.sha));
 f.completeDisclosure();assert.deepEqual(await promote(f.api,f.dev,9,'router'),{state:'release-pr-opened',number:10});
 const published=f.writes.find(w=>w.path===`repos/${PUBLIC}/pulls`);
 assert.match(published.body.body,/external-contributor/);assert.doesNotMatch(published.body.body,/PRIVATE-AGENT-RECORD|PUBLIC-REFERENCE-ONLY|## Agent contribution/);
 assert.equal(f.writes.find(w=>w.path===`repos/${PUBLIC}/git/refs`).body.sha,next.sha);
 assert.equal(f.writes.filter(w=>w.path.startsWith(`repos/${PUBLIC}/git/blobs`)).length,0);
});
test('promotion rejects private mode drift before any public write',async()=>{
 const f=hostedFixture();await route(f.api,PUBLIC,7,'caseysm');f.completeDisclosure();f.setMode('100755');
 const count=f.writes.length;await assert.rejects(promote(f.api,f.dev,9,'router'),/Development fixes differ/);
 assert.equal(f.writes.length,count);
});
