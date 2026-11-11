import test from 'node:test';
import assert from 'node:assert/strict';
import {PAIRS,planImport,ownerApproved,validationPassed,validateSource,route} from './contribution-router.mjs';
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
