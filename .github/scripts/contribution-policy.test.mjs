import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {missingSections,incompletePulls,openPulls,checkDisclosure,workflow} from './contribution-policy.mjs';
const names=['Summary','Human contribution','Agent contribution','External sources','Validation'];
const valid=names.map(n=>'<h2>'+n+'</h2><p>A factual answer.</p>').join('\n');
const pull=(number,body_html=valid,sha='a'.repeat(40),state='open')=>({number,body_html,head:{sha},state});
const fixtures=JSON.parse(readFileSync(new URL('./test-fixtures/contribution-rendering.json',import.meta.url),'utf8'));
for(const row of fixtures.cases)test('GitHub rendering: '+row.name,()=>assert.deepEqual(missingSections(row.html),row.missing));
test('checked-in privileged metadata workflow exactly matches its tested generator',()=>{
 assert.equal(readFileSync(new URL('../workflows/contribution-policy.yml',import.meta.url),'utf8'),workflow());
 assert.match(workflow(),/queue: max/);assert.match(workflow(),/cancel-in-progress: false/);
 assert.doesNotMatch(workflow(),/actions\/checkout|execSync\(|shell: true/);
});
test('rejects missing or oversized rendered bodies without a Markdown fallback',()=>{
 for(const value of [undefined,null,{},'x'.repeat(1024*1024+1)])assert.throws(()=>missingSections(value),/rendered/);
 assert.deepEqual(missingSections(''),names);
});
test('non-rendering HTML and empty sections cannot impersonate an answer',()=>{
 for(const answer of ['<script>hidden</script>','<style>hidden</style>','<template>hidden</template>','<p hidden>hidden</p>'])assert.deepEqual(missingSections(valid.replace('<p>A factual answer.</p>',answer)),['Summary']);
});
test('a complete PR cannot mask an incomplete open companion at the same SHA',()=>{
 for(const rows of [[pull(1),pull(2,'')],[pull(2,''),pull(1)]])assert.deepEqual(incompletePulls(rows,'a'.repeat(40)),[{number:2,missing:names}]);
 assert.deepEqual(incompletePulls([pull(1),pull(2,'','a'.repeat(40),'closed'),pull(3,'','b'.repeat(40))],'a'.repeat(40)),[]);
 assert.deepEqual(incompletePulls([pull(1,null)],'a'.repeat(40)),[{number:1,missing:names}]);
});
test('open PR metadata requests rendered HTML and paginates completely within its bound',async()=>{
 let calls=0;const github={rest:{pulls:{list:async({page,state,headers})=>{
  assert.equal(state,'open');assert.equal(headers.accept,'application/vnd.github.full+json');calls++;
  return {data:[pull(page)],headers:page===1?{link:'<next>; rel="next"'}:{}};
 }}}};
 assert.equal((await openPulls(github,'company','repo')).length,2);assert.equal(calls,2);
 github.rest.pulls.list=async()=>({data:[],headers:{link:'<next>; rel="next"'}});
 await assert.rejects(()=>openPulls(github,'company','repo'),/Too many/);
});
function fixture(current,related,payload={pull_request:current}) {
 const statuses=[],errors=[];
 const context={repo:{owner:'company',repo:'repo'},issue:{number:current.number},serverUrl:'https://github.com',runId:1,payload};
 const github={rest:{pulls:{get:async({headers})=>{assert.equal(headers.accept,'application/vnd.github.full+json');return {data:current};},list:async()=>({data:related,headers:{}})},repos:{createCommitStatus:async value=>{statuses.push(value);}}}};
 return {github,context,core:{setFailed:value=>errors.push(value)},statuses,errors};
}
test('workflow orchestration checks fresh current data and incomplete same-head companions',async()=>{
 const f=fixture(pull(1),[pull(2,'')]);await checkDisclosure(f.github,f.context,f.core);
 assert.equal(f.statuses.at(-1).state,'failure');assert.match(f.errors[0],/#2/);
 const g=fixture(pull(1),[]);await checkDisclosure(g.github,g.context,g.core);
 assert.equal(g.statuses.at(-1).state,'success');assert.deepEqual(g.errors,[]);
});
test('moving an incomplete companion clears the previous head and blocks its new head',async()=>{
 const a='a'.repeat(40),b='b'.repeat(40),f=fixture(pull(2,'',b),[pull(1),pull(2,'',b)],{before:a,pull_request:pull(2,'',b)});
 await checkDisclosure(f.github,f.context,f.core);
 const latest=new Map(f.statuses.map(s=>[s.sha,s.state]));assert.equal(latest.get(a),'success');assert.equal(latest.get(b),'failure');
});
test('old events check the fresh head rather than granting it the old body result',async()=>{
 const b='b'.repeat(40),f=fixture(pull(1,'',b),[pull(1,'',b)],{pull_request:pull(1)});
 await checkDisclosure(f.github,f.context,f.core);assert.equal(f.statuses.filter(s=>s.sha===b).at(-1).state,'failure');
});
test('closing an incomplete PR revalidates the remaining open companion',async()=>{
 const f=fixture(pull(2,'','a'.repeat(40),'closed'),[pull(1)]);await checkDisclosure(f.github,f.context,f.core);
 assert.equal(f.statuses.at(-1).state,'success');
});
test('lookup and pagination failures leave all known heads pending',async()=>{
 for(const fail of ['list','get']) {
  const f=fixture(pull(1),[],{pull_request:pull(1),before:'b'.repeat(40)});
  f.github.rest.pulls[fail]=async()=>{throw Error('API unavailable');};
  await assert.rejects(()=>checkDisclosure(f.github,f.context,f.core),/unavailable/);
  assert.equal(new Set(f.statuses.map(s=>s.sha)).size,2);assert.ok(f.statuses.every(s=>s.state==='pending'));
 }
 const f=fixture(pull(1),[]);f.github.rest.pulls.list=async()=>({data:[],headers:{link:'<next>; rel="next"'}});
 await assert.rejects(()=>checkDisclosure(f.github,f.context,f.core),/Too many/);assert.equal(f.statuses.at(-1).state,'pending');
});
test('a missing rendered body fails closed even when raw Markdown looks complete',async()=>{
 const current=pull(1);delete current.body_html;current.body='This must never be parsed';
 const f=fixture(current,[]);await assert.rejects(()=>checkDisclosure(f.github,f.context,f.core),/rendered/);
 assert.equal(f.statuses.at(-1).state,'pending');
});
test('unrelated heads consume neither parser work nor commit-status quota',async()=>{
 const other=pull(2,undefined,'b'.repeat(40));delete other.body_html;
 const f=fixture(pull(1),[pull(1),other]);await checkDisclosure(f.github,f.context,f.core);
 assert.ok(f.statuses.every(s=>s.sha==='a'.repeat(40)));assert.equal(f.statuses.at(-1).state,'success');
});
test('one failed final status write cannot prevent checking another affected head',async()=>{
 const a='a'.repeat(40),b='b'.repeat(40),f=fixture(pull(2,'',b),[pull(1),pull(2,'',b)],{before:a,pull_request:pull(2,'',b)});
 const write=f.github.rest.repos.createCommitStatus;
 f.github.rest.repos.createCommitStatus=async value=>{if(value.sha===b&&value.state==='failure')throw Error('Status unavailable');return write(value);};
 await assert.rejects(()=>checkDisclosure(f.github,f.context,f.core),/Status unavailable/);
 assert.equal(f.statuses.filter(s=>s.sha===a).at(-1).state,'success');
});
test('a failed invalidation still attempts other event heads before any lookup',async()=>{
 const a='a'.repeat(40),b='b'.repeat(40),f=fixture(pull(1),[],{before:b,pull_request:pull(1)});
 const write=f.github.rest.repos.createCommitStatus;
 f.github.rest.repos.createCommitStatus=async value=>{if(value.sha===a)throw Error('Status unavailable');return write(value);};
 f.github.rest.pulls.get=async()=>{throw Error('Should not be queried');};
 await assert.rejects(()=>checkDisclosure(f.github,f.context,f.core),/Status unavailable/);
 assert.equal(f.statuses[0].sha,b);assert.equal(f.statuses[0].state,'pending');
});
test('the exact generated github-script executes with stdin-only parsing',async()=>{
 const script=workflow().split('          script: |\n')[1].split('\n').map(line=>line.slice(12)).join('\n');
 const execute=new Function('require','github','context','core','return (async()=>{'+script+'})();');
 const f=fixture(pull(1),[]);await execute(createRequire(import.meta.url),f.github,f.context,f.core);
 assert.equal(f.statuses.at(-1).state,'success');
});
