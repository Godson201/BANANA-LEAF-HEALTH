/* DOM integration checks against the real local Flask server. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM} = require('jsdom');
const base = 'http://127.0.0.1:5050';
const root = path.resolve(__dirname, '../..');
async function waitFor(test, label) {
  const deadline = Date.now() + 60000;
  while (!test()) {
    if (Date.now() > deadline) throw new Error('Timed out: ' + label);
    await new Promise(resolve => setTimeout(resolve, 25));
  }
}
(async () => {
  const dom = new JSDOM(await (await fetch(base)).text(), {url: base, runScripts: 'outside-only'});
  const {window} = dom;
  const document = window.document;
  const id = name => document.getElementById(name);
  window.URL.createObjectURL = () => 'blob:preview';
  window.URL.revokeObjectURL = () => {};
  let printed = false;
  window.print = () => {printed = true;};
  window.fetch = async (url, options = {}) => {
    if (options.body instanceof window.FormData) {
      const body = new FormData();
      for (const [key, value] of options.body.entries()) {
        if (typeof value === 'string') {body.append(key, value); continue;}
        const bytes = await new Promise((resolve, reject) => {
          const reader = new window.FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject; reader.readAsArrayBuffer(value);
        });
        body.append(key, new Blob([bytes], {type: value.type}), value.name);
      }
      options = {...options, body};
    }
    return fetch(new URL(url, base), options);
  };
  window.eval(fs.readFileSync(path.join(root, 'dashboard/static/app.js'), 'utf8'));
  await waitFor(() => id('model-status').textContent.includes('Model ready') && id('disease-cards').children.length === 8, 'metadata and guide');
  id('disease-search').value = 'bunchy'; id('disease-search').dispatchEvent(new window.Event('input'));
  assert.equal(id('disease-cards').children.length, 1);
  assert.ok(id('disease-cards').textContent.includes('Not predicted'));
  id('disease-search').value = ''; id('disease-search').dispatchEvent(new window.Event('input'));
  document.querySelector('input[value="whole_plant_wilt"]').checked = true;
  id('field-form').dispatchEvent(new window.Event('submit', {bubbles: true, cancelable: true}));
  await waitFor(() => id('field-result').textContent.includes('Arrange an extension review'), 'field triage');
  assert.ok(id('field-result').textContent.includes('Rule-based'));
  console.log('PASS: searchable disease guide, guidance-only labels and whole-plant triage');

  for (const model of ['baseline', 'expanded']) {
    id('model-choice').value = model; id('model-choice').dispatchEvent(new window.Event('change'));
    assert.equal(id('metric-classes').textContent, model === 'baseline' ? '4' : '6');
    const buttons = [...document.querySelectorAll('.sample-button')].filter(button => !button.hidden);
    assert.equal(buttons.length, model === 'baseline' ? 4 : 6);
    for (const button of buttons) {
      button.click();
      await waitFor(() => !id('prediction-result').hidden, 'sample ' + button.dataset.sample);
      assert.equal(id('probability-chart').children.length, model === 'baseline' ? 4 : 6);
      assert.equal(id('care-panel').hidden, false);
      assert.equal(id('care-actions').children.length, 3);
      assert.ok(id('care-source').href.startsWith('https://'));
      if (model === 'expanded') assert.ok(id('result-description').textContent.includes('Research preview'));
      id('print-result').click(); assert.ok(printed);
      id('reset-button').click();
      assert.equal(id('care-panel').hidden, true);
    }
  }
  console.log('PASS: both real models, ten sample interactions, separate metrics and farmer action plans');
  id('model-choice').value = 'baseline'; id('model-choice').dispatchEvent(new window.Event('change'));
  const photo = new window.File([fs.readFileSync(path.join(root, 'data/OriginalSet/cordana/87.jpeg'))], 'known-banana.jpg', {type: 'image/jpeg'});
  Object.defineProperty(id('image-input'), 'files', {value: [photo], configurable: true});
  id('image-input').dispatchEvent(new window.Event('change', {bubbles: true}));
  id('upload-form').dispatchEvent(new window.Event('submit', {bubbles: true, cancelable: true}));
  await waitFor(() => !id('unassessed-result').hidden, 'uncertain photo');
  assert.equal(id('probability-chart').children.length, 0);
  id('confirm-banana').click();
  await waitFor(() => !id('prediction-result').hidden, 'confirmed photo');
  assert.ok(id('result-description').textContent.includes('confirmed by you'));
  assert.equal(id('care-panel').hidden, false);
  console.log('PASS: live upload, no stale scores, confirmation and sourced care guidance');
  dom.window.close();
})().catch(error => {console.error(error); process.exitCode = 1;});
