'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

test('counts Fluid highlighted code lines from the gutter', () => {
  const script = fs.readFileSync(
    path.join(__dirname, '..', 'scripts', 'code-collapse.js'),
    'utf8'
  );
  const article = fs.readFileSync(
    path.join(
      __dirname,
      '..',
      'public',
      '2026',
      '09',
      '19',
      '论文理解LLM_OR_AlphaOPTFormulatingOptimizationProgramswithSelf_ImprovingLLMExperien',
      'index.html'
    ),
    'utf8'
  );

  const longGutter = article.match(
    /<td class="gutter"><pre>((?:<span class="line">.*?<\/span><br>){21,})<\/pre><\/td>/s
  );

  assert.ok(longGutter, 'fixture should contain a Fluid code block longer than 20 lines');
  assert.match(script, /querySelectorAll\('td\.gutter span\.line'\)/);
});
