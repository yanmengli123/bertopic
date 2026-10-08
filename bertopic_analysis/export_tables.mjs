import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { Workbook } from '@oai/artifact-tool';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const out = path.join(root, 'outputs');
const quote = value => {
  const string = value === null || value === undefined ? '' : String(value);
  return /[",\r\n]/.test(string) ? `"${string.replace(/"/g, '""')}"` : string;
};

async function exportTable(input, target, schemaPath) {
  const records = JSON.parse(await fs.readFile(input, 'utf8'));
  const schema = schemaPath ? JSON.parse(await fs.readFile(schemaPath, 'utf8')) : {columns:Object.keys(records[0] ?? {}),rows:records.length};
  const headers = schema.columns;
  assert.equal(records.length, schema.rows);
  if (!headers.length) return;
  const matrix = [headers, ...records.map(record => headers.map(key => record[key] ?? null))];
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add('Results');
  sheet.getRangeByIndexes(0,0,matrix.length,headers.length).values = matrix;
  workbook.recalculate();
  const restored = sheet.getRangeByIndexes(0,0,matrix.length,headers.length).values;
  assert.deepEqual(restored, matrix, `Changed values in ${target}`);
  const csv = '\uFEFF' + restored.map(row => row.map(quote).join(',')).join('\r\n') + '\r\n';
  const temporary = target + '.tmp';
  await fs.writeFile(temporary,csv,'utf8');
  await fs.rename(temporary,target);
  console.log(JSON.stringify({file:target,records:records.length,columns:headers.length}));
}

await exportTable(path.join(out,'corpus.json'),path.join(out,'merged_four_databases.csv'));
async function visit(directory) {
  for (const item of await fs.readdir(directory,{withFileTypes:true})) {
    if (item.isDirectory() && !['portable_model','figures','visualizations'].includes(item.name)) {
      await visit(path.join(directory,item.name));
    } else if (item.isFile() && item.name.endsWith('.schema.json')) {
      const stem = item.name.slice(0,-'.schema.json'.length);
      await exportTable(path.join(directory,stem+'.json'),path.join(directory,stem+'.csv'),path.join(directory,item.name));
    }
  }
}
await visit(out);
