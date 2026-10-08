import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { Workbook } from '@oai/artifact-tool';

// Run with the bundled Node runtime. Input exports are never modified.
const inputDir = path.resolve(process.argv[2] ?? 'D:/bertopic/PubMed');
const outputPath = path.join(inputDir, 'pubmed_merged_bertopic.csv');
const reportPath = path.join(path.dirname(fileURLToPath(import.meta.url)), 'merge_report.json');
const names = {
  medline: 'pubmed-Oryzasativ-set.txt',
  csv: 'csv-Oryzasativ-set.csv',
  pmid: 'pmid-Oryzasativ-set.txt',
  abstract: 'abstract-Oryzasativ-set.txt',
  summary: 'summary-Oryzasativ-set.txt',
  timeline: 'PubMed_Timeline_Results_by_Year.csv',
};
const files = Object.fromEntries(await Promise.all(Object.entries(names).map(async ([key, name]) => {
  const bytes = await fs.readFile(path.join(inputDir, name));
  return [key, { text: bytes.toString('utf8').replace(/^\uFEFF/, ''), sha256: createHash('sha256').update(bytes).digest('hex') }];
})));
const clean = value => String(value ?? '').replace(/\s+/gu, ' ').trim();
const unique = values => [...new Set(values.filter(Boolean))];

function parseMedline(text) {
  const result = [];
  let record = null;
  let tag = null;
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (!line.trim()) { tag = null; continue; }
    const match = line.match(/^([A-Z0-9]{2,4})\s*-\s(.*)$/);
    if (match) {
      if (match[1] === 'PMID') {
        if (record) result.push(record);
        record = {};
      }
      assert(record, `Field before first PMID at line ${index + 1}`);
      tag = match[1];
      (record[tag] ??= []).push(match[2]);
    } else if (line.startsWith('      ') && record && tag) {
      record[tag][record[tag].length - 1] += ` ${line.trim()}`;
    } else {
      throw new Error(`Unrecognized MEDLINE line ${index + 1}`);
    }
  }
  if (record) result.push(record);
  for (const rec of result) {
    for (const key of Object.keys(rec)) rec[key] = rec[key].map(clean);
    assert.equal(rec.PMID?.length, 1, 'Each MEDLINE record must have exactly one PMID');
    assert.match(rec.PMID[0], /^\d+$/);
  }
  return result;
}

function recordMap(records, label, getId) {
  const result = new Map();
  for (const record of records) {
    const id = getId(record);
    assert.match(id, /^\d+$/, `Invalid ${label} PMID`);
    assert(!result.has(id), `Duplicate PMID ${id} in ${label}`);
    result.set(id, record);
  }
  return result;
}

function displayBlocks(text, startPattern, pmidPattern, label) {
  text = text.replace(/\r\n/g, '\n');
  const starts = [...text.matchAll(startPattern)];
  return recordMap(starts.map((match, i) => {
    const block = text.slice(match.index, starts[i + 1]?.index ?? text.length);
    const pmids = [...block.matchAll(pmidPattern)];
    assert.equal(pmids.length, 1, `Expected one article PMID in ${label} record ${i + 1}`);
    return { pmid: pmids[0][1], block: clean(block) };
  }), label, rec => rec.pmid);
}

const medline = parseMedline(files.medline.text);
const medlineById = recordMap(medline, 'MEDLINE', rec => rec.PMID[0]);
const csvWorkbook = await Workbook.fromCSV(files.csv.text, { sheetName: 'PubMed CSV' });
const csvMatrix = csvWorkbook.worksheets.getItem('PubMed CSV').getUsedRange().values;
const csvHeaders = csvMatrix[0].map(String);
const csvRows = csvMatrix.slice(1).filter(row => row.some(value => value !== null && value !== '')).map(row =>
  Object.fromEntries(csvHeaders.map((header, i) => [header, clean(row[i])])));
const csvById = recordMap(csvRows, 'CSV', rec => rec.PMID);
const pmidRows = files.pmid.text.trim().split(/\s+/);
const pmidById = recordMap(pmidRows, 'PMID list', id => id);
const abstractById = displayBlocks(files.abstract.text, /(?:^|\n\n\n)\d+\.[ \t]+/g, /^PMID:\s*(\d+)/gm, 'abstract export');
const summaryById = displayBlocks(files.summary.text, /^\d+:\s+/gm, /PMID:\s*(\d+)/g, 'summary export');
const sources = { medline: medlineById, csv: csvById, pmid: pmidById, abstract: abstractById, summary: summaryById };
const expectedPmids = [...medlineById.keys()].sort();
for (const [label, source] of Object.entries(sources)) {
  assert.deepEqual([...source.keys()].sort(), expectedPmids, `${label} export has different PMID coverage`);
}

const first = (rec, tag) => rec[tag]?.[0] ?? '';
const list = (rec, tag) => unique(rec[tag] ?? []).join('; ');
const taggedId = (rec, suffix) => {
  const values = [...(rec.AID ?? []), ...(rec.LID ?? [])];
  const expression = new RegExp(`^(.*?)\\s*\\[${suffix}\\]$`);
  return values.map(value => value.match(expression)?.[1]).find(Boolean) ?? '';
};
const problems = { title_conflicts: [], year_conflicts: [], doi_conflicts: [], abstract_export_mismatches: [] };
const rows = pmidRows.map(pmid => {
  const rec = medlineById.get(pmid);
  const csv = csvById.get(pmid);
  const title = first(rec, 'TI') || csv.Title;
  const abstract = clean((rec.AB ?? []).join(' '));
  const pubDate = first(rec, 'DP');
  const medlineYear = pubDate.match(/\b\d{4}\b/)?.[0] ?? '';
  const doi = taggedId(rec, 'doi') || csv.DOI;
  const equivalentTitle = value => clean(value).replace(/\.$/, '');
  if (equivalentTitle(title) !== equivalentTitle(csv.Title)) problems.title_conflicts.push(pmid);
  if (medlineYear && csv['Publication Year'] && medlineYear !== csv['Publication Year']) problems.year_conflicts.push(pmid);
  if (doi && csv.DOI && doi.toLowerCase() !== csv.DOI.toLowerCase()) problems.doi_conflicts.push(pmid);
  if (abstract && !abstractById.get(pmid).block.includes(abstract)) problems.abstract_export_mismatches.push(pmid);
  assert(title, `Missing title for ${pmid}`);
  const text = abstract ? `${title}${/[.!?]$/.test(title) ? '' : '.'} ${abstract}` : title;
  return {
    pmid, title, abstract, text,
    has_abstract: abstract ? 1 : 0,
    year: csv['Publication Year'] || medlineYear,
    publication_date: pubDate,
    journal: first(rec, 'TA') || csv['Journal/Book'],
    journal_full: first(rec, 'JT'),
    authors: csv.Authors || list(rec, 'AU'),
    first_author: csv['First Author'] || first(rec, 'AU'),
    authors_full: list(rec, 'FAU'),
    doi,
    pmcid: first(rec, 'PMC') || csv.PMCID,
    nihms_id: csv['NIHMS ID'] || first(rec, 'MID'),
    language: list(rec, 'LA'),
    keywords: list(rec, 'OT'),
    mesh_terms: list(rec, 'MH'),
    publication_types: list(rec, 'PT'),
    volume: first(rec, 'VI'),
    issue: first(rec, 'IP'),
    pages: first(rec, 'PG'),
    citation: first(rec, 'SO') || csv.Citation,
    create_date: csv['Create Date'] || first(rec, 'CRDT').split(' ')[0],
    other_abstract: clean((rec.OAB ?? []).join(' ')),
    other_abstract_language: list(rec, 'OABL'),
    pubmed_url: `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`,
    source: 'PubMed',
    source_files: Object.keys(sources).map(key => names[key]).join('; '),
  };
});

const headers = Object.keys(rows[0]);
const matrix = [headers, ...rows.map(row => headers.map(header => row[header]))];
const outputWorkbook = Workbook.create();
const sheet = outputWorkbook.worksheets.add('PubMed');
sheet.getRangeByIndexes(0, 0, matrix.length, headers.length).values = matrix;
outputWorkbook.recalculate();
const savedMatrix = sheet.getRangeByIndexes(0, 0, matrix.length, headers.length).values;
assert.deepEqual(savedMatrix, matrix, 'Workbook changed source text or values');
const inspection = await outputWorkbook.inspect({ kind: 'table', range: 'PubMed!A1:F4', include: 'values', maxChars: 1800, tableMaxRows: 4, tableMaxCols: 6, tableMaxCellChars: 90 });
console.log(inspection.ndjson);

// CSV has no formatting or formulas. Export exact cell values with RFC-compatible quoting.
const quote = value => {
  const string = String(value ?? '');
  return /[",\r\n]/.test(string) ? `"${string.replace(/"/g, '""')}"` : string;
};
const outputCsv = '\uFEFF' + savedMatrix.map(row => row.map(quote).join(',')).join('\r\n') + '\r\n';
try { await fs.access(outputPath); throw new Error(`Output already exists: ${outputPath}`); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
await fs.writeFile(outputPath, outputCsv, { encoding: 'utf8', flag: 'wx' });

const timelineLines = files.timeline.text.trim().split(/\r?\n/).slice(2);
const timelineCounts = Object.fromEntries(timelineLines.map(line => {
  const [year, count] = line.split(',');
  return [year, Number(count)];
}));
const yearCounts = {};
for (const row of rows) yearCounts[row.year] = (yearCounts[row.year] ?? 0) + 1;
const timelineDifferences = unique([...Object.keys(timelineCounts), ...Object.keys(yearCounts)]).filter(year =>
  (timelineCounts[year] ?? 0) !== (yearCounts[year] ?? 0)).map(year => ({ year, timeline: timelineCounts[year] ?? 0, articles: yearCounts[year] ?? 0 }));
const report = {
  output: outputPath, rows: rows.length, columns: headers,
  unique_pmids: new Set(rows.map(row => row.pmid)).size,
  source_record_counts: Object.fromEntries(Object.entries(sources).map(([key, source]) => [names[key], source.size])),
  source_sha256: Object.fromEntries(Object.entries(files).map(([key, value]) => [names[key], value.sha256])),
  with_abstract: rows.filter(row => row.has_abstract).length,
  missing_abstract_pmids: rows.filter(row => !row.has_abstract).map(row => row.pmid),
  missing_text: rows.filter(row => !row.text).length,
  language_counts: rows.reduce((counts, row) => { counts[row.language] = (counts[row.language] ?? 0) + 1; return counts; }, {}),
  year_range: [Math.min(...rows.map(row => Number(row.year))), Math.max(...rows.map(row => Number(row.year)))],
  year_counts: yearCounts,
  timeline_total: Object.values(timelineCounts).reduce((a, b) => a + b, 0),
  timeline_year_differences: timelineDifferences,
  ...problems,
  encoding: 'UTF-8 with BOM',
  text_rule: 'MEDLINE title + abstract; title only when the abstract is absent',
  cleaning: 'Join wrapped lines and normalize whitespace. Keep punctuation, case, numbers, gene names and all articles.',
  output_sha256: createHash('sha256').update(Buffer.from(outputCsv, 'utf8')).digest('hex'),
};
await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n', 'utf8');
console.log(JSON.stringify({ output: report.output, rows: report.rows, with_abstract: report.with_abstract, missing_abstract: report.missing_abstract_pmids.length, missing_text: report.missing_text, year_range: report.year_range, language_counts: report.language_counts, timeline_total: report.timeline_total, timeline_year_differences: report.timeline_year_differences, title_conflicts: report.title_conflicts, year_conflicts: report.year_conflicts, doi_conflicts: report.doi_conflicts, abstract_export_mismatches: report.abstract_export_mismatches }, null, 2));
