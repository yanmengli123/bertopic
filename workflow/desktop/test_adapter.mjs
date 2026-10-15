import assert from 'node:assert/strict';
import { invoke, workerEnvironment } from './dsh_adapter.mjs';

const python = process.argv[2];
process.env.DSH_PRIVATE_TOKEN = 'test-not-inherited';
assert.equal(workerEnvironment().DSH_PRIVATE_TOKEN, undefined);
assert.equal(workerEnvironment({ SILICONFLOW_API_KEY: 'test-only' }).SILICONFLOW_API_KEY, 'test-only');
assert.throws(() => workerEnvironment({ PYTHONPATH: 'untrusted-path' }));
assert.ok(python, 'Pass the absolute Python interpreter path.');
const events = [];
const result = await invoke({ python, args: ['capabilities'], onEvent: event => events.push(event) });
assert.equal(result.event.event, 'capabilities');
assert.equal(events[0].data.protocol_version, 1);
assert.ok(events[0].data.sources.includes('VIP'));
await assert.rejects(() => invoke({ python, args: ['inspect', '--input', 'missing.csv'] }),
  error => error.event.data.code === 'INPUT_INVALID');
const controller = new AbortController();
controller.abort();
await assert.rejects(() => invoke({ python, args: ['capabilities'], signal: controller.signal }), /cancelled/);
console.log('Desktop adapter: capabilities, error propagation and pre-launch cancellation passed.');
