/** Generic subprocess bridge. DSH's official registration awaits its SDK. */
import { spawn } from 'node:child_process';
import { StringDecoder } from 'node:string_decoder';
import path from 'node:path';

export function workerEnvironment(credentials = {}) {
  const names = ['PATH', 'Path', 'PATHEXT', 'SystemRoot', 'SYSTEMROOT', 'WINDIR', 'COMSPEC',
    'TEMP', 'TMP', 'HOME', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'PROGRAMDATA',
    'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'LANG', 'LC_ALL', 'SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE'];
  const environment = {};
  for (const name of names) if (process.env[name] !== undefined) environment[name] = process.env[name];
  for (const [name, value] of Object.entries(credentials)) {
    if (!/^[A-Z][A-Z0-9_]*(?:_API_KEY|_TOKEN)$/.test(name) || typeof value !== 'string') {
      throw new TypeError('Only explicitly named API credential variables are accepted.');
    }
    environment[name] = value;
  }
  return { ...environment, PYTHONUTF8: '1', PYTHONUNBUFFERED: '1' };
}

export function invoke({ python, args, cwd, credentialEnv = {}, onEvent = () => {}, signal,
                         maxStderrBytes = 131072, cancellationGraceMs = 15000 }) {
  if (!path.isAbsolute(python) || !Array.isArray(args) || args.some(a => typeof a !== 'string')) {
    throw new TypeError('Use an absolute trusted Python executable and an array of arguments.');
  }
  if (signal?.aborted) return Promise.reject(new Error('Workflow cancelled before launch.'));
  return new Promise((resolve, reject) => {
    const child = spawn(python, ['-u', '-B', '-m', 'bertopic_workflow', ...args], {
      cwd, env: workerEnvironment(credentialEnv),
      shell: false, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
    });
    const decoder = new StringDecoder('utf8');
    let buffer = '', stderr = '', stderrBytes = 0, lastEvent, runDirectory, failure, timer, cancelled = false;
    function terminate() {
      if (child.exitCode !== null) return;
      if (process.platform === 'win32') {
        // The venv launcher can have a real Python child; terminate its process tree.
        const killer = spawn(path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'taskkill.exe'),
          ['/PID', String(child.pid), '/T', '/F'], { shell: false, windowsHide: true, stdio: 'ignore' });
        killer.on('error', () => child.kill());
      } else child.kill();
    }
    function receive(line) {
      if (!line.trim()) return;
      const item = JSON.parse(line);
      if (item.schema_version !== 1 || typeof item.event !== 'string' || !item.data) {
        throw new Error('Invalid workflow protocol event.');
      }
      lastEvent = item;
      if (item.event === 'run_started') runDirectory = item.data.run_directory;
      if (item.event === 'error' || item.event === 'run_failed') failure = item;
      onEvent(item);
      if (cancelled && runDirectory) requestCancel();
    }
    let cancelSent = false;
    function requestCancel() {
      if (!runDirectory || cancelSent) return;
      cancelSent = true;
      const command = spawn(python, ['-B', '-m', 'bertopic_workflow', 'cancel', '--run-dir', runDirectory], {
        cwd, env: workerEnvironment(), shell: false, windowsHide: true, stdio: 'ignore',
      });
      command.on('error', () => {});
      // Force stop is a fallback. A interrupted stage has no committed stage marker.
      timer = setTimeout(terminate, cancellationGraceMs);
    }
    const abort = () => { cancelled = true; requestCancel(); };
    if (signal?.aborted) abort();
    signal?.addEventListener('abort', abort, { once: true });
    child.stdout.on('data', chunk => {
      try {
        buffer += decoder.write(chunk);
        if (Buffer.byteLength(buffer) > 2 * 1024 * 1024) throw new Error('Workflow event exceeded the IPC limit.');
        let newline;
        while ((newline = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, newline);
          buffer = buffer.slice(newline + 1);
          receive(line);
        }
      } catch (error) { failure = error; terminate(); }
    });
    child.stderr.on('data', chunk => {
      const remaining = Math.max(0, maxStderrBytes - stderrBytes);
      stderr += chunk.subarray(0, remaining).toString('utf8');
      stderrBytes += Math.min(chunk.length, remaining);
    });
    child.on('error', reject);
    child.on('close', code => {
      if (timer) clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      try {
        buffer += decoder.end();
        if (buffer.trim()) receive(buffer);
      } catch (error) { failure = error; }
      if (code !== 0 || failure || cancelled) {
        const error = new Error(cancelled ? 'Workflow cancelled.' : 'Workflow failed.');
        error.exitCode = code; error.event = failure; error.stderr = stderr;
        reject(error);
      } else resolve({ event: lastEvent, stderr });
    });
  });
}

export function analyze({ python, configPath, runId, resume = false, ...options }) {
  const args = ['run', '--config', path.resolve(configPath)];
  if (runId) args.push('--run-id', runId);
  if (resume) args.push('--resume');
  return invoke({ python, args, ...options });
}
