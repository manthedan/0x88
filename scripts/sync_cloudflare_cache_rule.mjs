#!/usr/bin/env node

import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { isDeepStrictEqual } from 'node:util';

const API_BASE = 'https://api.cloudflare.com/client/v4';
const DEFAULT_CONFIG = new URL('../cloudflare/app-origin-cache-rule.json', import.meta.url);

function usage(message) {
  if (message) console.error(message);
  console.error('Usage: node scripts/sync_cloudflare_cache_rule.mjs (--check | --apply) [--config PATH]');
  process.exit(2);
}

function parseArgs(argv) {
  let mode = null;
  let config = DEFAULT_CONFIG;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--check' || arg === '--apply') {
      if (mode) usage('Choose exactly one of --check or --apply.');
      mode = arg.slice(2);
    } else if (arg === '--config') {
      const path = argv[index + 1];
      if (!path) usage('--config requires a path.');
      config = new URL(path, `file://${process.cwd()}/`);
      index += 1;
    } else {
      usage(`Unknown argument: ${arg}`);
    }
  }
  if (!mode) usage('Choose --check or --apply.');
  return { mode, config };
}

function keychainToken(service) {
  if (process.platform !== 'darwin' || !service) return null;
  try {
    return (
      execFileSync('security', ['find-generic-password', '-s', service, '-w'], {
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'ignore'],
      }).trim() || null
    );
  } catch {
    return null;
  }
}

async function cloudflare(path, token, init = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
      ...init.headers,
    },
  });
  const body = await response.json().catch(() => null);
  if (!response.ok || !body?.success) {
    const detail =
      body?.errors?.map((error) => `${error.code ?? 'unknown'}: ${error.message ?? 'Cloudflare API error'}`).join('; ') ??
      `${response.status} ${response.statusText}`;
    const error = new Error(`Cloudflare API ${init.method ?? 'GET'} ${path} failed: ${detail}`);
    error.status = response.status;
    throw error;
  }
  return body.result;
}

function comparableRule(rule) {
  return {
    ref: rule?.ref,
    description: rule?.description,
    expression: rule?.expression,
    action: rule?.action,
    action_parameters: rule?.action_parameters,
    enabled: rule?.enabled,
  };
}

function rulesMatch(actual, desired) {
  return isDeepStrictEqual(comparableRule(actual), comparableRule(desired));
}

async function rulesetEntrypoint(zoneId, phase, token) {
  try {
    return await cloudflare(`/zones/${zoneId}/rulesets/phases/${phase}/entrypoint`, token);
  } catch (error) {
    if (error.status === 404) return null;
    throw error;
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const config = JSON.parse(await readFile(args.config, 'utf8'));
  if (config.schema !== 'lc0_browser.cloudflare_cache_rule.v1') throw new Error(`Unsupported config schema: ${config.schema}`);

  const token = process.env.CLOUDFLARE_API_TOKEN || keychainToken(config.keychainService);
  if (!token) {
    throw new Error(`Cloudflare API token unavailable. Set CLOUDFLARE_API_TOKEN or store it in macOS Keychain service ${config.keychainService}.`);
  }

  const zones = await cloudflare(`/zones?name=${encodeURIComponent(config.zone)}&status=active&per_page=2`, token);
  if (zones.length !== 1) throw new Error(`Expected one active Cloudflare zone named ${config.zone}; found ${zones.length}.`);
  const zoneId = zones[0].id;
  let ruleset = await rulesetEntrypoint(zoneId, config.ruleset.phase, token);
  let actual = ruleset?.rules?.find((rule) => rule.ref === config.rule.ref) ?? null;

  if (args.mode === 'check') {
    if (!actual) throw new Error(`Cloudflare Cache Rule ${config.rule.ref} is missing.`);
    if (!rulesMatch(actual, config.rule)) throw new Error(`Cloudflare Cache Rule ${config.rule.ref} has drifted from cloudflare/app-origin-cache-rule.json.`);
    console.log(JSON.stringify({ ok: true, mode: 'check', zone: config.zone, rulesetId: ruleset.id, ruleId: actual.id, ref: actual.ref }));
    return;
  }

  if (!ruleset) {
    ruleset = await cloudflare(`/zones/${zoneId}/rulesets`, token, {
      method: 'POST',
      body: JSON.stringify({ ...config.ruleset, rules: [config.rule] }),
    });
  } else if (!actual) {
    await cloudflare(`/zones/${zoneId}/rulesets/${ruleset.id}/rules`, token, {
      method: 'POST',
      body: JSON.stringify(config.rule),
    });
  } else if (!rulesMatch(actual, config.rule)) {
    await cloudflare(`/zones/${zoneId}/rulesets/${ruleset.id}/rules/${actual.id}`, token, {
      method: 'PATCH',
      body: JSON.stringify(config.rule),
    });
  }

  ruleset = await rulesetEntrypoint(zoneId, config.ruleset.phase, token);
  actual = ruleset?.rules?.find((rule) => rule.ref === config.rule.ref) ?? null;
  if (!actual || !rulesMatch(actual, config.rule)) throw new Error(`Cloudflare Cache Rule ${config.rule.ref} did not converge.`);
  console.log(JSON.stringify({ ok: true, mode: 'apply', zone: config.zone, rulesetId: ruleset.id, ruleId: actual.id, ref: actual.ref }));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
