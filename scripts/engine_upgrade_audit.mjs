#!/usr/bin/env node
import { audit, parseArgs, printReport, saveReport, usage } from './engine_upgrade_audit/core.mjs';
import { hasFailures } from './engine_upgrade_audit/parsers.mjs';

try {
  const { engine, options } = parseArgs(process.argv.slice(2));
  if (options.help) console.log(usage);
  else {
    const report = await audit(engine, options);
    printReport(report);
    if (options.json) saveReport(report, options.json);
    if (hasFailures(report.gates)) process.exitCode = 1;
  }
} catch (error) {
  console.error(`${error.message}\n${usage}`);
  process.exitCode = 2;
}
