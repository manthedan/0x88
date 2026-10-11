export function wasiBenchProtocol(engine) {
  // Reckless's argv bench is aggregate-only CLI mode; stdin bench is the
  // row-producing UCI mode used by runNativeBuiltin. Keep both sides aligned.
  return engine === 'reckless'
    ? { args: [engine], stdin: 'bench\nquit\n' }
    : { args: [engine, 'bench'], stdin: '' };
}
