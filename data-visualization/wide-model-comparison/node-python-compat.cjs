// Use the already installed project Python for the chart checker's python3 call.
const childProcess = require('node:child_process');
const originalSpawnSync = childProcess.spawnSync;
childProcess.spawnSync = function (command, args, options) {
  const executable = command === 'python3' && process.env.FIRE_COMPARISON_PYTHON
    ? process.env.FIRE_COMPARISON_PYTHON : command;
  return originalSpawnSync.call(this, executable, args, options);
};
require('node:module').syncBuiltinESMExports();
