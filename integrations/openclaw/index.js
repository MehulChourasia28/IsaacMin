import { readFileSync } from 'node:fs';

// This trusted plugin has no executable user/model input. Python owns validation,
// operation identities, source access and the native worker queue.
export default {
  id: 'isaacmin',
  register(api) {
    const request = JSON.parse(readFileSync(api.pluginConfig.requestFile, 'utf8'));
    const endpoint = process.env.ISAACMIN_BRIDGE_URL;
    if (!/^http:\/\/127\.0\.0\.1:\d+$/.test(endpoint || '')) throw Error('Local bridge required');
    api.on('before_prompt_build', () => ({
      appendSystemContext: request.system,
      toolsAllow: request.tools.map(t => t.function.name),
    }));
    for (const { function: tool } of request.tools) {
      api.registerTool({
        name: tool.name, label: tool.name, description: tool.description,
        parameters: tool.parameters,
        async execute(id, args, signal) {
          const response = await fetch(endpoint + '/tool', {
            method: 'POST', signal,
            headers: { 'Content-Type': 'application/json',
              Authorization: 'Bearer ' + process.env.ISAACMIN_BRIDGE_TOKEN },
            body: JSON.stringify({ id, name: tool.name, arguments: args }),
          });
          const result = await response.json();
          if (!response.ok) throw Error(result.error || 'Tool bridge rejected the call');
          return { content: [{ type: 'text', text: JSON.stringify(result) }], details: result };
        },
      }, { optional: true });
    }
  },
};
