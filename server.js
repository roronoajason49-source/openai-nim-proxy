// server.js - OpenAI to NVIDIA NIM API Proxy with Native Fetch & Zero-Buffering
(async () => {
  const expressModule = await import('express');
  const express = expressModule.default || expressModule;
  const corsModule = await import('cors');
  const cors = corsModule.default || corsModule;

  const app = express();
  const PORT = process.env.PORT || 3000;

  // Middleware
  app.use(cors());
  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ limit: '50mb', extended: true }));

  // Environment variable sanitization
  let rawBase = (process.env.NIM_API_BASE || '').trim();
  if (!rawBase || rawBase === 'undefined' || rawBase === 'null' || rawBase.length < 5) {
    rawBase = 'https://integrate.api.nvidia.com/v1';
  }
  rawBase = rawBase.replace(/['"]/g, '').replace(/\/chat\/completions\/?$/, '');
  if (!rawBase.startsWith('http://') && !rawBase.startsWith('https://')) {
    rawBase = 'https://' + rawBase;
  }
  const NIM_API_BASE = rawBase.replace(/\/+$/, '');

  const NIM_API_KEY = (process.env.NIM_API_KEY || '').trim().replace(/['"]/g, '');
  const SHOW_REASONING = true; 

  const MODEL_MAPPING = {
    'deepseek-v4-pro-0813': 'deepseek-ai/deepseek-v4-pro-0813',
    'deepseek-ai/deepseek-v4-pro-0813': 'deepseek-ai/deepseek-v4-pro-0813',
    'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro-0813',
    'deepseek-ai/deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro-0813',
    'deepseek-v4-flash-0731': 'deepseek-ai/deepseek-v4-flash-0731',
    'deepseek-ai/deepseek-v4-flash-0731': 'deepseek-ai/deepseek-v4-flash-0731',
    'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash-0731',
    'deepseek-ai/deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash-0731',
    'kimi-k3': 'moonshotai/kimi-k3',
    'moonshotai/kimi-k3': 'moonshotai/kimi-k3',
    'kimi': 'moonshotai/kimi-k3',
    'kimi-k2-thinking': 'moonshotai/kimi-k2-thinking',
    'moonshotai/kimi-k2-thinking': 'moonshotai/kimi-k2-thinking',
    'kimi-k2.5': 'moonshotai/kimi-k2.5',
    'moonshotai/kimi-k2.5': 'moonshotai/kimi-k2.5',
    'inkling': 'thinkingmachines/inkling',
    'thinkingmachines/inkling': 'thinkingmachines/inkling',
    'minimax-m3': 'minimaxai/minimax-m3',
    'minimaxai/minimax-m3': 'minimaxai/minimax-m3',
    'minimax-m2.7': 'minimaxai/minimax-m2.7',
    'step-3.7-flash': 'stepfun-ai/step-3.7-flash',
    'stepfun-ai/step-3.7-flash': 'stepfun-ai/step-3.7-flash', 
    'glm-5.2': 'z-ai/glm-5.2',
    'z-ai/glm-5.2': 'z-ai/glm-5.2',
    'qwen-122b': 'qwen/qwen3.5-122b-a10b'
  };

  app.get('/health', (req, res) => {
    res.json({ 
      status: 'ok', 
      service: 'OpenAI to NVIDIA NIM Proxy', 
      default_model: 'moonshotai/kimi-k3', 
      reasoning_display: SHOW_REASONING 
    });
  });

  app.get('/v1/models', (req, res) => {
    const models = Object.keys(MODEL_MAPPING).map(model => ({
      id: model, 
      object: 'model', 
      created: Date.now(), 
      owned_by: 'nvidia-nim-proxy'
    }));
    res.json({ object: 'list', data: models });
  });

  app.post('/v1/chat/completions', async (req, res) => {
    const streamMode = req.body?.stream || false;

    try {
      const { model, messages, temperature } = req.body;
      let nimModel = MODEL_MAPPING[model] || MODEL_MAPPING[model?.toLowerCase()] || 'moonshotai/kimi-k3';
      
      const normalizedMessages = [];
      let systemFound = false;

      const FORCE_THINKING_PROMPT = "\n\n[CRITICAL SYSTEM DIRECTIVE: You are an advanced reasoning model. You MUST always think step-by-step before answering. Write out your detailed internal thoughts and roleplay planning inside <think> and </think> tags. Never skip the <think> phase. Write your dialogue and actions only AFTER closing the </think> tag.]";

      if (Array.isArray(messages)) {
        for (const msg of messages) {
          if (!msg.content || typeof msg.content !== 'string' || msg.content.trim() === '') continue;
          
          let role = msg.role.toLowerCase();
          
          if (role === 'system') {
            if (!systemFound) {
              normalizedMessages.push({ role: 'system', content: msg.content + FORCE_THINKING_PROMPT });
              systemFound = true;
              continue;
            } else {
              role = 'user';
            }
          }
          
          if (normalizedMessages.length > 0 && normalizedMessages[normalizedMessages.length - 1].role === role) {
            normalizedMessages[normalizedMessages.length - 1].content += '\n\n' + msg.content;
          } else {
            normalizedMessages.push({ role, content: msg.content });
          }
        }
      }

      if (!systemFound) {
        normalizedMessages.unshift({
          role: 'system',
          content: 'You are an expert AI assistant.' + FORCE_THINKING_PROMPT
        });
      }
      
      if (normalizedMessages.length > 1 && normalizedMessages[1].role === 'assistant') {
        normalizedMessages.splice(1, 0, { role: 'user', content: 'Hello.' });
      }

      const isKimi = nimModel.includes('kimi') || nimModel.includes('moonshot');
      const safe_temp = isKimi ? 1.0 : (parseFloat(temperature) > 0 ? parseFloat(temperature) : 0.7);
      
      const nimRequest = {
        model: nimModel,
        messages: normalizedMessages,
        temperature: safe_temp,
        top_p: req.body.top_p ?? 0.95,
        max_tokens: req.body.max_tokens ? Math.max(req.body.max_tokens, 8192) : 8192, 
        stream: streamMode
      };

      if (isKimi) {
        nimRequest.reasoning_effort = "max";
      } else if (nimModel.includes('deepseek-v4') || nimModel.includes('deepseek')) {
        nimRequest.reasoning_effort = "high";
        nimRequest.thinking = { type: "enabled" };
        nimRequest.chat_template_kwargs = { enable_thinking: true, thinking: true };
      } else if (nimModel.includes('inkling')) {
        nimRequest.reasoning_effort = "high";
        nimRequest.chat_template_kwargs = { enable_thinking: true };
      } else if (nimModel.includes('minimax')) {
        nimRequest.thinking = { type: "enabled" }; 
        nimRequest.reasoning_effort = "high";
      }

      // Native fetch bypassing Axios stream zlib deadlocks
      const upstreamResponse = await fetch(`${NIM_API_BASE}/chat/completions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
          'Accept': streamMode ? 'text/event-stream' : 'application/json',
          'Accept-Encoding': 'identity'
        },
        body: JSON.stringify(nimRequest)
      });

      if (!upstreamResponse.ok) {
        const errText = await upstreamResponse.text();
        return res.status(upstreamResponse.status).json({
          error: { message: errText, code: upstreamResponse.status }
        });
      }

      if (streamMode) {
        res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
        res.setHeader('Cache-Control', 'no-cache, no-transform');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no');
        if (typeof res.flushHeaders === 'function') {
          res.flushHeaders();
        }

        const initChunk = {
          id: `chatcmpl-${Date.now()}`,
          object: 'chat.completion.chunk',
          created: Math.floor(Date.now() / 1000),
          model: nimModel,
          choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }]
        };
        res.write(`data: ${JSON.stringify(initChunk)}\n\n`);

        const decoder = new TextDecoder();
        let buffer = '';
        let reasoningStarted = false;
        let inChannelReasoning = false;

        for await (const chunk of upstreamResponse.body) {
          buffer += decoder.decode(chunk, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (let line of lines) {
            line = line.trim();
            if (!line) continue;

            if (line.startsWith('data: ')) {
              if (line.includes('[DONE]')) {
                if (reasoningStarted) {
                  const closeChunk = {
                    id: `chatcmpl-${Date.now()}`,
                    object: 'chat.completion.chunk',
                    created: Math.floor(Date.now() / 1000),
                    model: nimModel,
                    choices: [{ index: 0, delta: { content: '\n</think>\n\n' }, finish_reason: 'stop' }]
                  };
                  res.write(`data: ${JSON.stringify(closeChunk)}\n\n`);
                  reasoningStarted = false;
                }
                res.write('data: [DONE]\n\n');
                return res.end();
              }

              try {
                const data = JSON.parse(line.slice(6));
                if (data.choices?.[0]?.delta) {
                  const delta = data.choices[0].delta;
                  let reasoning = delta.reasoning_content || delta.reasoning || '';
                  let content = delta.content || '';

                  if (content) {
                    content = content.replace(/<thought>/gi, '<think>')
                                     .replace(/<\/thought>/gi, '</think>');
                  }

                  if (SHOW_REASONING) {
                    let streamText = '';

                    if (reasoning) {
                      if (!reasoningStarted) {
                        streamText += '<think>\n';
                        reasoningStarted = true;
                        inChannelReasoning = true;
                      }
                      streamText += reasoning;
                    }

                    if (content) {
                      if (inChannelReasoning && reasoningStarted) {
                        streamText += '\n</think>\n\n';
                        reasoningStarted = false;
                        inChannelReasoning = false;
                      }
                      streamText += content;
                    }

                    data.choices[0].delta.content = streamText;
                  } else {
                    data.choices[0].delta.content = content.replace(/<think>[\s\S]*?<\/think>/g, '');
                  }

                  delete data.choices[0].delta.reasoning_content;
                  delete data.choices[0].delta.reasoning;
                }
                res.write(`data: ${JSON.stringify(data)}\n\n`);
              } catch {
                res.write(line + '\n\n');
              }
            }
          }
        }

        if (reasoningStarted) {
          const closeChunk = {
            id: `chatcmpl-${Date.now()}`,
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model: nimModel,
            choices: [{ index: 0, delta: { content: '\n</think>\n\n' }, finish_reason: 'stop' }]
          };
          res.write(`data: ${JSON.stringify(closeChunk)}\n\n`);
        }
        res.write('data: [DONE]\n\n');
        return res.end();
      } else {
        const json = await upstreamResponse.json();
        return res.json(json);
      }
    } catch (error) {
      return res.status(500).json({ error: { message: error.message, type: 'proxy_error' } });
    }
  });

  app.all('*', (req, res) => {
    res.status(404).json({ error: { message: 'Endpoint not found', type: 'invalid_request_error', code: 404 } });
  });

  app.listen(PORT, () => {
    console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  });
})();
