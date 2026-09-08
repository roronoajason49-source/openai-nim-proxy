// server.js - Universal OpenAI to NVIDIA NIM Proxy (Strict Template Override)
import express from 'express';
import cors from 'cors';

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

// Model mapping dictionary
const MODEL_MAPPING = {
  // GLM Models
  'glm-5.3': 'z-ai/glm-5.3',
  'z-ai/glm-5.3': 'z-ai/glm-5.3',
  'glm-5.2': 'z-ai/glm-5.2',
  'z-ai/glm-5.2': 'z-ai/glm-5.2',
  'glm-5.1': 'z-ai/glm-5.2',
  'z-ai/glm-5.1': 'z-ai/glm-5.2',

  // DeepSeek Models 
  'deepseek-v4-pro-0813': 'deepseek-ai/deepseek-v4-pro-0813',
  'deepseek-ai/deepseek-v4-pro-0813': 'deepseek-ai/deepseek-v4-pro-0813',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro-0813',
  'deepseek-ai/deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro-0813',
  'deepseek-v4-flash-0731': 'deepseek-ai/deepseek-v4-flash-0731',
  'deepseek-ai/deepseek-v4-flash-0731': 'deepseek-ai/deepseek-v4-flash-0731',
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash-0731',
  'deepseek-ai/deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash-0731',

  // Moonshot Kimi Models
  'kimi-k3': 'moonshotai/kimi-k3',
  'moonshotai/kimi-k3': 'moonshotai/kimi-k3',
  'kimi': 'moonshotai/kimi-k3',
  'kimi-k2-thinking': 'moonshotai/kimi-k2-thinking',
  'moonshotai/kimi-k2-thinking': 'moonshotai/kimi-k2-thinking',
  'kimi-k2.5': 'moonshotai/kimi-k2.5',
  'moonshotai/kimi-k2.5': 'moonshotai/kimi-k2.5',
  'kimi-k2.6': 'moonshotai/kimi-k2.6',

  // Other NIM Models
  'inkling': 'thinkingmachines/inkling',
  'thinkingmachines/inkling': 'thinkingmachines/inkling',
  'minimax-m3': 'minimaxai/minimax-m3',
  'minimaxai/minimax-m3': 'minimaxai/minimax-m3',
  'minimax-m2.7': 'minimaxai/minimax-m2.7',
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash',
  'stepfun-ai/step-3.7-flash': 'stepfun-ai/step-3.7-flash',
  'qwen-122b': 'qwen/qwen3.5-122b-a10b'
};

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    default_model: 'z-ai/glm-5.3',
    reasoning_display: SHOW_REASONING
  });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((model) => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  const streamMode = req.body?.stream ?? false;

  try {
    const { model, messages, temperature } = req.body;
    const nimModel = MODEL_MAPPING[model] || MODEL_MAPPING[model?.toLowerCase()] || 'z-ai/glm-5.3';

    const normalizedMessages = [];
    let systemFound = false;

    // Strict two-phase formatting template to prevent the model from cheating
    const FORCE_THINKING_PROMPT = `

[CRITICAL FORMATTING REQUIREMENT]
You MUST process your response in two distinct phases:
Phase 1: Internal Planning. You must write your reasoning, character analysis, emotional state, and plot planning strictly inside <think> and </think> tags.
Phase 2: Final Output. You must write your actual dialogue, actions, and roleplay AFTER the </think> tag.
NEVER place your character's spoken dialogue or final actions inside the <think> tags.`;

    if (Array.isArray(messages)) {
      for (const msg of messages) {
        if (!msg.content || typeof msg.content !== 'string' || msg.content.trim() === '') continue;
        let role = msg.role.toLowerCase();

        if (role === 'developer') role = 'system';

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
        content: 'You are an expert roleplay assistant.' + FORCE_THINKING_PROMPT
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

    if (nimModel.includes('deepseek')) {
      nimRequest.chat_template_kwargs = { thinking: true, reasoning_effort: "high" };
    } else if (isKimi) {
      nimRequest.reasoning_effort = "high";
      nimRequest.chat_template_kwargs = { thinking: true };
    } else if (nimModel.includes('glm')) {
      nimRequest.chat_template_kwargs = { enable_thinking: true, clear_thinking: false };
    } else if (nimModel.includes('minimax')) {
      nimRequest.chat_template_kwargs = { thinking_mode: "enabled" };
    } else if (nimModel.includes('inkling')) {
      nimRequest.chat_template_kwargs = { reasoning_effort: "high" };
    } else {
      nimRequest.reasoning_effort = "high";
    }

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
      let channelReasoningActive = false;

      for await (const chunk of upstreamResponse.body) {
        buffer += decoder.decode(chunk, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (let line of lines) {
          line = line.trim();
          if (!line) continue;

          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              if (channelReasoningActive) {
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
            }

            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const delta = data.choices[0].delta;
                let reasoning = delta.reasoning_content || delta.reasoning || '';
                let content = delta.content || '';

                if (SHOW_REASONING) {
                  let streamText = '';

                  // Handle dedicated NIM reasoning channel
                  if (reasoning) {
                    if (!channelReasoningActive) {
                      streamText += '<think>\n';
                      channelReasoningActive = true;
                    }
                    reasoning = reasoning.replace(/<think>/gi, '').replace(/<\/think>/gi, '');
                    streamText += reasoning;
                  }

                  if (content) {
                    // Auto-close native backend reasoning channel if it switches to standard content
                    if (channelReasoningActive) {
                      streamText += '\n</think>\n\n';
                      channelReasoningActive = false;
                    }

                    // Normalize tags so Janitor AI can read them correctly
                    content = content.replace(/<thought>/gi, '<think>').replace(/<\/thought>/gi, '</think>');

                    // Prevent duplicate tags if it outputs </think> right when we did
                    if (streamText.includes('</think>') && content.trim().startsWith('<think>')) {
                      content = content.replace('<think>', '');
                    }

                    streamText += content;
                  }

                  data.choices[0].delta.content = streamText;
                } else {
                  data.choices[0].delta.content = content.replace(/<think>[\s\S]*?<\/think>/g, '').replace(/<thought>[\s\S]*?<\/thought>/g, '');
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

      if (channelReasoningActive) {
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
      
      const upstreamJson = await upstreamResponse.json();
      
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: upstreamJson.choices?.map((choice) => {
          let fullContent = choice.message?.content || '';
          let reasoning = choice.message?.reasoning_content || choice.message?.reasoning || '';

          if (SHOW_REASONING) {
            fullContent = fullContent.replace(/<thought>/gi, '<think>').replace(/<\/thought>/gi, '</think>');

            if (reasoning) {
              reasoning = reasoning.replace(/<think>/gi, '').replace(/<\/think>/gi, '');
              fullContent = fullContent.replace(/^[\s\n]*<think>/i, '');
              fullContent = '<think>\n' + reasoning.trim() + '\n</think>\n\n' + fullContent;
            }
          } else {
            fullContent = fullContent.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
          }

          return {
            index: choice.index,
            message: { role: choice.message?.role || 'assistant', content: fullContent },
            finish_reason: choice.finish_reason || 'stop'
          };
        }) || [],
        usage: upstreamJson.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      
      return res.json(openaiResponse);
    }
  } catch (error) {
    return res.status(500).json({ error: { message: error.message, type: 'proxy_error' } });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: 'Endpoint not found', code: 404 } });
});

if (process.env.NODE_ENV !== 'production' || !process.env.VERCEL) {
  app.listen(PORT, () => {
    console.log(`Proxy running on port ${PORT}`);
  });
}

export default app;
    
