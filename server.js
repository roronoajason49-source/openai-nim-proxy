// server.js - OpenAI -> NVIDIA NIM Proxy
// Optimized for NVIDIA DeepSeek V4 / reasoning models
// Keeps HIGH/MAX reasoning while avoiding unnecessary prompt-level
// reasoning instructions and correctly handling NVIDIA 202 pending requests.

import express from 'express';
import cors from 'cors';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// ============================================================
// NVIDIA CONFIG
// ============================================================

let rawBase = (process.env.NIM_API_BASE || '').trim();

if (
  !rawBase ||
  rawBase === 'undefined' ||
  rawBase === 'null' ||
  rawBase.length < 5
) {
  rawBase = 'https://integrate.api.nvidia.com/v1';
}

rawBase = rawBase
  .replace(/['"]/g, '')
  .replace(/\/chat\/completions\/?$/, '');

if (!rawBase.startsWith('http://') && !rawBase.startsWith('https://')) {
  rawBase = 'https://' + rawBase;
}

const NIM_API_BASE = rawBase.replace(/\/+$/, '');
const NIM_API_KEY = (process.env.NIM_API_KEY || '')
  .trim()
  .replace(/['"]/g, '');

// ============================================================
// REASONING CONFIG
// ============================================================

// IMPORTANT:
// Do NOT lower this to medium.
// DeepSeek V4 currently supports:
//   none / high / max
//
// HIGH is the default because MAX can substantially increase
// reasoning latency on a busy public endpoint.

const DEFAULT_REASONING_EFFORT = 'high';

// Maximum amount of time our proxy will wait for NVIDIA.
const UPSTREAM_TIMEOUT_MS = 5 * 60 * 1000;

// NVIDIA 202 polling interval.
// We don't want to hammer the API while a request is queued.
const STATUS_POLL_INTERVAL_MS = 1500;

// How many times we will poll a 202 request.
const MAX_STATUS_POLLS = Math.floor(
  UPSTREAM_TIMEOUT_MS / STATUS_POLL_INTERVAL_MS
);

// ============================================================
// MODEL MAPPING
// ============================================================

const MODEL_MAPPING = {
  // GLM
  'glm-5.3': 'z-ai/glm-5.3',
  'z-ai/glm-5.3': 'z-ai/glm-5.3',

  'glm-5.2': 'z-ai/glm-5.2',
  'z-ai/glm-5.1': 'z-ai/glm-5.2',

  // DeepSeek V4
  'deepseek-v4-pro-0813': 'deepseek-ai/deepseek-v4-pro-0813',
  'deepseek-ai/deepseek-v4-pro-0813':
    'deepseek-ai/deepseek-v4-pro-0813',

  'deepseek-v4-pro':
    'deepseek-ai/deepseek-v4-pro-0813',

  'deepseek-v4-flash-0731':
    'deepseek-ai/deepseek-v4-flash-0731',

  'deepseek-ai/deepseek-v4-flash-0731':
    'deepseek-ai/deepseek-v4-flash-0731',

  // Kimi
  'kimi-k3': 'moonshotai/kimi-k3',
  'moonshotai/kimi-k3': 'moonshotai/kimi-k3',

  'kimi-k2-thinking': 'moonshotai/kimi-k2-thinking',
  'kimi-k2.5': 'moonshotai/kimi-k2.5',

  // Other
  'inkling': 'thinkingmachines/inkling',
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash',
  'qwen-122b': 'qwen/qwen3.5-122b-a10b'
};

// ============================================================
// HELPERS
// ============================================================

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function getReasoningEffort(value, model) {
  const requested = String(value || '').toLowerCase();

  // DeepSeek V4 officially supports high/max.
  if (model.includes('deepseek-v4')) {
    if (requested === 'max') return 'max';
    return 'high';
  }

  // Kimi models that support high/max.
  if (model.includes('kimi')) {
    if (requested === 'max') return 'max';
    return 'high';
  }

  // GLM 5.3 supports low/high/max, but we intentionally
  // don't downgrade the default for roleplay.
  if (model.includes('glm')) {
    if (requested === 'max') return 'max';
    return 'high';
  }

  return requested === 'max' ? 'max' : 'high';
}

function normalizeMessages(messages) {
  const normalizedMessages = [];
  let systemFound = false;

  if (!Array.isArray(messages)) {
    return normalizedMessages;
  }

  for (const msg of messages) {
    if (!msg || !msg.content) continue;

    // NVIDIA expects string content for these models.
    if (typeof msg.content !== 'string') continue;

    if (!msg.content.trim()) continue;

    let role = String(msg.role || 'user').toLowerCase();

    if (role === 'developer') {
      role = 'system';
    }

    // NVIDIA expects system first.
    if (role === 'system') {
      if (!systemFound) {
        normalizedMessages.push({
          role: 'system',
          content: msg.content
        });

        systemFound = true;
      } else {
        // Don't create multiple system messages.
        normalizedMessages.push({
          role: 'user',
          content: msg.content
        });
      }

      continue;
    }

    // Merge adjacent same-role messages.
    if (
      normalizedMessages.length > 0 &&
      normalizedMessages[normalizedMessages.length - 1].role === role
    ) {
      normalizedMessages[
        normalizedMessages.length - 1
      ].content += '\n\n' + msg.content;
    } else {
      normalizedMessages.push({
        role,
        content: msg.content
      });
    }
  }

  return normalizedMessages;
}

// ============================================================
// NVIDIA STATUS POLLING
// ============================================================

async function pollNvidiaRequest(requestId, signal) {
  const statusUrl =
    `${NIM_API_BASE}/status/${encodeURIComponent(requestId)}`;

  for (let attempt = 0; attempt < MAX_STATUS_POLLS; attempt++) {
    if (signal?.aborted) {
      throw new Error('NVIDIA request polling aborted.');
    }

    await sleep(STATUS_POLL_INTERVAL_MS);

    const response = await fetch(statusUrl, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Accept': 'application/json'
      },
      signal
    });

    // Finished.
    if (response.status === 200) {
      return response;
    }

    // Still queued/running.
    if (response.status === 202) {
      continue;
    }

    // NVIDIA returned an actual error.
    const errorText = await response.text();

    throw new Error(
      `NVIDIA status polling failed (${response.status}): ${errorText}`
    );
  }

  throw new Error(
    'NVIDIA request remained pending for too long.'
  );
}

// ============================================================
// REQUEST NVIDIA
// ============================================================

async function requestNvidia(nimRequest, signal) {
  const upstreamResponse = await fetch(
    `${NIM_API_BASE}/chat/completions`,
    {
      method: 'POST',

      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
        'Accept': nimRequest.stream
          ? 'text/event-stream'
          : 'application/json'
      },

      body: JSON.stringify(nimRequest),
      signal
    }
  );

  // Normal immediate response.
  if (upstreamResponse.status !== 202) {
    return upstreamResponse;
  }

  // NVIDIA accepted the request but has not completed it.
  //
  // Example:
  // {
  //   "requestId": "..."
  // }
  //
  // We now poll /v1/status/{requestId} instead of immediately
  // treating 202 as an error.

  let pendingData;

  try {
    pendingData = await upstreamResponse.json();
  } catch {
    throw new Error(
      'NVIDIA returned HTTP 202 but no valid requestId.'
    );
  }

  const requestId =
    pendingData.requestId ||
    pendingData.request_id ||
    pendingData.id;

  if (!requestId) {
    throw new Error(
      'NVIDIA returned HTTP 202 without a requestId.'
    );
  }

  return await pollNvidiaRequest(requestId, signal);
}

// ============================================================
// HEALTH
// ============================================================

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    default_model: 'deepseek-ai/deepseek-v4-pro-0813',
    default_reasoning_effort: DEFAULT_REASONING_EFFORT,
    reasoning_modes: ['high', 'max']
  });
});

// ============================================================
// MODELS
// ============================================================

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// ============================================================
// CHAT COMPLETIONS
// ============================================================

app.post('/v1/chat/completions', async (req, res) => {
  const streamMode = Boolean(req.body?.stream);
  let heartbeat = null;

  // Track whether the connection is still usable.
  let clientClosed = false;

  req.on('close', () => {
    clientClosed = true;

    if (heartbeat) {
      clearInterval(heartbeat);
      heartbeat = null;
    }
  });

  try {
    const {
      model,
      messages,
      temperature,
      top_p,
      max_tokens,
      reasoning_effort
    } = req.body || {};

    const requestedModel =
      String(model || '').trim();

    const nimModel =
      MODEL_MAPPING[requestedModel] ||
      MODEL_MAPPING[requestedModel.toLowerCase()] ||
      'deepseek-ai/deepseek-v4-pro-0813';

    // ========================================================
    // MESSAGE NORMALIZATION
    // ========================================================

    const normalizedMessages =
      normalizeMessages(messages);

    if (normalizedMessages.length === 0) {
      return res.status(400).json({
        error: {
          message: 'No valid messages were provided.',
          type: 'invalid_request_error'
        }
      });
    }

    // ========================================================
    // REASONING
    // ========================================================

    const effort =
      getReasoningEffort(
        reasoning_effort,
        nimModel
      );

    // ========================================================
    // TEMPERATURE
    // ========================================================

    const parsedTemperature =
      Number.parseFloat(temperature);

    let safeTemperature;

    if (Number.isFinite(parsedTemperature)) {
      safeTemperature =
        Math.min(
          Math.max(parsedTemperature, 0),
          1
        );
    } else {
      // Good default for roleplay.
      safeTemperature =
        nimModel.includes('kimi') ? 1.0 : 0.8;
    }

    // ========================================================
    // TOP P
    // ========================================================

    const parsedTopP =
      Number.parseFloat(top_p);

    const safeTopP =
      Number.isFinite(parsedTopP)
        ? Math.min(Math.max(parsedTopP, 0), 1)
        : 0.95;

    // ========================================================
    // MAX TOKENS
    // ========================================================
    //
    // IMPORTANT:
    // Do NOT force every request to 8192.
    //
    // DeepSeek V4 reasoning + visible answer use the same
    // max_tokens budget.
    //
    // We preserve Janitor's requested value when supplied.
    //
    // NVIDIA DeepSeek V4 Pro supports up to 16384 for this
    // endpoint.

    const requestedMaxTokens =
      Number.parseInt(max_tokens, 10);

    let safeMaxTokens;

    if (
      Number.isFinite(requestedMaxTokens) &&
      requestedMaxTokens > 0
    ) {
      safeMaxTokens =
        Math.min(requestedMaxTokens, 16384);
    } else {
      safeMaxTokens = 8192;
    }

    // ========================================================
    // BUILD NIM REQUEST
    // ========================================================

    const nimRequest = {
      model: nimModel,
      messages: normalizedMessages,
      temperature: safeTemperature,
      top_p: safeTopP,
      max_tokens: safeMaxTokens,
      stream: streamMode
    };

    // ========================================================
    // MODEL-SPECIFIC REASONING
    // ========================================================

    if (nimModel.includes('deepseek-v4')) {
      // NVIDIA's native DeepSeek V4 API supports:
      // high / max
      //
      // No artificial <think> prompt.
      // No duplicate reasoning setting.

      nimRequest.reasoning_effort = effort;

    } else if (nimModel.includes('kimi')) {
      nimRequest.reasoning_effort = effort;

    } else if (nimModel.includes('glm')) {
      // Keep native GLM thinking enabled.
      nimRequest.chat_template_kwargs = {
        enable_thinking: true
      };

      // GLM 5.3 supports reasoning_effort.
      nimRequest.reasoning_effort = effort;

    } else {
      // For other reasoning-capable models, preserve high/max.
      nimRequest.reasoning_effort = effort;
    }

    // ========================================================
    // EARLY STREAM INITIALIZATION
    // ========================================================

    if (streamMode) {
      res.status(200);

      res.setHeader(
        'Content-Type',
        'text/event-stream; charset=utf-8'
      );

      res.setHeader(
        'Cache-Control',
        'no-cache, no-transform'
      );

      res.setHeader(
        'Connection',
        'keep-alive'
      );

      res.setHeader(
        'X-Accel-Buffering',
        'no'
      );

      if (typeof res.flushHeaders === 'function') {
        res.flushHeaders();
      }

      // Small initial padding to defeat intermediary buffering.
      res.write(': heartbeat-init\n\n');

      const initChunk = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: nimModel,
        choices: [
          {
            index: 0,
            delta: {
              role: 'assistant',
              content: ''
            },
            finish_reason: null
          }
        ]
      };

      res.write(
        `data: ${JSON.stringify(initChunk)}\n\n`
      );

      // Keep Janitor connection alive while NVIDIA queues
      // or performs inference.
      heartbeat = setInterval(() => {
        if (
          !res.writableEnded &&
          !clientClosed
        ) {
          try {
            res.write(': keep-alive\n\n');
          } catch {
            clearInterval(heartbeat);
            heartbeat = null;
          }
        }
      }, 2000);
    }

    // ========================================================
    // NVIDIA REQUEST TIMEOUT
    // ========================================================

    const controller =
      new AbortController();

    const timeoutId =
      setTimeout(() => {
        controller.abort();
      }, UPSTREAM_TIMEOUT_MS);

    let upstreamResponse;

    try {
      upstreamResponse =
        await requestNvidia(
          nimRequest,
          controller.signal
        );
    } catch (fetchErr) {
      clearTimeout(timeoutId);

      if (heartbeat) {
        clearInterval(heartbeat);
        heartbeat = null;
      }

      const errorMessage =
        fetchErr?.name === 'AbortError'
          ? 'NVIDIA request timed out after 5 minutes.'
          : fetchErr.message;

      if (streamMode) {
        if (!res.writableEnded) {
          const errorChunk = {
            id: `error-${Date.now()}`,
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            choices: [
              {
                index: 0,
                delta: {
                  content:
                    `\n\n*[NVIDIA Error: ${errorMessage}]*`
                },
                finish_reason: 'stop'
              }
            ]
          };

          res.write(
            `data: ${JSON.stringify(errorChunk)}\n\n`
          );

          res.write('data: [DONE]\n\n');
          return res.end();
        }

        return;
      }

      return res.status(504).json({
        error: {
          message: errorMessage,
          type: 'gateway_timeout'
        }
      });
    }

    clearTimeout(timeoutId);

    // ========================================================
    // NVIDIA ERROR
    // ========================================================

    if (!upstreamResponse.ok) {
      if (heartbeat) {
        clearInterval(heartbeat);
        heartbeat = null;
      }

      const errText =
        await upstreamResponse.text();

      if (streamMode) {
        if (!res.writableEnded) {
          const errorChunk = {
            id: `error-${Date.now()}`,
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            choices: [
              {
                index: 0,
                delta: {
                  content:
                    `\n\n*[NVIDIA Error ${upstreamResponse.status}: ${errText}]*`
                },
                finish_reason: 'stop'
              }
            ]
          };

          res.write(
            `data: ${JSON.stringify(errorChunk)}\n\n`
          );

          res.write('data: [DONE]\n\n');

          return res.end();
        }

        return;
      }

      return res
        .status(upstreamResponse.status)
        .json({
          error: {
            message: errText,
            code: upstreamResponse.status
          }
        });
    }

    // ========================================================
    // STREAMING RESPONSE
    // ========================================================

    if (streamMode) {
      const decoder =
        new TextDecoder();

      let buffer = '';

      let reasoningStarted = false;
      let reasoningClosed = false;

      for await (
        const chunk of upstreamResponse.body
      ) {
        if (clientClosed || res.writableEnded) {
          break;
        }

        buffer += decoder.decode(
          chunk,
          { stream: true }
        );

        const lines =
          buffer.split('\n');

        buffer =
          lines.pop() || '';

        for (let line of lines) {
          line = line.trim();

          if (!line) continue;

          if (!line.startsWith('data:')) {
            continue;
          }

          const payload =
            line.slice(5).trim();

          if (payload === '[DONE]') {
            continue;
          }

          try {
            const data =
              JSON.parse(payload);

            const delta =
              data?.choices?.[0]?.delta;

            if (delta) {
              let reasoning =
                delta.reasoning_content ||
                delta.reasoning ||
                '';

              let content =
                delta.content || '';

              // ------------------------------------------------
              // Convert upstream reasoning into visible
              // <think> tags for Janitor.
              // ------------------------------------------------

              let output = '';

              if (reasoning) {
                if (!reasoningStarted) {
                  output += '<think>\n';
                  reasoningStarted = true;
                  reasoningClosed = false;
                }

                output += reasoning;
              }

              if (content) {
                if (
                  reasoningStarted &&
                  !reasoningClosed
                ) {
                  output +=
                    '\n</think>\n\n';

                  reasoningClosed = true;
                  reasoningStarted = false;
                }

                output += content;
              }

              // If upstream already sends <think>,
              // don't duplicate the tags.
              if (
                output.includes('<think>') &&
                output.includes('</think>')
              ) {
                reasoningStarted = false;
                reasoningClosed = true;
              }

              data.choices[0].delta.con
