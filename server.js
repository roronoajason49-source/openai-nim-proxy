// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Dynamic string constructor to prevent mobile app rendering glitches
const b3 = String.fromCharCode(96, 96, 96);

// Automatically cleans and repairs broken or messy Env Variables
let rawBase = (process.env.NIM_API_BASE || '').trim();
if (!rawBase || rawBase === 'undefined' || rawBase === 'null' || rawBase.length < 5) {
  rawBase = 'https://integrate.api.nvidia.com/v1';
}
rawBase = rawBase.replace(/['"]/g, '');
rawBase = rawBase.replace(/\/chat\/completions\/?$/, '');
if (!rawBase.startsWith('http://') && !rawBase.startsWith('https://')) {
  rawBase = 'https://' + rawBase;
}
rawBase = rawBase.replace(/\/+$/, '');
const NIM_API_BASE = rawBase;

let rawKey = (process.env.NIM_API_KEY || '').trim();
const NIM_API_KEY = rawKey.replace(/['"]/g, '');

const SHOW_REASONING = true; 

// Model mapping
const MODEL_MAPPING = {
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash',
  'stepfun-ai/step-3.7-flash': 'stepfun-ai/step-3.7-flash', 
  'glm-5.2': 'z-ai/glm-5.2',
  'z-ai/glm-5.2': 'z-ai/glm-5.2',
  'minimax-m3': 'minimaxai/minimax-m3',
  'minimaxai/minimax-m3': 'minimaxai/minimax-m3',
  'minimax-m2.7': 'minimaxai/minimax-m2.7',
  'qwen-122b': 'qwen/qwen3.5-122b-a10b',         
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro',
  'z-ai/glm-5.1': 'z-ai/glm-5.2', 
  'glm-5.1': 'z-ai/glm-5.2'
};

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'OpenAI to NVIDIA NIM Proxy', reasoning_display: SHOW_REASONING });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, stream } = req.body;
    let nimModel = MODEL_MAPPING[model] || MODEL_MAPPING[model?.toLowerCase()] || 'stepfun-ai/step-3.7-flash';
    
    const normalizedMessages = [];
    let isFirstSystem = true;

    // Forces reasoning to trigger and keeps character dialogue outside the thought block
    const FORCE_THINKING_PROMPT = "\n\n[CRITICAL SYSTEM DIRECTIVE: You are an advanced reasoning model. You MUST ALWAYS start every single response by thinking. Wrap your internal thoughts, character logic, and planning strictly inside <think> and </think> tags. NEVER skip the <think> phase, even for short responses. NEVER put actual roleplay dialogue inside the <think> tags. Write your actual roleplay response only AFTER closing the </think> tag.]";

    for (const msg of messages) {
      if (!msg.content || typeof msg.content !== 'string' || msg.content.trim() === '') continue;
      
      let role = msg.role.toLowerCase();
      
      if (role === 'system') {
        if (isFirstSystem && normalizedMessages.length === 0) {
          normalizedMessages.push({ role: 'system', content: msg.content + FORCE_THINKING_PROMPT });
          isFirstSystem = false;
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
    
    if (normalizedMessages.length > 1 && normalizedMessages[1].role === 'assistant') {
      normalizedMessages.splice(1, 0, { role: 'user', content: 'Hello.' });
    }

    const safe_temp = (parseFloat(temperature) > 0) ? parseFloat(temperature) : 0.6;
    
    const nimRequest = {
      model: nimModel,
      messages: normalizedMessages,
      temperature: safe_temp,
      top_p: req.body.top_p ?? 1.0,
      max_tokens: 4096, 
      stream: stream || false
    };
    
    // Set proper API hardware keys for each model family
    if (nimModel.includes('step-3.7')) {
      nimRequest.reasoning_effort = "high";
    } else if (nimModel.includes('glm-5.2')) {
      // 🔥 THE FIX: Override GLM-5.2 to absolute maximum reasoning depth & force thinking
      nimRequest.reasoning_effort = "max"; 
      nimRequest.chat_template_kwargs = { 
        enable_thinking: true, 
        reasoning_effort: "max" 
      };
    } else if (nimModel.includes('minimax')) {
      nimRequest.reasoning_effort = "high";
      nimRequest.thinking = { type: "enabled" }; 
    } else if (nimModel.includes('deepseek-v4')) {
      nimRequest.chat_template_kwargs = { enable_thinking: true, thinking: true };
    }
    
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no'); 
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
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
                  choices: [{ index: 0, delta: { content: '\n' + b3 + '\n\n' }, finish_reason: 'stop' }]
                };
                res.write(`data: ${JSON.stringify(closeChunk)}\n\n`);
                reasoningStarted = false;
              }
              res.write('data: [DONE]\n\n'); 
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const delta = data.choices[0].delta;
                
                let reasoning = delta.reasoning_content || delta.reasoning || '';
                let content = delta.content || '';
                
                // Track standard inline thought tags
                if (content.includes('<think>')) {
                  content = content.replace(/<think>/g, b3 + 'thought\n');
                  reasoningStarted = true;
                }
                if (content.includes('</think>')) {
                  content = content.replace(/<\/think>/g, '\n' + b3 + '\n\n');
                  reasoningStarted = false;
                }
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning) {
                    if (!reasoningStarted) {
                      combinedContent += b3 + 'thought\n';
                      reasoningStarted = true;
                    }
                    combinedContent += reasoning;
                  } else if (content) {
                    // 🔥 THE FIX: Transition out of thinking mode only when REAL dialogue text arrives
                    if (reasoningStarted) {
                      combinedContent += '\n' + b3 + '\n\n';
                      reasoningStarted = false;
                    }
                    combinedContent += content;
                  }
                  
                  data.choices[0].delta.content = combinedContent;
                } else {
                  data.choices[0].delta.content = content.replace(/<think>/g, '').replace(/<\/think>/g, '');
                }
                
                delete data.choices[0].delta.reasoning_content;
                delete data.choices[0].delta.reasoning;
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n\n');
            }
          }
        }
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream processing interruption:', err);
        res.end();
      });
    } else {
      res.json({}); 
    }
    
  } catch (error) {
    const statusCode = error.response?.status || 500;
    let exactMessage = error.message;

    if (error.response?.data) {
      if (typeof error.response.data === 'object') {
        exactMessage = JSON.stringify(error.response.data);
      } else {
        exactMessage = error.response.data;
      }
    }

    if (req.body && req.body.stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');
      
      let chatMessage = `\n\n*[System Error ${statusCode}: NVIDIA rejected the request.*\n\n**REASON:**\n\`${exactMessage}\`]*`;

      const errorChunk = {
        id: `error-${Date.now()}`,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: req.body.model || 'proxy-error',
        choices: [{ index: 0, delta: { content: chatMessage }, finish_reason: 'stop' }]
      };
      
      res.write(`data: ${JSON.stringify(errorChunk)}\n\n`);
      res.write('data: [DONE]\n\n');
      return res.end();
    } else {
      res.status(statusCode).json({
        error: { message: exactMessage, type: 'proxy_error', code: statusCode }
      });
    }
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
});
