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

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = true; 

// Model mapping
const MODEL_MAPPING = {
  'glm-5.2': 'z-ai/glm-5.2',
  'z-ai/glm-5.2': 'z-ai/glm-5.2',
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash', 
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
    const { model, messages, temperature, max_tokens, stream } = req.body;
    let nimModel = MODEL_MAPPING[model] || MODEL_MAPPING[model?.toLowerCase()] || 'z-ai/glm-5.2';
    
    // 🔥 THE FIX: Aggressive Message Normalizer
    const normalizedMessages = [];
    
    for (const msg of messages) {
      // 1. Drop completely empty messages
      if (!msg.content || typeof msg.content !== 'string' || msg.content.trim() === '') continue;
      
      let role = msg.role.toLowerCase();
      
      // 2. Convert mid-chat 'system' prompts (like Author's Notes) into 'user' prompts
      if (role === 'system' && normalizedMessages.length > 0) {
        role = 'user'; 
      }
      
      // 3. Force Strict Alternation: Merge consecutive messages of the same role
      if (normalizedMessages.length > 0 && normalizedMessages[normalizedMessages.length - 1].role === role) {
        normalizedMessages[normalizedMessages.length - 1].content += '\n\n' + msg.content;
      } else {
        normalizedMessages.push({ role, content: msg.content });
      }
    }
    
    // 4. Ensure the chat history doesn't start with an assistant reply
    if (normalizedMessages.length > 0 && normalizedMessages[0].role === 'assistant') {
      normalizedMessages.unshift({ role: 'user', content: 'Hello.' });
    }

    // Safely clamp numbers to prevent schema threshold crashes
    const safe_max_tokens = (parseInt(max_tokens) > 0) ? Math.min(parseInt(max_tokens), 4096) : 4096;
    const safe_temp = (parseFloat(temperature) > 0) ? parseFloat(temperature) : 0.6;
    
    const nimRequest = {
      model: nimModel,
      messages: normalizedMessages,
      temperature: safe_temp,
      top_p: req.body.top_p ?? 1.0,
      max_tokens: safe_max_tokens,
      stream: stream || false
    };
    
    // Append reasoning tags ONLY for models that strictly require custom flags
    if (nimModel.includes('deepseek-v4')) {
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
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          line = line.trim();
          if (!line) return; 
          
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write('data: [DONE]\n\n'); 
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const delta = data.choices[0].delta;
                const reasoning = delta.reasoning_content || delta.reasoning || '';
                const content = delta.content || '';
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '\n</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  data.choices[0].delta.content = combinedContent || content;
                } else {
                  data.choices[0].delta.content = content;
                }
                
                delete data.choices[0].delta.reasoning_content;
                delete data.choices[0].delta.reasoning;
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n\n');
            }
          }
        });
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
    
    // Capture the exact JSON data NVIDIA sent back
    const errorDetail = error.response?.data;
    const exactMessage = errorDetail 
      ? (typeof errorDetail === 'object' ? JSON.stringify(errorDetail) : errorDetail) 
      : error.message;

    console.error(`Proxy crashed with status ${statusCode}:`, exactMessage);

    if (req.body && req.body.stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let chatMessage = `\n\n*[System Error ${statusCode}: `;
      if (statusCode === 429) {
        chatMessage += `NVIDIA API Rate Limit Reached. Wait 60 seconds and try again.]*`;
      } else if (statusCode === 504 || statusCode === 502) {
        chatMessage += `Server Timeout. The AI took too long to think.]*`;
      } else if (statusCode === 401) {
        chatMessage += `Invalid NVIDIA API Key.]*`;
      } else {
        // 🔥 Output the EXACT RAW ERROR to the chat screen
        chatMessage += `NVIDIA rejected the prompt schema.\n\nRAW ERROR DATA:\n${exactMessage}]*`;
      }

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
