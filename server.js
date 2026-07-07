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
    
    // Strip out hidden Janitor fields
    const sanitizedMessages = messages.map(msg => ({
      role: msg.role,
      content: msg.content
    }));
    
    const nimRequest = {
      model: nimModel,
      messages: sanitizedMessages,
      temperature: temperature ?? 0.6,
      top_p: req.body.top_p ?? 1.0,
      max_tokens: max_tokens ? Math.min(max_tokens, 8192) : 4096,
      stream: stream || false
    };
    
    // Reasoning triggers only for models that strictly require it
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
      // Non-streaming fallback omitted for brevity
      res.json({}); 
    }
    
  } catch (error) {
    const statusCode = error.response?.status || 500;
    console.error(`Proxy crashed with status ${statusCode}`);

    // 🔥 THE FIX: Stop JanitorAI from hanging forever by spoofing a clean error message into the chat
    if (req.body && req.body.stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let chatMessage = `\n\n*[System Error ${statusCode}: `;
      if (statusCode === 429) {
        chatMessage += `NVIDIA API Rate Limit Reached. You hit the 40 requests/minute limit, or Render's shared IP is currently blocked by NVIDIA. Please wait 60 seconds and try regenerating.]*`;
      } else if (statusCode === 504 || statusCode === 502) {
        chatMessage += `Server Timeout. The AI took too long to think and the host killed the connection.]*`;
      } else if (statusCode === 401) {
        chatMessage += `Invalid NVIDIA API Key. Check your environment variables.]*`;
      } else {
        chatMessage += `NVIDIA rejected the prompt schema. Check Render logs.]*`;
      }

      // Package the error perfectly so Janitor reads it as standard character dialogue
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
        error: { message: `Proxy Error: ${statusCode}`, type: 'proxy_error', code: statusCode }
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
