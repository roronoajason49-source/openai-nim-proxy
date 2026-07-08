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
    const { model, messages, temperature, max_tokens, stream } = req.body;
    let nimModel = MODEL_MAPPING[model] || MODEL_MAPPING[model?.toLowerCase()] || 'stepfun-ai/step-3.7-flash';
    
    // Normalize and clean chat history roles
    const normalizedMessages = [];
    let isFirstSystem = true;

    for (const msg of messages) {
      if (!msg.content || typeof msg.content !== 'string' || msg.content.trim() === '') continue;
      
      let role = msg.role.toLowerCase();
      
      if (role === 'system') {
        if (isFirstSystem && normalizedMessages.length === 0) {
          normalizedMessages.push({ role: 'system', content: msg.content });
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
    
    // Exact model hardware triggers
    if (nimModel.includes('step-3.7')) {
      nimRequest.reasoning_effort = "high";
    }
    
    if (nimModel.includes('glm-5.2')) {
      nimRequest.chat_template_kwargs = { 
        enable_thinking: true, 
        reasoning_effort: "high" 
      };
    }
    
    if (nimModel.includes('minimax')) {
      nimRequest.reasoning_effort = "high";
      nimRequest.thinking = { type: "enabled" }; 
    }
    
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
                  
                  if (reasoning) {
                    if (!reasoningStarted) {
                      combinedContent = '<think>\n' + reasoning;
                      reasoningStarted = true;
                    } else {
                      combinedContent = reasoning;
                    }
                  }
                  
                  // 🔥 THE FIX: Safely transition only when actual dialogue text appears
                  if (content !== '') {
                    if (reasoningStarted) {
                      combinedContent += '\n</think>\n\n' + content;
                      reasoningStarted = false;
                    } else {
                      combinedContent += content;
                    }
                  } else if (reasoningStarted && ('content' in delta) && !reasoning) {
                    // Catch the exact transition token safely
                    combinedContent += '\n</think>\n\n';
                    reasoningStarted = false;
                  }
                  
                  data.choices[0].delta.content = combinedContent;
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
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          const reasoning = choice.message?.reasoning_content || choice.message?.reasoning;
          
          if (SHOW_REASONING && reasoning) {
            fullContent = '<think>\n' + reasoning + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: { role: choice.message.role, content: fullContent },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse); 
    }
    
  } catch (error) {
    const statusCode = error.response?.status || 500;
    let exactMessage = error.message;

    if (error.response?.data) {
      if (typeof error.response.data.on === 'function') {
        try {
          const chunks = [];
          for await (const chunk of error.response.data) {
            chunks.push(chunk);
          }
          exactMessage = Buffer.concat(chunks).toString();
        } catch (streamErr) {
          exactMessage = `Failed to parse NVIDIA error stream: ${error.message}`;
        }
      } else if (typeof error.response.data === 'object') {
        exactMessage = JSON.stringify(error.response.data);
      } else {
        exactMessage = error.response.data;
      }
    }

    console.error(`Proxy crashed with status ${statusCode}:`, exactMessage);

    if (req.body && req.body.stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
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
