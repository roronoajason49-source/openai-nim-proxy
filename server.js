// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = true; 

// Model mapping - Streamlined for DeepSeek V4 Pro
const MODEL_MAPPING = {
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro', 
  'claude-3-opus': 'deepseek-ai/deepseek-v4-pro',
  'z-ai/glm-5.1': 'deepseek-ai/deepseek-v4-pro', 
  'glm-5.1': 'deepseek-ai/deepseek-v4-pro'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING
  });
});

// List models endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Exact match or safe fallback to DeepSeek V4 Pro
    let nimModel = MODEL_MAPPING[model] || MODEL_MAPPING[model.toLowerCase()] || 'deepseek-ai/deepseek-v4-pro';
    
    // Construct the payload as a perfect vanilla OpenAI request (no custom keys to trigger 400 errors)
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature ?? 0.6,
      top_p: req.body.top_p ?? 1.0,
      // Clamp tokens to prevent NIM schema errors (NIM hard-caps outputs)
      max_tokens: max_tokens ? Math.min(max_tokens, 8192) : 4096,
      stream: stream || false
    };
    
    // Request execution
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
          if (!line) return; // Skip empty lines to prevent JSON parse crashes
          
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              // CRITICAL FIX: SSE streams MUST end in a double newline \n\n, otherwise JanitorAI hangs endlessly
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
              // Pass malformed lines cleanly
              res.write(line + '\n\n');
            }
          }
        });
      });
      
      response.data.on('end', () => {
        res.end();
      });
      
      response.data.on('error', (err) => {
        console.error('Stream processing interruption:', err);
        res.end();
      });
    } else {
      // Non-streaming fallback
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
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy connectivity error:', error.response?.data || error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.response?.data?.detail || error.response?.data?.error?.message || error.message || 'Internal proxy error',
        type: 'proxy_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
});
