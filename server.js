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

// 🔥 REASONING DISPLAY TOGGLE - Formats internal reasoning tracks into <think> blocks
const SHOW_REASONING = true; 

// Model mapping dictionary (GLM removed)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta/llama-3.3-70b-instruct',
  'gpt-4': 'nvidia/llama-3.3-nemotron-super-49b-v1',
  'gpt-4-turbo': 'nvidia/llama-3.3-nemotron-super-49b-v1',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro', 
  'claude-3-opus': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-sonnet': 'nvidia/llama-3.3-nemotron-super-49b-v1',
  'gemini-pro': 'nvidia/llama-3.3-nemotron-super-49b-v1',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro'
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
    
    // ⚡ INSTANT LOCAL MODEL SELECTION
    let nimModel = MODEL_MAPPING[model];
    
    if (!nimModel) {
      if (model && model.includes('/')) {
        nimModel = model;
      } else {
        const modelLower = (model || '').toLowerCase();
        if (modelLower.includes('deepseek') || modelLower.includes('v4') || modelLower.includes('opus')) {
          nimModel = 'deepseek-ai/deepseek-v4-pro';
        } else if (modelLower.includes('nemotron') || modelLower.includes('49b')) {
          nimModel = 'nvidia/llama-3.3-nemotron-super-49b-v1';
        } else {
          nimModel = 'meta/llama-3.3-70b-instruct';
        }
      }
    }
    
    // Request payload configuration
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature !== undefined ? temperature : 0.6,
      max_tokens: max_tokens || 8192, // Bumped up for long reasoning chains
      stream: stream || false
    };

    // 🧠 TRIGGER NIM REASONING: NVIDIA requires this top-level flag to activate DeepSeek
    if (nimModel.includes('deepseek') || nimModel.includes('nemotron-3') || nimModel.includes('thinking')) {
      nimRequest.reasoning_effort = "high";
    }
    
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 60000 
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
          const trimmedLine = line.trim();
          if (!trimmedLine) return; 
          
          // FIX 1: Proper double newline requirement for SSE termination (Stops Janitor from hanging)
          if (trimmedLine === 'data: [DONE]') {
            res.write('data: [DONE]\n\n');
            return;
          }
          
          if (trimmedLine.startsWith('data:')) {
            try {
              // FIX 2: Safe JSON stripping
              const jsonStr = trimmedLine.replace(/^data:\s*/, '');
              const data = JSON.parse(jsonStr);
              
              if (data.choices?.[0]?.delta) {
                const delta = data.choices[0].delta;
                
                // FIX 3: Catch NVIDIA NIM's distinct `.reasoning` key for DeepSeek
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
                  
                  // FIX 4: Auto-close tag if reasoning finishes or ends natively
                  if (data.choices[0].finish_reason && reasoningStarted) {
                     combinedContent += '\n</think>\n\n';
                     reasoningStarted = false;
                  }
                  
                  data.choices[0].delta.content = combinedContent;
                  delete data.choices[0].delta.reasoning_content;
                  delete data.choices[0].delta.reasoning;
                } else {
                  data.choices[0].delta.content = content;
                  delete data.choices[0].delta.reasoning_content;
                  delete data.choices[0].delta.reasoning;
                }
              }
              // Double newline forces Janitor to read the chunk immediately
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(`${trimmedLine}\n\n`);
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream processing error:', err);
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
          const reasoning = choice.message?.reasoning_content || choice.message?.reasoning || '';
          
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
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error details:', error.response?.data || error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
});
