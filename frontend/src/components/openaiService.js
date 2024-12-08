import axios from 'axios';

const API_KEY = process.env.REACT_APP_OPENAI_API_KEY;

const openai = axios.create({
  baseURL: 'https://api.openai.com/v1',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_KEY}`,
  },
});

export const getOpenAIResponse = async (prompt) => {
  const response = await openai.post('/chat/completions', {
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: prompt }],
    max_tokens: 2048,
  });
  console.log(response.data.choices[0].message.content);
  return response.data.choices[0].message.content;
};
