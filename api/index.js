const express = require('express');
const { HfInference } = require('@huggingface/inference');
const cors = require('cors');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 3001;

const hf = new HfInference(process.env.HF_ACCESS_TOKEN);

// Mengaktifkan CORS untuk semua permintaan
app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
  res.send('Quantum Core API is running.');
});

app.post('/analyze', async (req, res) => {
  const { text } = req.body;
  if (!text) {
    return res.status(400).json({ error: 'Text is required' });
  }
  try {
    const result = await hf.sentimentAnalysis({
      model: 'distilbert-base-uncased-finetuned-sst-2-english',
      inputs: text,
    });
    res.json(result);
  } catch (error) {
    console.error('Hugging Face API error:', error);
    res.status(500).json({ error: 'Failed to analyze sentiment', details: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
