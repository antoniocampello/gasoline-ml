const path = require('path');
const express = require('express');
const { inicializarModelo, obterOpcoes, preverPreco } = require('./services/mlService');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/opcoes', (req, res) => {
  res.json(obterOpcoes());
});

app.post('/api/prever', async (req, res) => {
  try {
    const resultado = await preverPreco(req.body);
    res.json(resultado);
  } catch (err) {
    res.status(400).json({ erro: err.message });
  }
});

app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok' });
});

inicializarModelo()
  .then(() => {
    app.listen(PORT, () => {
      console.log(`Servidor rodando em http://localhost:${PORT}`);
    });
  })
  .catch((err) => {
    console.error('Falha ao inicializar modelo ONNX:', err);
    process.exit(1);
  });