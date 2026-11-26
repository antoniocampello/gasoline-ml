const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');
const { S3Client, GetObjectCommand } = require('@aws-sdk/client-s3');

const BUCKET_NAME = process.env.BUCKET_NAME || 'gasoline-regression-model-datalake';
const AWS_REGION = process.env.AWS_REGION || 'us-east-1';
const CACHE_DIR = path.join(__dirname, '..', 'modelo_cache');
const ONNX_PATH = path.join(CACHE_DIR, 'modelo_random_forest.onnx');
const JSON_PATH = path.join(CACHE_DIR, 'metadados_modelo.json');

const MAPA_UF_REGIAO = {
  AC: 'N', AP: 'N', AM: 'N', PA: 'N', RO: 'N', RR: 'N', TO: 'N',
  AL: 'NE', BA: 'NE', CE: 'NE', MA: 'NE', PB: 'NE', PE: 'NE', PI: 'NE', RN: 'NE', SE: 'NE',
  DF: 'CO', GO: 'CO', MT: 'CO', MS: 'CO',
  ES: 'SE', MG: 'SE', RJ: 'SE', SP: 'SE',
  PR: 'S', RS: 'S', SC: 'S',
};

let state = {
  session: null,
  colunas: [],
  indiceColunas: {},
  topBandeiras: [],
  produtosValidos: [],
};

async function baixarArquivoS3(s3, key, destinoLocal) {
  const command = new GetObjectCommand({ Bucket: BUCKET_NAME, Key: key });
  const response = await s3.send(command);
  const bytes = await response.Body.transformToByteArray();
  fs.writeFileSync(destinoLocal, Buffer.from(bytes));
}

async function inicializarModelo() {
  if (!fs.existsSync(CACHE_DIR)) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
  }

  const s3 = new S3Client({ region: AWS_REGION });

  if (!fs.existsSync(ONNX_PATH)) {
    console.log(`Baixando modelo ONNX de s3://${BUCKET_NAME}/modelo/modelo_random_forest.onnx...`);
    await baixarArquivoS3(s3, 'modelo/modelo_random_forest.onnx', ONNX_PATH);
  }

  if (!fs.existsSync(JSON_PATH)) {
    console.log(`Baixando metadados de s3://${BUCKET_NAME}/modelo/metadados_modelo.json...`);
    await baixarArquivoS3(s3, 'modelo/metadados_modelo.json', JSON_PATH);
  }

  const metadados = JSON.parse(fs.readFileSync(JSON_PATH, 'utf-8'));
  const colunas = metadados.colunas;
  const topBandeiras = metadados.top_bandeiras;

  const indiceColunas = {};
  colunas.forEach((col, idx) => {
    indiceColunas[col] = idx;
  });

  // Reconstrói produtos incluindo 'DIESEL'
  const produtos = colunas
    .filter((c) => c.startsWith('Produto_'))
    .map((c) => c.replace('Produto_', ''));
  if (!produtos.includes('DIESEL')) {
    produtos.push('DIESEL');
  }

  const session = await ort.InferenceSession.create(ONNX_PATH);

  state = {
    session,
    colunas,
    indiceColunas,
    topBandeiras,
    produtosValidos: produtos.sort(),
  };

  console.log(`Modelo ONNX carregado no Node.js (${colunas.length} features).`);
}

function parseDataSegura(dataStr) {
  const limpo = String(dataStr || '').trim();
  let ano, mes, dia;

  if (/^\d{4}-\d{2}-\d{2}$/.test(limpo)) {
    [ano, mes, dia] = limpo.split('-').map(Number);
  } else if (/^\d{2}\/\d{2}\/\d{4}$/.test(limpo)) {
    [dia, mes, ano] = limpo.split('/').map(Number);
  } else {
    throw new Error('Formato de data inválido. Use YYYY-MM-DD ou DD/MM/YYYY.');
  }

  const dataObj = new Date(ano, mes - 1, dia);
  if (
    dataObj.getFullYear() !== ano ||
    dataObj.getMonth() !== mes - 1 ||
    dataObj.getDate() !== dia
  ) {
    throw new Error('Data inexistente no calendário.');
  }

  // Converte JS getDay() (0=Dom..6=Sab) para Python dayofweek (0=Seg..6=Dom)
  const diaSemanaPython = (dataObj.getDay() + 6) % 7;

  return { ano, mes, dia, diaSemana: diaSemanaPython };
}

function obterOpcoes() {
  const bandeiras = Array.from(new Set([...state.topBandeiras, 'OUTRAS'])).sort();
  return {
    estados: Object.keys(MAPA_UF_REGIAO).sort(),
    produtos: state.produtosValidos,
    bandeiras,
  };
}

async function preverPreco({ data, estado, produto, bandeira }) {
  if (!state.session) {
    throw new Error('O modelo ainda está sendo carregado.');
  }

  const { ano, mes, dia, diaSemana } = parseDataSegura(data);
  const uf = String(estado || '').trim().toUpperCase();
  const prod = String(produto || '').trim().toUpperCase();
  const bandRaw = String(bandeira || '').trim().toUpperCase();

  if (!MAPA_UF_REGIAO[uf]) {
    throw new Error(`Estado inválido: '${uf}'.`);
  }

  const regiao = MAPA_UF_REGIAO[uf];
  const bandAplicada = state.topBandeiras.includes(bandRaw) ? bandRaw : 'OUTRAS';

  const vetor = new Float32Array(state.colunas.length);
  const idxMap = state.indiceColunas;

  vetor[idxMap['Ano']] = ano;
  vetor[idxMap['Mes']] = mes;
  vetor[idxMap['Dia']] = dia;
  vetor[idxMap['Dia_Semana']] = diaSemana;

  const chavesOneHot = [
    `Regiao - Sigla_${regiao}`,
    `Estado - Sigla_${uf}`,
    `Produto_${prod}`,
    `Bandeira_${bandAplicada}`,
  ];

  for (const chave of chavesOneHot) {
    if (chave in idxMap) {
      vetor[idxMap[chave]] = 1.0;
    }
  }

  const inputName = state.session.inputNames[0];
  const outputName = state.session.outputNames[0];
  const tensorEntrada = new ort.Tensor('float32', vetor, [1, state.colunas.length]);

  const resultados = await state.session.run({ [inputName]: tensorEntrada });
  const precoEstimado = Number(resultados[outputName].data[0]);

  return {
    preco_estimado: Number(precoEstimado.toFixed(2)),
    preco_exato: Number(precoEstimado.toFixed(4)),
    moeda: 'BRL',
    entrada_processada: {
      data,
      estado: uf,
      regiao,
      produto: prod,
      bandeira_aplicada: bandAplicada,
    },
  };
}

module.exports = {
  inicializarModelo,
  obterOpcoes,
  preverPreco,
};