document.addEventListener('DOMContentLoaded', async () => {
  const inputData = document.getElementById('data');
  const selectEstado = document.getElementById('estado');
  const selectProduto = document.getElementById('produto');
  const selectBandeira = document.getElementById('bandeira');
  const form = document.getElementById('form-predicao');
  const btnPrever = document.getElementById('btn-prever');
  const boxResultado = document.getElementById('resultado');
  const textoPreco = document.getElementById('valor-preco');
  const textoDetalhes = document.getElementById('detalhes-entrada');
  const boxErro = document.getElementById('erro');

  // Data de hoje padrão
  inputData.value = new Date().toISOString().split('T')[0];

  function popularSelect(select, itens, padrao) {
    select.innerHTML = '';
    itens.forEach((item) => {
      const opt = document.createElement('option');
      opt.value = item;
      opt.textContent = item;
      if (item === padrao) opt.selected = true;
      select.appendChild(opt);
    });
  }

  try {
    const res = await fetch('/api/opcoes');
    if (!res.ok) throw new Error('Falha ao carregar opções do servidor.');
    const opcoes = await res.json();

    popularSelect(selectEstado, opcoes.estados, 'SP');
    popularSelect(selectProduto, opcoes.produtos, 'GASOLINA');
    popularSelect(selectBandeira, opcoes.bandeiras, 'IPIRANGA');
  } catch (err) {
    boxErro.textContent = err.message;
    boxErro.classList.remove('oculto');
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    boxErro.classList.add('oculto');
    btnPrever.disabled = true;
    btnPrever.textContent = 'Calculando...';

    const payload = {
      data: inputData.value,
      estado: selectEstado.value,
      produto: selectProduto.value,
      bandeira: selectBandeira.value,
    };

    try {
      const res = await fetch('/api/prever', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const dados = await res.json();
      if (!res.ok) throw new Error(dados.erro || 'Erro ao processar previsão.');

      const formatado = new Intl.NumberFormat('pt-BR', {
        style: 'currency',
        currency: 'BRL',
      }).format(dados.preco_estimado);

      textoPreco.textContent = formatado;
      textoDetalhes.textContent =
        `${dados.entrada_processada.produto} • ${dados.entrada_processada.estado} (${dados.entrada_processada.regiao}) • Bandeira: ${dados.entrada_processada.bandeira_aplicada}`;
      boxResultado.classList.remove('oculto');
    } catch (err) {
      boxResultado.classList.add('oculto');
      boxErro.textContent = err.message;
      boxErro.classList.remove('oculto');
    } finally {
      btnPrever.disabled = false;
      btnPrever.textContent = 'Calcular Preço Estimado';
    }
  });
});