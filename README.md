# 🎥 Sistema VAR Portátil

<p align="center">
  <strong>Revisão de lances em vídeo com captura de câmeras e replay instantâneo.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-Visão%20Computacional-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/NumPy-Processamento%20de%20Dados-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Status-Em%20desenvolvimento-orange?style=for-the-badge" alt="Status do projeto">
</p>

---

## 📌 Sobre o projeto

O **Sistema VAR Portátil** é um projeto desenvolvido em Python com OpenCV para capturar imagens de câmeras e permitir a revisão de lances por meio de replays.

A proposta é oferecer uma solução prática para situações em que seja necessário rever uma jogada, com acesso rápido aos últimos segundos de vídeo e ferramentas para controlar a reprodução.

O projeto está em desenvolvimento e passa por testes com câmeras reais para validar a estabilidade da captura, a sincronização das imagens e o funcionamento dos replays.

## ✨ Funcionalidades

- 📹 **Captura de vídeo:** suporte previsto para até quatro câmeras.
- 🔄 **Buffer de vídeo:** mantém os últimos 30 segundos disponíveis para revisão.
- 🎬 **Replay:** permite salvar os lances capturados em arquivos de vídeo.
- 🎮 **Controle de reprodução:** pausa, reinício e ajuste da velocidade do replay.
- ⏩ **Navegação por quadros:** permite avançar e voltar durante a análise.
- 📊 **Barra de progresso:** acompanha a posição da reprodução e facilita a navegação.
- 🖥️ **Visualização de câmeras:** organiza as imagens em uma interface de monitoramento.
- 💾 **Salvamento local:** armazena os replays na pasta `replays`.

## 🛠️ Tecnologias utilizadas

| Tecnologia | Utilização |
|---|---|
| Python | Linguagem principal |
| OpenCV | Captura, processamento e exibição de vídeo |
| NumPy | Manipulação de dados das imagens |
| Threading | Execução concorrente das tarefas de captura |
| pathlib | Organização dos caminhos e arquivos |
| datetime | Registro de data e horário dos replays |

## ⚙️ Requisitos

- Python 3 instalado.
- Câmera ou webcam compatível com o OpenCV.
- Windows, considerando a configuração atual de captura com DirectShow.
- Bibliotecas Python necessárias instaladas.

## 🚀 Como executar

### 1. Clone o repositório

```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
```

Entre na pasta do projeto:

```bash
cd SEU-REPOSITORIO
```

### 2. Instale as dependências

```bash
pip install opencv-python numpy
```

### 3. Execute o sistema

```bash
python main.py
```

> **Observação:** substitua `main.py` pelo nome real do arquivo principal, caso seja diferente. Os comandos de clonagem também devem ser atualizados com o endereço real do seu repositório.

## 🎮 Controles do sistema

| Tecla | Ação |
|---|---|
| `ESPAÇO` | Salvar o replay durante a visualização ao vivo |
| `A` | Voltar um quadro |
| `D` | Avançar um quadro |
| `←` | Voltar um quadro |
| `→` | Avançar um quadro |
| `1` | Reproduzir a 0,25× |
| `2` | Reproduzir a 0,5× |
| `3` | Reproduzir na velocidade normal |
| `4` | Reproduzir a 2× |
| `R` | Reiniciar a reprodução |
| `Q` ou `ESC` | Sair da janela ou encerrar a aplicação, conforme o modo atual |

*Os controles podem depender da janela que estiver ativa. Consulte também as indicações exibidas pela própria interface.*

## 📁 Estrutura do projeto

```text
sistema-var-portatil/
├── main.py
├── replays/
└── README.md
```

A estrutura acima é ilustrativa. A pasta `replays` é utilizada para armazenar os vídeos gerados e pode ser criada pelo próprio sistema durante a execução.

## 🧪 Testes em andamento

A validação do projeto considera:

- Funcionamento com diferentes quantidades de câmeras.
- Estabilidade e proporção correta das imagens.
- Retenção dos últimos segundos no buffer.
- Sincronização entre as câmeras.
- Salvamento e reprodução dos arquivos gerados.
- Funcionamento dos controles durante a análise.
- Continuidade da captura ao vivo durante o replay.
- Liberação dos dispositivos ao encerrar o programa.

## 🔮 Próximas melhorias

- Aprimorar a sincronização entre câmeras.
- Permitir configurar a duração do replay.
- Melhorar as mensagens de erro quando uma câmera falhar.
- Facilitar a organização dos vídeos salvos.
- Expandir os controles e as opções de configuração.

## 👨‍💻 Autor

**Arthur Luiz da Silva Oliveira**

Projeto pessoal desenvolvido para explorar captura de vídeo, visão computacional e ferramentas de revisão de lances esportivos.

---

<p align="center">
  Desenvolvido com Python e OpenCV.
</p>
