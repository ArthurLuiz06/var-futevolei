
import cv2
import threading
import time
import math
import numpy as np

from collections import deque
from datetime import datetime
from pathlib import Path


# =========================================================
# CONFIGURAÇÕES
# =========================================================

DURACAO_BUFFER = 30
QUALIDADE_JPEG = 80
FPS_REPLAY = 30

PASTA_REPLAYS = Path(__file__).resolve().parent / "replays"


# =========================================================
# CAPTURA DA WEBCAM E BUFFER CONTÍNUO
# =========================================================

class CapturaWebcam:
    def __init__(self, indice_camera=0):
        self.indice_camera = indice_camera
        self.camera = None

        self.frame_atual = None
        self.lock_frame = threading.Lock()

        # Cada item: (instante, imagem JPEG).
        self.buffer = deque()
        self.lock_buffer = threading.Lock()

        self.ativa = False
        self.thread = None
        self.erro = None
        self.total_bytes_buffer = 0

    def iniciar(self):
        self.camera = cv2.VideoCapture(self.indice_camera)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            self.erro = "Não foi possível abrir a webcam."
            return False

        self.ativa = True

        self.thread = threading.Thread(
            target=self._capturar,
            daemon=True,
        )

        self.thread.start()
        return True

    def _capturar(self):
        try:
            while self.ativa:
                sucesso, frame = self.camera.read()

                if not sucesso:
                    if self.ativa:
                        self.erro = "Falha ao capturar imagem da webcam."
                        time.sleep(0.02)

                    continue

                instante = time.monotonic()

                sucesso_jpeg, imagem_jpeg = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, QUALIDADE_JPEG],
                )

                if not sucesso_jpeg:
                    continue

                dados_jpeg = imagem_jpeg.tobytes()

                with self.lock_frame:
                    self.frame_atual = dados_jpeg

                with self.lock_buffer:
                    self.buffer.append((instante, dados_jpeg))
                    self.total_bytes_buffer += len(dados_jpeg)

                    limite = instante - DURACAO_BUFFER

                    while (
                        self.buffer
                        and self.buffer[0][0] < limite
                    ):
                        _, dados_antigos = self.buffer.popleft()
                        self.total_bytes_buffer -= len(dados_antigos)

        except Exception as erro:
            self.erro = f"Erro durante a captura: {erro}"

        finally:
            self.ativa = False

    def obter_frame(self):
        with self.lock_frame:
            dados = self.frame_atual

        if dados is None:
            return None

        return cv2.imdecode(
            np.frombuffer(dados, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

    def obter_replay(self):
        """Retorna uma cópia do buffer atual."""
        with self.lock_buffer:
            return list(self.buffer)

    def obter_status_buffer(self):
        with self.lock_buffer:
            quantidade = len(self.buffer)
            memoria_mb = self.total_bytes_buffer / (1024 * 1024)

            if quantidade < 2:
                return 0.0, quantidade, memoria_mb

            duracao = (
                self.buffer[-1][0] - self.buffer[0][0]
            )

            return duracao, quantidade, memoria_mb

    def parar(self):
        self.ativa = False

        if self.thread is not None:
            self.thread.join(timeout=3)

        if self.camera is not None:
            self.camera.release()
            self.camera = None


# =========================================================
# DECODIFICAÇÃO DOS QUADROS
# =========================================================

def decodificar_frame(frame_jpeg):
    """Converte bytes JPEG em uma imagem OpenCV."""
    return cv2.imdecode(
        np.frombuffer(frame_jpeg, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )


# =========================================================
# SALVAMENTO DO REPLAY EM MP4
# =========================================================

def salvar_replay_mp4(clipe):
    """Salva o buffer em MP4 e retorna o caminho do arquivo criado."""
    if not clipe:
        print("Não há imagens disponíveis para salvar.")
        return None

    PASTA_REPLAYS.mkdir(parents=True, exist_ok=True)

    primeiro_frame = None

    for _, frame_jpeg in clipe:
        primeiro_frame = decodificar_frame(frame_jpeg)

        if primeiro_frame is not None:
            break

    if primeiro_frame is None:
        print("Não foi possível decodificar os quadros do replay.")
        return None

    altura, largura = primeiro_frame.shape[:2]

    nome_arquivo = (
        "replay_"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        + ".mp4"
    )

    caminho_arquivo = PASTA_REPLAYS / nome_arquivo

    quatrocc = cv2.VideoWriter_fourcc(*"mp4v")

    gravador = cv2.VideoWriter(
        str(caminho_arquivo),
        quatrocc,
        FPS_REPLAY,
        (largura, altura),
    )

    if not gravador.isOpened():
        gravador.release()
        print("Não foi possível criar o arquivo MP4.")
        print("Verifique o suporte a vídeo do OpenCV.")
        return None

    quadros_salvos = 0
    erro_gravacao = None

    try:
        for _, frame_jpeg in clipe:
            frame = decodificar_frame(frame_jpeg)

            if frame is None:
                continue

            if (
                frame.shape[1] != largura
                or frame.shape[0] != altura
            ):
                frame = cv2.resize(frame, (largura, altura))

            # Grava somente o vídeo da câmera, sem a interface.
            gravador.write(frame)
            quadros_salvos += 1

    except Exception as erro:
        erro_gravacao = erro

    finally:
        gravador.release()

    if erro_gravacao is not None:
        print(f"Erro ao gravar o replay: {erro_gravacao}")
        return None

    if quadros_salvos == 0:
        print("Nenhum quadro foi salvo.")
        return None

    if (
        not caminho_arquivo.exists()
        or caminho_arquivo.stat().st_size == 0
    ):
        print("O arquivo MP4 não foi criado corretamente.")
        return None

    print("\n" + "=" * 48)
    print("REPLAY SALVO COM SUCESSO!")
    print("=" * 48)
    print(f"Arquivo: {caminho_arquivo}")
    print(f"Quadros salvos: {quadros_salvos}")
    print(
        f"Tamanho: "
        f"{caminho_arquivo.stat().st_size / (1024 * 1024):.2f} MB"
    )
    print("O replay será aberto no visualizador do VAR.")
    print("=" * 48)

    return caminho_arquivo


# =========================================================
# GERENCIADOR DE SALVAMENTO EM SEGUNDO PLANO
# =========================================================

class SalvadorReplay:
    def __init__(self):
        self.thread = None
        self.lock = threading.Lock()
        self.em_andamento = False
        self.replay_pronto = None

    def iniciar(self, clipe):
        """Inicia a gravação em uma thread separada."""
        if not clipe:
            print("O buffer ainda não possui quadros para salvar.")
            return False

        with self.lock:
            if self.em_andamento:
                print(
                    "Já existe um replay sendo salvo. "
                    "Aguarde a conclusão."
                )
                return False

            self.em_andamento = True

            try:
                self.thread = threading.Thread(
                    target=self._executar_salvamento,
                    args=(clipe,),
                    daemon=False,
                )

                self.thread.start()

            except Exception:
                self.em_andamento = False
                raise

        print(
            "Solicitação recebida! "
            "O vídeo ao vivo continuará funcionando."
        )

        return True

    def _executar_salvamento(self, clipe):
        caminho = None

        try:
            caminho = salvar_replay_mp4(clipe)

        except Exception as erro:
            print(f"Erro inesperado ao salvar replay: {erro}")

        finally:
            with self.lock:
                self.replay_pronto = caminho
                self.em_andamento = False

    def esta_salvando(self):
        with self.lock:
            return self.em_andamento

    def obter_replay_pronto(self):
        """Entrega o replay recém-salvo uma única vez."""
        with self.lock:
            caminho = self.replay_pronto
            self.replay_pronto = None
            return caminho

    def aguardar(self):
        """Aguarda a conclusão do salvamento antes de encerrar."""
        with self.lock:
            thread = self.thread

        if thread is not None and thread.is_alive():
            print("Aguardando a conclusão do salvamento...")
            thread.join()


# =========================================================
# FUNÇÕES AUXILIARES DA INTERFACE DO REPLAY
# =========================================================

def desenhar_texto_legivel(
    imagem,
    texto,
    posicao,
    escala=0.42,
    cor=(255, 255, 255),
):
    """Desenha texto com contorno para facilitar a leitura."""
    cv2.putText(
        imagem,
        texto,
        posicao,
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        imagem,
        texto,
        posicao,
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        cor,
        1,
        cv2.LINE_AA,
    )


def formatar_tempo(segundos):
    segundos = max(0, int(segundos))

    minutos = segundos // 60
    segundos_restantes = segundos % 60

    return f"{minutos:02d}:{segundos_restantes:02d}"


# =========================================================
# VISUALIZADOR DE REPLAY
# =========================================================

def visualizar_replay(caminho_arquivo):
    """
    Visualizador com:
    - Barra de progresso clicável.
    - Tempo atual e restante.
    - Câmera lenta e câmera acelerada.
    - Navegação quadro a quadro.
    - Reinício do replay.
    - Temporização baseada no relógio monotônico.
    - Compensação de atrasos durante a reprodução.
    - Interface sem instruções sobrepostas ao vídeo.
    """

    captura_replay = cv2.VideoCapture(str(caminho_arquivo))

    if not captura_replay.isOpened():
        print("Não foi possível abrir o replay no visualizador.")
        captura_replay.release()
        return

    total_quadros = int(
        captura_replay.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps_arquivo = captura_replay.get(cv2.CAP_PROP_FPS)

    if not fps_arquivo or fps_arquivo <= 0:
        fps_arquivo = FPS_REPLAY

    duracao_total = (
        total_quadros / fps_arquivo
        if total_quadros > 0
        else 0.0
    )

    indice_quadro = 0
    velocidade = 1.0
    pausado = False
    quadro_atual = None

    # Relógio usado para manter o ritmo da reprodução.
    proximo_instante_frame = time.perf_counter()

    nome_janela = "VAR Futevolei - Revisao do Replay"

    estado_mouse = {
        "solicitar_quadro": None
    }

    def ao_clicar_mouse(evento, x, y, flags, parametro):
        if evento != cv2.EVENT_LBUTTONDOWN:
            return

        largura_janela = parametro["largura"]
        altura_janela = parametro["altura"]

        y_barra = altura_janela - 34
        x_inicio = 20
        x_fim = max(x_inicio + 1, largura_janela - 20)

        if (
            y_barra - 12 <= y <= y_barra + 12
            and x_inicio <= x <= x_fim
        ):
            proporcao = (
                (x - x_inicio) / (x_fim - x_inicio)
            )

            alvo = int(
                proporcao * max(total_quadros - 1, 0)
            )

            estado_mouse["solicitar_quadro"] = alvo

    cv2.namedWindow(
        nome_janela,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        nome_janela,
        1100,
        650,
    )

    dimensoes_janela = {
        "largura": 1100,
        "altura": 650,
    }

    cv2.setMouseCallback(
        nome_janela,
        ao_clicar_mouse,
        dimensoes_janela,
    )

    print("\n" + "=" * 48)
    print("VISUALIZADOR DE REPLAY")
    print("ESPAÇO: pausar/continuar")
    print("1: 0,25x | 2: 0,5x | 3: 1x | 4: 2x")
    print("A / seta esquerda: quadro anterior")
    print("D / seta direita: próximo quadro")
    print("R: reiniciar | clique na barra: buscar momento")
    print("Q ou ESC: fechar replay e voltar ao ao vivo")
    print("=" * 48 + "\n")

    try:
        while True:
            inicio_ciclo = time.perf_counter()

            try:
                retangulo = cv2.getWindowImageRect(nome_janela)

                if (
                    retangulo
                    and retangulo[2] > 0
                    and retangulo[3] > 0
                ):
                    dimensoes_janela["largura"] = retangulo[2]
                    dimensoes_janela["altura"] = retangulo[3]

            except (cv2.error, AttributeError):
                pass

            # -------------------------------------------------
            # BUSCA PELO CLIQUE NA BARRA
            # -------------------------------------------------

            quadro_solicitado = estado_mouse["solicitar_quadro"]

            if quadro_solicitado is not None:
                indice_quadro = max(
                    0,
                    min(
                        quadro_solicitado,
                        max(total_quadros - 1, 0),
                    ),
                )

                captura_replay.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    indice_quadro,
                )

                sucesso, frame = captura_replay.read()

                estado_mouse["solicitar_quadro"] = None
                pausado = True

                # Reinicia a referência temporal após a busca.
                proximo_instante_frame = time.perf_counter()

                if sucesso:
                    quadro_atual = frame
                else:
                    continue

            # -------------------------------------------------
            # LEITURA DO PRÓXIMO QUADRO
            # -------------------------------------------------

            if not pausado or quadro_atual is None:
                sucesso, frame = captura_replay.read()

                if not sucesso:
                    if total_quadros <= 0:
                        break

                    pausado = True
                    indice_quadro = max(0, total_quadros - 1)

                    captura_replay.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        indice_quadro,
                    )

                    sucesso, frame = captura_replay.read()

                    if not sucesso:
                        break

                else:
                    indice_quadro = max(
                        0,
                        int(
                            captura_replay.get(
                                cv2.CAP_PROP_POS_FRAMES
                            )
                        ) - 1,
                    )

                quadro_atual = frame

            tela = quadro_atual.copy()

            altura, largura = tela.shape[:2]

            tempo_atual = (
                indice_quadro / fps_arquivo
                if fps_arquivo > 0
                else 0.0
            )

            tempo_restante = max(
                0.0,
                duracao_total - tempo_atual,
            )

            # -------------------------------------------------
            # STATUS SUPERIOR
            # -------------------------------------------------

            cv2.rectangle(
                tela,
                (0, 0),
                (largura, 48),
                (25, 25, 25),
                -1,
            )

            estado_texto = (
                "PAUSADO"
                if pausado
                else "REPRODUZINDO"
            )

            texto_status = (
                f"{estado_texto} | {velocidade:g}x | "
                f"Tempo: {formatar_tempo(tempo_atual)} / "
                f"{formatar_tempo(duracao_total)} | "
                f"Quadro: {indice_quadro + 1}/"
                f"{max(total_quadros, 1)}"
            )

            cv2.putText(
                tela,
                texto_status,
                (12, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # -------------------------------------------------
            # BARRA DE PROGRESSO
            # -------------------------------------------------

            y_barra = max(55, altura - 34)

            x_inicio = 20
            x_fim = max(x_inicio + 1, largura - 20)

            progresso = (
                indice_quadro / max(total_quadros - 1, 1)
                if total_quadros > 1
                else 0.0
            )

            x_progresso = int(
                x_inicio + progresso * (x_fim - x_inicio)
            )

            cv2.line(
                tela,
                (x_inicio, y_barra),
                (x_fim, y_barra),
                (190, 190, 190),
                2,
                cv2.LINE_AA,
            )

            cv2.line(
                tela,
                (x_inicio, y_barra),
                (x_progresso, y_barra),
                (0, 200, 255),
                3,
                cv2.LINE_AA,
            )

            cv2.circle(
                tela,
                (x_progresso, y_barra),
                5,
                (0, 0, 0),
                -1,
                cv2.LINE_AA,
            )

            cv2.circle(
                tela,
                (x_progresso, y_barra),
                3,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )

            # -------------------------------------------------
            # TEMPO ATUAL E TEMPO RESTANTE
            # -------------------------------------------------

            texto_atual = formatar_tempo(tempo_atual)
            texto_restante = f"-{formatar_tempo(tempo_restante)}"

            escala_tempo = 0.42
            y_tempo = altura - 8

            desenhar_texto_legivel(
                tela,
                texto_atual,
                (x_inicio, y_tempo),
                escala=escala_tempo,
            )

            tamanho_restante = cv2.getTextSize(
                texto_restante,
                cv2.FONT_HERSHEY_SIMPLEX,
                escala_tempo,
                1,
            )[0]

            x_tempo_restante = max(
                x_inicio,
                largura - tamanho_restante[0] - 20,
            )

            desenhar_texto_legivel(
                tela,
                texto_restante,
                (x_tempo_restante, y_tempo),
                escala=escala_tempo,
            )

            cv2.imshow(nome_janela, tela)

            # -------------------------------------------------
            # TEMPORIZAÇÃO DA REPRODUÇÃO
            # -------------------------------------------------
            #
            # O intervalo é calculado usando o FPS real do
            # arquivo e a velocidade selecionada.
            #
            # Exemplo: 30 FPS em 2x = 60 quadros por segundo
            # de reprodução, ou aproximadamente 16,67 ms
            # entre quadros exibidos.
            #
            # O relógio monotônico evita que o tempo gasto
            # renderizando a interface seja somado ao atraso
            # de cada quadro.
            # -------------------------------------------------

            if pausado:
                tecla = cv2.waitKeyEx(30)

                # Não acumula atrasos enquanto estiver pausado.
                proximo_instante_frame = time.perf_counter()

            else:
                intervalo_frame = 1.0 / (
                    fps_arquivo * velocidade
                )

                proximo_instante_frame += intervalo_frame

                agora = time.perf_counter()
                tempo_espera = proximo_instante_frame - agora

                if tempo_espera > 0:
                    atraso_ms = max(
                        1,
                        int(math.ceil(tempo_espera * 1000)),
                    )

                    tecla = cv2.waitKeyEx(atraso_ms)

                else:
                    # Continua verificando os comandos sem
                    # acrescentar um atraso artificial grande.
                    tecla = cv2.waitKeyEx(1)

                    # Se o processamento atrasou a reprodução,
                    # pula quadros para recuperar o ritmo real.
                    agora = time.perf_counter()
                    atraso = agora - proximo_instante_frame

                    if atraso >= intervalo_frame:
                        quadros_pular = int(
                            atraso / intervalo_frame
                        )

                        quadros_disponiveis = max(
                            0,
                            total_quadros - indice_quadro - 1,
                        )

                        quadros_pular = min(
                            quadros_pular,
                            quadros_disponiveis,
                        )

                        for _ in range(quadros_pular):
                            if not captura_replay.grab():
                                break

                            indice_quadro += 1

                            proximo_instante_frame += (
                                intervalo_frame
                            )

            # -------------------------------------------------
            # CONTROLES DO REPLAY
            # -------------------------------------------------

            if tecla in (27, ord("q"), ord("Q")):
                break

            elif tecla == 32:  # Espaço
                pausado = not pausado
                proximo_instante_frame = time.perf_counter()

            elif tecla == ord("1"):
                velocidade = 0.25
                proximo_instante_frame = time.perf_counter()

            elif tecla == ord("2"):
                velocidade = 0.5
                proximo_instante_frame = time.perf_counter()

            elif tecla == ord("3"):
                velocidade = 1.0
                proximo_instante_frame = time.perf_counter()

            elif tecla == ord("4"):
                velocidade = 2.0
                proximo_instante_frame = time.perf_counter()

            elif tecla in (ord("r"), ord("R")):
                # Reinicia desde o começo e volta a reproduzir.
                indice_quadro = 0

                captura_replay.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0,
                )

                sucesso, frame = captura_replay.read()

                if sucesso:
                    quadro_atual = frame

                pausado = False
                proximo_instante_frame = time.perf_counter()

            elif tecla in (
                ord("a"),
                ord("A"),
                2424832,
                81,
            ):
                # Volta exatamente um quadro e pausa.
                pausado = True

                indice_quadro = max(
                    0,
                    indice_quadro - 1,
                )

                captura_replay.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    indice_quadro,
                )

                sucesso, frame = captura_replay.read()

                if sucesso:
                    quadro_atual = frame

                proximo_instante_frame = time.perf_counter()

            elif tecla in (
                ord("d"),
                ord("D"),
                2555904,
                83,
            ):
                # Avança exatamente um quadro e pausa.
                pausado = True

                indice_quadro = min(
                    max(0, total_quadros - 1),
                    indice_quadro + 1,
                )

                captura_replay.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    indice_quadro,
                )

                sucesso, frame = captura_replay.read()

                if sucesso:
                    quadro_atual = frame

                proximo_instante_frame = time.perf_counter()

            try:
                if (
                    cv2.getWindowProperty(
                        nome_janela,
                        cv2.WND_PROP_VISIBLE,
                    ) < 1
                ):
                    break

            except cv2.error:
                break

    finally:
        captura_replay.release()

        try:
            cv2.destroyWindow(nome_janela)

        except cv2.error:
            pass


# =========================================================
# PROGRAMA PRINCIPAL
# =========================================================

def main():
    captura = CapturaWebcam(indice_camera=0)
    salvador = SalvadorReplay()

    if not captura.iniciar():
        print(captura.erro)
        return

    nome_janela = "VAR Futevolei"

    cv2.namedWindow(
        nome_janela,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        nome_janela,
        1100,
        650,
    )

    print("\n" + "=" * 48)
    print("VAR FUTEVOLEI INICIADO")
    print("=" * 48)
    print("Aguarde o buffer acumular os primeiros segundos.")
    print("ESPACO: salvar e abrir os últimos 30 segundos.")
    print("ESC: encerrar o programa.")
    print("=" * 48 + "\n")

    executando = True

    try:
        while executando:
            # Abre automaticamente o replay após o salvamento.
            replay_pronto = salvador.obter_replay_pronto()

            if replay_pronto is not None:
                visualizar_replay(replay_pronto)
                continue

            frame = captura.obter_frame()

            if frame is None:
                if captura.erro:
                    print(captura.erro)
                    break

                cv2.waitKey(10)
                continue

            duracao, quantidade, memoria_mb = (
                captura.obter_status_buffer()
            )

            texto = (
                f"AO VIVO | Buffer: "
                f"{duracao:.1f}/{DURACAO_BUFFER}s"
                f" | Quadros: {quantidade}"
                f" | Memoria: {memoria_mb:.1f} MB"
            )

            tela = frame.copy()

            cv2.rectangle(
                tela,
                (0, 0),
                (tela.shape[1], 45),
                (25, 25, 25),
                -1,
            )

            cv2.putText(
                tela,
                texto,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if salvador.esta_salvando():
                texto_salvamento = "SALVANDO REPLAY..."

                tamanho_texto = cv2.getTextSize(
                    texto_salvamento,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2,
                )[0]

                posicao_x = max(
                    15,
                    tela.shape[1] - tamanho_texto[0] - 20,
                )

                cv2.putText(
                    tela,
                    texto_salvamento,
                    (posicao_x, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(nome_janela, tela)

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == 27:
                executando = False

            elif tecla == 32:
                clipe = captura.obter_replay()
                salvador.iniciar(clipe)

            try:
                if (
                    cv2.getWindowProperty(
                        nome_janela,
                        cv2.WND_PROP_VISIBLE,
                    ) < 1
                ):
                    executando = False

            except cv2.error:
                executando = False

    except KeyboardInterrupt:
        print("\nEncerrando o programa...")

    finally:
        captura.parar()
        salvador.aguardar()
        cv2.destroyAllWindows()
        print("VAR Futevolei encerrado.")


if __name__ == "__main__":
    main()
