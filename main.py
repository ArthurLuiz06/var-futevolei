
import cv2
import threading
import time
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
                        self.erro = (
                            "Falha ao capturar imagem da webcam."
                        )
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
        # Fecha o arquivo antes de tentar abri-lo.
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
# VISUALIZADOR DE REPLAY
# CÂMERA LENTA E QUADRO A QUADRO
# =========================================================

def visualizar_replay(caminho_arquivo):
    """
    Abre o replay salvo em uma janela própria.

    CONTROLES:
    ESPAÇO: pausar ou continuar
    1: câmera lenta 0,25x
    2: câmera lenta 0,5x
    3: velocidade normal 1x
    4: velocidade rápida 2x
    A: voltar um quadro
    D: avançar um quadro
    Q ou ESC: fechar o replay e voltar ao vivo
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

    indice_quadro = 0
    velocidade = 1.0
    pausado = False
    quadro_atual = None

    nome_janela = "VAR Futevolei - Revisao do Replay"

    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nome_janela, 1100, 650)

    print("\n" + "=" * 54)
    print("VISUALIZADOR DE REPLAY")
    print("ESPACO: pausar/continuar")
    print("1: 0,25x | 2: 0,5x | 3: 1x | 4: 2x")
    print("A: voltar quadro | D: avancar quadro")
    print("Q ou ESC: fechar o replay")
    print("=" * 54 + "\n")

    try:
        while True:

            # Carrega o próximo quadro quando estiver reproduzindo.
            if not pausado or quadro_atual is None:
                sucesso, frame = captura_replay.read()

                if not sucesso:
                    # Ao chegar ao fim, pausa no último quadro.
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

            # Barra superior.
            cv2.rectangle(
                tela,
                (0, 0),
                (largura, 44),
                (25, 25, 25),
                -1,
            )

            estado = "PAUSADO" if pausado else "REPRODUZINDO"

            texto = (
                f"{estado} | Velocidade: {velocidade:g}x | "
                f"Quadro: {indice_quadro + 1}/"
                f"{max(total_quadros, 1)}"
            )

            cv2.putText(
                tela,
                texto,
                (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Barra inferior com os controles.
            cv2.rectangle(
                tela,
                (0, max(44, altura - 38)),
                (largura, altura),
                (25, 25, 25),
                -1,
            )

            ajuda = (
                "ESPACO: pausa | 1: 0.25x | 2: 0.5x | "
                "3: 1x | 4: 2x | A/D: quadro | Q/ESC: sair"
            )

            cv2.putText(
                tela,
                ajuda,
                (12, altura - 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(nome_janela, tela)

            # A pausa não bloqueia indefinidamente a leitura de teclas.
            if pausado:
                atraso_ms = 30
            else:
                atraso_ms = max(
                    1,
                    int((1000 / fps_arquivo) / velocidade),
                )

            tecla = cv2.waitKey(atraso_ms) & 0xFF

            if tecla in (27, ord("q"), ord("Q")):
                break

            elif tecla == 32:
                # Espaço: pausa ou continua.
                pausado = not pausado

            elif tecla == ord("1"):
                velocidade = 0.25

            elif tecla == ord("2"):
                velocidade = 0.5

            elif tecla == ord("3"):
                velocidade = 1.0

            elif tecla == ord("4"):
                velocidade = 2.0

            elif tecla in (ord("a"), ord("A")):
                # Volta um quadro e pausa.
                pausado = True

                indice_quadro = max(0, indice_quadro - 1)

                captura_replay.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    indice_quadro,
                )

                sucesso, frame = captura_replay.read()

                if sucesso:
                    quadro_atual = frame

            elif tecla in (ord("d"), ord("D")):
                # Avança um quadro e pausa.
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

            # Detecta o fechamento pelo botão X.
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

    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nome_janela, 1100, 650)

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

            # Abre o replay quando a gravação terminar.
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

            # Barra superior com o status.
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

            # Indicador de salvamento.
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
                # Copia o buffer e inicia o salvamento em segundo plano.
                clipe = captura.obter_replay()
                salvador.iniciar(clipe)

            # Detecta o fechamento da janela ao vivo.
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

        # Espera a gravação terminar antes de encerrar.
        salvador.aguardar()

        cv2.destroyAllWindows()

        print("VAR Futevolei encerrado.")


if __name__ == "__main__":
    main()
