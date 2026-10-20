
import cv2
import threading
import time
import numpy as np

from collections import deque
from datetime import datetime
from pathlib import Path


DURACAO_BUFFER = 10
QUALIDADE_JPEG = 80
FPS_REPLAY = 30

PASTA_REPLAYS = Path(__file__).resolve().parent / "replays"


class CapturaWebcam:
    def __init__(self, indice_camera=0):
        self.indice_camera = indice_camera
        self.camera = None

        self.frame_atual = None
        self.lock_frame = threading.Lock()

        # Cada item: (instante, imagem JPEG comprimida).
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
        """Retorna uma cópia do buffer para reprodução ou gravação."""
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


def decodificar_frame(frame_jpeg):
    """Converte os bytes JPEG em uma imagem OpenCV."""
    return cv2.imdecode(
        np.frombuffer(frame_jpeg, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )


def mostrar_replay(nome_janela, clipe):
    """Reproduz manualmente um clipe do buffer."""
    if not clipe:
        print("Nenhum frame disponível para replay.")
        return True

    intervalo = 1 / FPS_REPLAY

    for instante, frame_jpeg in clipe:
        frame = decodificar_frame(frame_jpeg)

        if frame is None:
            continue

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 50),
            (25, 25, 25),
            -1,
        )

        cv2.putText(
            frame,
            "REPLAY | R: voltar ao vivo | ESC: sair",
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(nome_janela, frame)

        tecla = cv2.waitKey(
            max(1, int(intervalo * 1000))
        ) & 0xFF

        if tecla in (ord("r"), ord("R")):
            print("Retornando ao vídeo ao vivo...")
            return True

        if tecla == 27:
            return False

    print("Replay finalizado. Retornando ao vídeo ao vivo.")
    return True


def salvar_replay_mp4(clipe):
    """Salva o conteúdo do buffer em um arquivo MP4."""
    if not clipe:
        print("Não há imagens disponíveis para salvar.")
        return False

    PASTA_REPLAYS.mkdir(parents=True, exist_ok=True)

    primeiro_frame = None

    for _, frame_jpeg in clipe:
        primeiro_frame = decodificar_frame(frame_jpeg)

        if primeiro_frame is not None:
            break

    if primeiro_frame is None:
        print("Não foi possível decodificar os frames do replay.")
        return False

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
        return False

    quadros_salvos = 0
    erro_gravacao = None

    try:
        for instante, frame_jpeg in clipe:
            frame = decodificar_frame(frame_jpeg)

            if frame is None:
                continue

            if (
                frame.shape[1] != largura
                or frame.shape[0] != altura
            ):
                frame = cv2.resize(frame, (largura, altura))

            # Grava apenas a imagem da câmera, sem a interface.
            gravador.write(frame)
            quadros_salvos += 1

    except Exception as erro:
        erro_gravacao = erro

    finally:
        gravador.release()

    if erro_gravacao is not None:
        print(f"Erro ao gravar o replay: {erro_gravacao}")
        return False

    if quadros_salvos == 0:
        print("Nenhum quadro foi salvo.")
        return False

    if (
        not caminho_arquivo.exists()
        or caminho_arquivo.stat().st_size == 0
    ):
        print("O arquivo MP4 não foi criado corretamente.")
        return False

    print("Replay salvo com sucesso!")
    print(f"Arquivo: {caminho_arquivo}")
    print(f"Quadros salvos: {quadros_salvos}")
    print(
        f"Tamanho: "
        f"{caminho_arquivo.stat().st_size / (1024 * 1024):.2f} MB"
    )

    return True


def main():
    captura = CapturaWebcam(indice_camera=0)

    if not captura.iniciar():
        print(captura.erro)
        return

    nome_janela = "VAR Futevolei"

    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nome_janela, 1100, 650)

    print("VAR Futevolei iniciado.")
    print("Aguarde o buffer acumular alguns segundos.")
    print("ESPACO: salvar os últimos segundos em MP4.")
    print("R: voltar ao vivo durante um replay manual.")
    print("ESC: encerrar.")

    executando = True

    try:
        while executando:
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

            cv2.imshow(nome_janela, tela)

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == 27:
                executando = False

            elif tecla == 32:
                # Salva o buffer sem iniciar a reprodução.
                clipe = captura.obter_replay()
                salvar_replay_mp4(clipe)

    except KeyboardInterrupt:
        print("Encerrando o programa...")

    finally:
        captura.parar()
        cv2.destroyAllWindows()
        print("VAR encerrado.")


if __name__ == "__main__":
    main()
