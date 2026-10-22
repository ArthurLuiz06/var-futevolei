
import cv2
import threading
import time
import math
import bisect
import numpy as np

from datetime import datetime
from pathlib import Path


# =========================================================
# CONFIGURAÇÕES
# =========================================================

DURACAO_BUFFER = 30
QUALIDADE_JPEG = 92
FPS_REPLAY = 30

PASTA_REPLAYS = Path(__file__).resolve().parent / "replays"

LARGURA_JANELA_AO_VIVO = 1200
ALTURA_JANELA_AO_VIVO = 720

LARGURA_PAINEL_CAMERA = 580
ALTURA_IMAGEM_CAMERA = 326
ALTURA_CABECALHO_CAMERA = 42

LARGURA_REPLAY = 1100
ALTURA_REPLAY = 680

NOME_JANELA_AO_VIVO = "VAR Futevolei - Cameras ao Vivo"
NOME_JANELA_REPLAY = "VAR Futevolei - Revisao do Replay"


# =========================================================
# CAPTURA DA WEBCAM E BUFFER CONTÍNUO
# =========================================================

class CapturaWebcam:
    def __init__(self, indice_camera=0, nome_camera=None):
        self.indice_camera = indice_camera
        self.nome_camera = nome_camera or f"Câmera {indice_camera + 1}"
        self.camera = None

        self.frame_atual = None
        self.lock_frame = threading.Lock()

        # Cada item: (instante_monotônico, imagem_jpeg).
        self.buffer = []
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
            self.erro = (
                f"Não foi possível abrir a {self.nome_camera} "
                f"(índice {self.indice_camera})."
            )
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
                if self.camera is None:
                    break

                sucesso, frame = self.camera.read()

                if not sucesso:
                    if self.ativa:
                        self.erro = (
                            f"Falha ao capturar imagem da "
                            f"{self.nome_camera}."
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
                        _, dados_antigos = self.buffer.pop(0)
                        self.total_bytes_buffer -= len(dados_antigos)

        except Exception as erro:
            self.erro = (
                f"Erro durante a captura da {self.nome_camera}: "
                f"{erro}"
            )

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

            duracao = self.buffer[-1][0] - self.buffer[0][0]

            return duracao, quantidade, memoria_mb

    def parar(self):
        self.ativa = False

        if self.thread is not None:
            self.thread.join(timeout=3)

        if self.camera is not None:
            self.camera.release()
            self.camera = None


# =========================================================
# FUNÇÕES DE VÍDEO
# =========================================================

def decodificar_frame(frame_jpeg):
    """Converte bytes JPEG em uma imagem OpenCV."""
    return cv2.imdecode(
        np.frombuffer(frame_jpeg, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )


def sincronizar_clipes(clipes_por_camera):
    """
    Cria uma linha do tempo compartilhada para as câmeras.

    Os buffers usam time.monotonic(), permitindo comparar
    os instantes capturados durante a mesma execução.

    Os vídeos resultantes terão a mesma quantidade de quadros
    e FPS para facilitar a alternância entre ângulos.
    """
    clipes_validos = {}

    for indice, clipe in clipes_por_camera.items():
        clipe_valido = [
            (instante, imagem)
            for instante, imagem in clipe
            if imagem is not None
        ]

        if clipe_valido:
            clipes_validos[indice] = clipe_valido

    if not clipes_validos:
        print("Nenhuma câmera possui quadros disponíveis.")
        return {}

    inicio_comum = max(
        clipe[0][0]
        for clipe in clipes_validos.values()
    )

    fim_comum = min(
        clipe[-1][0]
        for clipe in clipes_validos.values()
    )

    if fim_comum <= inicio_comum:
        print(
            "Não existe intervalo compartilhado suficiente "
            "para sincronizar as câmeras."
        )
        return {}

    quantidade_quadros = max(
        1,
        int((fim_comum - inicio_comum) * FPS_REPLAY) + 1,
    )

    clipes_sincronizados = {}

    for indice, clipe in clipes_validos.items():
        instantes = [item[0] for item in clipe]
        quadros = []
        posicao = 0

        for numero_quadro in range(quantidade_quadros):
            instante_alvo = (
                inicio_comum + numero_quadro / FPS_REPLAY
            )

            posicao = bisect.bisect_left(
                instantes,
                instante_alvo,
                lo=posicao,
            )

            if posicao <= 0:
                escolhido = 0
            elif posicao >= len(clipe):
                escolhido = len(clipe) - 1
            else:
                anterior = posicao - 1

                distancia_anterior = abs(
                    instantes[anterior] - instante_alvo
                )

                distancia_seguinte = abs(
                    instantes[posicao] - instante_alvo
                )

                escolhido = (
                    anterior
                    if distancia_anterior <= distancia_seguinte
                    else posicao
                )

            quadros.append(clipe[escolhido][1])

        clipes_sincronizados[indice] = quadros

    return clipes_sincronizados


def salvar_replay_mp4(clipe, identificador_lance, indice_camera):
    """Salva um ângulo do lance em MP4."""
    if not clipe:
        print(
            f"Câmera {indice_camera + 1}: "
            "não há quadros para salvar."
        )
        return None

    PASTA_REPLAYS.mkdir(parents=True, exist_ok=True)

    primeiro_frame = None

    for frame_jpeg in clipe:
        primeiro_frame = decodificar_frame(frame_jpeg)

        if primeiro_frame is not None:
            break

    if primeiro_frame is None:
        print(
            f"Câmera {indice_camera + 1}: "
            "não foi possível decodificar os quadros."
        )
        return None

    altura, largura = primeiro_frame.shape[:2]

    nome_arquivo = (
        f"{identificador_lance}_camera{indice_camera + 1}.mp4"
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

        print(
            f"Câmera {indice_camera + 1}: "
            "não foi possível criar o arquivo MP4."
        )
        return None

    quadros_salvos = 0
    erro_gravacao = None

    try:
        for frame_jpeg in clipe:
            frame = decodificar_frame(frame_jpeg)

            if frame is None:
                continue

            if (
                frame.shape[1] != largura
                or frame.shape[0] != altura
            ):
                frame = cv2.resize(frame, (largura, altura))

            # Salva somente o vídeo, sem a interface.
            gravador.write(frame)
            quadros_salvos += 1

    except Exception as erro:
        erro_gravacao = erro

    finally:
        gravador.release()

    if erro_gravacao is not None:
        print(
            f"Erro ao gravar a Câmera {indice_camera + 1}: "
            f"{erro_gravacao}"
        )
        return None

    if quadros_salvos == 0:
        print(
            f"Câmera {indice_camera + 1}: "
            "nenhum quadro foi salvo."
        )
        return None

    if (
        not caminho_arquivo.exists()
        or caminho_arquivo.stat().st_size == 0
    ):
        print(
            f"Câmera {indice_camera + 1}: "
            "o arquivo MP4 não foi criado corretamente."
        )
        return None

    tamanho_mb = caminho_arquivo.stat().st_size / (1024 * 1024)

    print(
        f"Câmera {indice_camera + 1} salva: "
        f"{caminho_arquivo.name} | "
        f"{quadros_salvos} quadros | {tamanho_mb:.2f} MB"
    )

    return caminho_arquivo


# =========================================================
# SALVAMENTO DOS ÂNGULOS EM SEGUNDO PLANO
# =========================================================

class SalvadorReplay:
    def __init__(self):
        self.thread = None
        self.lock = threading.Lock()
        self.em_andamento = False
        self.replay_pronto = None

    def iniciar(self, clipes_por_camera):
        """Recebe os buffers e inicia o salvamento."""
        clipes_validos = {
            indice: list(clipe)
            for indice, clipe in clipes_por_camera.items()
            if clipe
        }

        if not clipes_validos:
            print("Nenhuma câmera possui quadros para salvar.")
            return False

        with self.lock:
            if self.em_andamento:
                print(
                    "Já existe um replay sendo salvo. "
                    "Aguarde a conclusão."
                )
                return False

            self.em_andamento = True
            self.replay_pronto = None

            try:
                self.thread = threading.Thread(
                    target=self._executar_salvamento,
                    args=(clipes_validos,),
                    daemon=False,
                )
                self.thread.start()

            except Exception:
                self.em_andamento = False
                raise

        print(
            "Solicitação recebida! "
            "Os ângulos serão salvos em segundo plano."
        )

        return True

    def _executar_salvamento(self, clipes_por_camera):
        resultado = None

        try:
            sincronizados = sincronizar_clipes(clipes_por_camera)

            if sincronizados:
                identificador = (
                    "replay_"
                    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
                )

                caminhos = {}

                for indice, quadros in sincronizados.items():
                    caminho = salvar_replay_mp4(
                        quadros,
                        identificador,
                        indice,
                    )

                    if caminho is not None:
                        caminhos[indice] = caminho

                if caminhos:
                    resultado = {
                        "caminhos": caminhos,
                        "identificador": identificador,
                    }

                    print("\n" + "=" * 48)
                    print("LANCE SALVO COM SUCESSO!")
                    print("=" * 48)

                    for indice, caminho in caminhos.items():
                        print(f"Câmera {indice + 1}: {caminho}")

                    print("Abrindo o visualizador de replay...")
                    print("=" * 48)

        except Exception as erro:
            print(f"Erro inesperado ao salvar o replay: {erro}")

        finally:
            with self.lock:
                self.replay_pronto = resultado
                self.em_andamento = False

    def esta_salvando(self):
        with self.lock:
            return self.em_andamento

    def obter_replay_pronto(self):
        """Entrega os vídeos recém-salvos uma única vez."""
        with self.lock:
            resultado = self.replay_pronto
            self.replay_pronto = None
            return resultado

    def aguardar(self):
        """Aguarda o término do salvamento."""
        with self.lock:
            thread = self.thread

        if thread is not None and thread.is_alive():
            print("Aguardando a conclusão do salvamento...")
            thread.join()


# =========================================================
# INTERFACE E FORMATAÇÃO
# =========================================================

def desenhar_texto_legivel(
    imagem,
    texto,
    posicao,
    escala=0.42,
    cor=(255, 255, 255),
):
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


def ajustar_imagem(frame, largura, altura):
    """Redimensiona mantendo a proporção original."""
    altura_frame, largura_frame = frame.shape[:2]

    escala = min(
        largura / largura_frame,
        altura / altura_frame,
    )

    nova_largura = max(1, int(largura_frame * escala))
    nova_altura = max(1, int(altura_frame * escala))

    redimensionado = cv2.resize(
        frame,
        (nova_largura, nova_altura),
        interpolation=cv2.INTER_AREA,
    )

    tela = np.zeros((altura, largura, 3), dtype=np.uint8)
    tela[:] = (12, 15, 20)

    x = (largura - nova_largura) // 2
    y = (altura - nova_altura) // 2

    tela[
        y:y + nova_altura,
        x:x + nova_largura,
    ] = redimensionado

    return tela


# =========================================================
# VISUALIZADOR MULTICÂMERA DE REPLAY
# =========================================================

def visualizar_replay(
    resultado_replay,
    capturas_ao_vivo=None,
    nome_janela_ao_vivo=NOME_JANELA_AO_VIVO,
    salvador=None,
):
    """
    Reproduz o lance enquanto mantém a janela ao vivo atualizada.

    A captura das webcams continua nas respectivas threads.
    Durante o replay, esta função também atualiza a interface
    ao vivo a cada ciclo da reprodução.
    """
    caminhos = resultado_replay.get("caminhos", {})

    if not caminhos:
        print("Não há vídeos disponíveis para reproduzir.")
        return

    capturas = {}

    for indice, caminho in caminhos.items():
        captura = cv2.VideoCapture(str(caminho))

        if captura.isOpened():
            capturas[indice] = captura
        else:
            captura.release()
            print(
                f"Não foi possível abrir o vídeo da "
                f"Câmera {indice + 1}."
            )

    if not capturas:
        print("Não foi possível abrir nenhum ângulo do replay.")
        return

    indices_disponiveis = sorted(capturas.keys())

    total_quadros_por_camera = {
        indice: int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
        for indice, captura in capturas.items()
    }

    fps_arquivo = FPS_REPLAY

    total_quadros = min(total_quadros_por_camera.values())

    if total_quadros <= 0:
        for captura in capturas.values():
            captura.release()

        print("O replay não contém quadros válidos.")
        return

    duracao_total = total_quadros / fps_arquivo

    camera_atual = indices_disponiveis[0]
    indice_quadro = 0
    velocidade = 1.0
    pausado = False

    quadros_atuais = {}

    # Carrega o primeiro quadro de cada câmera.
    for indice, captura in capturas.items():
        captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
        sucesso, frame = captura.read()

        if sucesso:
            quadros_atuais[indice] = frame

    if camera_atual not in quadros_atuais:
        for captura in capturas.values():
            captura.release()

        print("Não foi possível ler o primeiro quadro do replay.")
        return

    estado_mouse = {
        "solicitar_quadro": None,
        "solicitar_camera": None,
    }

    botoes_camera = {}

    def posicionar_em_quadro(indice_alvo):
        """Posiciona os vídeos no mesmo quadro."""
        nonlocal indice_quadro

        indice_quadro = max(
            0,
            min(indice_alvo, total_quadros - 1),
        )

        for indice, captura in capturas.items():
            captura.set(
                cv2.CAP_PROP_POS_FRAMES,
                indice_quadro,
            )

            sucesso, frame = captura.read()

            if sucesso:
                quadros_atuais[indice] = frame

    def avancar_quadro_sequencial(quantidade):
        """
        Avança sequencialmente no vídeo da câmera atual,
        evitando reposicionar o arquivo em cada quadro.
        """
        nonlocal indice_quadro

        novo_indice = min(
            total_quadros - 1,
            indice_quadro + quantidade,
        )

        quantidade_real = novo_indice - indice_quadro

        if quantidade_real <= 0:
            return

        captura = capturas[camera_atual]

        for _ in range(quantidade_real - 1):
            if not captura.grab():
                break

        sucesso, frame = captura.read()

        if sucesso:
            quadros_atuais[camera_atual] = frame
            indice_quadro = novo_indice
        else:
            posicionar_em_quadro(novo_indice)

    def ao_clicar_mouse(evento, x, y, flags, parametro):
        if evento != cv2.EVENT_LBUTTONDOWN:
            return

        # Seleção de câmera.
        for indice, area in botoes_camera.items():
            x1, y1, x2, y2 = area

            if x1 <= x <= x2 and y1 <= y <= y2:
                estado_mouse["solicitar_camera"] = indice
                return

        # Barra de progresso.
        y_barra = 610
        x_inicio = 20
        x_fim = LARGURA_REPLAY - 20

        if (
            y_barra - 12 <= y <= y_barra + 12
            and x_inicio <= x <= x_fim
        ):
            proporcao = (x - x_inicio) / (x_fim - x_inicio)

            estado_mouse["solicitar_quadro"] = int(
                proporcao * max(total_quadros - 1, 0)
            )

    cv2.namedWindow(NOME_JANELA_REPLAY, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        NOME_JANELA_REPLAY,
        LARGURA_REPLAY,
        ALTURA_REPLAY,
    )
    cv2.setMouseCallback(
        NOME_JANELA_REPLAY,
        ao_clicar_mouse,
    )

    print("\n" + "=" * 54)
    print("VISUALIZADOR MULTICÂMERA DE REPLAY")
    print("O ao vivo continuará funcionando durante a análise.")
    print("Clique em CÂMERA 1 ou CÂMERA 2 para mudar o ângulo.")
    print("ESPAÇO: pausar/continuar")
    print("1: 0,25x | 2: 0,5x | 3: 1x | 4: 2x")
    print("A / seta esquerda: quadro anterior")
    print("D / seta direita: próximo quadro")
    print("R: reiniciar | clique na barra: buscar momento")
    print("Q ou ESC: fechar replay e voltar ao ao vivo")
    print("=" * 54 + "\n")

    intervalo_frame = 1.0 / (fps_arquivo * velocidade)
    proximo_instante_frame = time.perf_counter() + intervalo_frame

    try:
        while True:
            # =================================================
            # ATUALIZAÇÃO DO AO VIVO DURANTE O REPLAY
            # =================================================
            #
            # A captura continua nas threads. Aqui atualizamos
            # a janela visual do ao vivo enquanto o replay roda.
            # Não abrimos novamente nenhuma webcam.
            #
            if capturas_ao_vivo:
                try:
                    tela_ao_vivo = montar_grade_ao_vivo(
                        capturas_ao_vivo,
                        salvador.esta_salvando() if salvador else False,
                    )

                    cv2.imshow(
                        nome_janela_ao_vivo,
                        tela_ao_vivo,
                    )

                except cv2.error:
                    pass

            # ---------------------------------------------
            # BUSCA NA BARRA DE PROGRESSO
            # ---------------------------------------------

            quadro_solicitado = estado_mouse["solicitar_quadro"]

            if quadro_solicitado is not None:
                posicionar_em_quadro(quadro_solicitado)

                estado_mouse["solicitar_quadro"] = None
                pausado = True
                proximo_instante_frame = (
                    time.perf_counter() + intervalo_frame
                )

            # ---------------------------------------------
            # TROCA DE CÂMERA
            # ---------------------------------------------

            camera_solicitada = estado_mouse["solicitar_camera"]

            if (
                camera_solicitada is not None
                and camera_solicitada in capturas
            ):
                camera_atual = camera_solicitada

                capturas[camera_atual].set(
                    cv2.CAP_PROP_POS_FRAMES,
                    indice_quadro,
                )

                sucesso, frame = capturas[camera_atual].read()

                if sucesso:
                    quadros_atuais[camera_atual] = frame

                estado_mouse["solicitar_camera"] = None

            # ---------------------------------------------
            # QUADRO ATUAL DO REPLAY
            # ---------------------------------------------

            frame = quadros_atuais.get(camera_atual)

            if frame is None:
                posicionar_em_quadro(indice_quadro)
                frame = quadros_atuais.get(camera_atual)

                if frame is None:
                    break

            tela = np.zeros(
                (ALTURA_REPLAY, LARGURA_REPLAY, 3),
                dtype=np.uint8,
            )
            tela[:] = (18, 22, 29)

            cv2.rectangle(
                tela,
                (0, 0),
                (LARGURA_REPLAY, 44),
                (30, 36, 46),
                -1,
            )

            botoes_camera.clear()
            posicao_x = 14

            for indice in indices_disponiveis:
                texto_botao = f"CAMERA {indice + 1}"

                tamanho = cv2.getTextSize(
                    texto_botao,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    1,
                )[0]

                largura_botao = tamanho[0] + 26

                x1 = posicao_x
                y1 = 7
                x2 = x1 + largura_botao
                y2 = 37

                cor_fundo = (
                    (0, 130, 190)
                    if indice == camera_atual
                    else (55, 63, 76)
                )

                cv2.rectangle(
                    tela,
                    (x1, y1),
                    (x2, y2),
                    cor_fundo,
                    -1,
                )

                cv2.rectangle(
                    tela,
                    (x1, y1),
                    (x2, y2),
                    (105, 120, 140),
                    1,
                )

                cv2.putText(
                    tela,
                    texto_botao,
                    (x1 + 13, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                botoes_camera[indice] = (x1, y1, x2, y2)
                posicao_x = x2 + 9

            estado_texto = "PAUSADO" if pausado else "REPRODUZINDO"

            cv2.putText(
                tela,
                f"{estado_texto} | {velocidade:g}x",
                (posicao_x + 8, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (100, 230, 150),
                1,
                cv2.LINE_AA,
            )

            video = ajustar_imagem(
                frame,
                LARGURA_REPLAY,
                530,
            )

            tela[48:578, :] = video

            cv2.rectangle(
                tela,
                (12, 56),
                (155, 84),
                (15, 18, 24),
                -1,
            )

            cv2.putText(
                tela,
                f"ANGULO {camera_atual + 1}",
                (21, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # ---------------------------------------------
            # BARRA DE PROGRESSO
            # ---------------------------------------------

            y_barra = 610
            x_inicio = 20
            x_fim = LARGURA_REPLAY - 20

            progresso = (
                indice_quadro / max(total_quadros - 1, 1)
            )

            x_progresso = int(
                x_inicio + progresso * (x_fim - x_inicio)
            )

            cv2.line(
                tela,
                (x_inicio, y_barra),
                (x_fim, y_barra),
                (105, 115, 130),
                4,
                cv2.LINE_AA,
            )

            cv2.line(
                tela,
                (x_inicio, y_barra),
                (x_progresso, y_barra),
                (0, 190, 255),
                4,
                cv2.LINE_AA,
            )

            cv2.circle(
                tela,
                (x_progresso, y_barra),
                6,
                (245, 245, 245),
                -1,
                cv2.LINE_AA,
            )

            tempo_atual = indice_quadro / fps_arquivo
            tempo_restante = max(
                0.0,
                duracao_total - tempo_atual,
            )

            desenhar_texto_legivel(
                tela,
                formatar_tempo(tempo_atual),
                (20, 638),
                escala=0.43,
            )

            texto_restante = f"-{formatar_tempo(tempo_restante)}"

            tamanho_restante = cv2.getTextSize(
                texto_restante,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                1,
            )[0]

            desenhar_texto_legivel(
                tela,
                texto_restante,
                (
                    LARGURA_REPLAY - tamanho_restante[0] - 20,
                    638,
                ),
                escala=0.43,
            )

            cv2.putText(
                tela,
                "ESPACO: pausar | A/D: quadro | 1-4: velocidade | R: reiniciar | Q: sair",
                (20, 665),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (205, 213, 225),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(NOME_JANELA_REPLAY, tela)

            # ---------------------------------------------
            # TEMPORIZAÇÃO DA REPRODUÇÃO
            # ---------------------------------------------

            if pausado:
                tecla = cv2.waitKeyEx(15)
            else:
                agora = time.perf_counter()
                espera = proximo_instante_frame - agora

                if espera > 0:
                    tecla = cv2.waitKeyEx(
                        max(1, int(math.ceil(espera * 1000)))
                    )
                else:
                    tecla = cv2.waitKeyEx(1)

            # ---------------------------------------------
            # CONTROLES
            # ---------------------------------------------

            if tecla in (27, ord("q"), ord("Q")):
                break

            elif tecla == 32:
                pausado = not pausado

                proximo_instante_frame = (
                    time.perf_counter() + intervalo_frame
                )

            elif tecla == ord("1"):
                velocidade = 0.25
                intervalo_frame = 1.0 / (
                    fps_arquivo * velocidade
                )
                proximo_instante_frame = (
                    time.perf_counter() + intervalo_frame
                )

            elif tecla == ord("2"):
                velocidade = 0.5
                intervalo_frame = 1.0 / (
                    fps_arquivo * velocidade
                )
                proximo_instante_frame = (
                    time.perf_counter() + intervalo_frame
                )

            elif tecla == ord("3"):
                velocidade = 1.0
                intervalo_frame = 1.0 / (
                    fps_arquivo * velocidade
                )
                proximo_instante_frame = (
                    time.perf_counter() + intervalo_frame
                )

            elif tecla == ord("4"):
                velocidade = 2.0
                intervalo_frame = 1.0 / (
                    fps_arquivo * velocidade
                )
                proximo_instante_frame = (
                    time.perf_counter() + intervalo_frame
                )

            elif tecla in (ord("r"), ord("R")):
                posicionar_em_quadro(0)
                pausado = False

                proximo_instante_frame = (
                    time.perf_counter() + intervalo_frame
                )

            elif tecla in (
                ord("a"),
                ord("A"),
                2424832,
                81,
            ):
                pausado = True
                posicionar_em_quadro(indice_quadro - 1)

                proximo_instante_frame = (
                    time.perf_counter() + intervalo_frame
                )

            elif tecla in (
                ord("d"),
                ord("D"),
                2555904,
                83,
            ):
                pausado = True
                posicionar_em_quadro(indice_quadro + 1)

                proximo_instante_frame = (
                    time.perf_counter() + intervalo_frame
                )

            # ---------------------------------------------
            # AVANÇO NORMAL DO VÍDEO
            # ---------------------------------------------

            if not pausado:
                agora = time.perf_counter()

                if agora >= proximo_instante_frame:
                    atraso = agora - proximo_instante_frame

                    quadros_para_avancar = (
                        1 + int(atraso / intervalo_frame)
                    )

                    quadros_para_avancar = min(
                        quadros_para_avancar,
                        total_quadros - 1 - indice_quadro,
                    )

                    if quadros_para_avancar > 0:
                        avancar_quadro_sequencial(
                            quadros_para_avancar
                        )

                    if indice_quadro >= total_quadros - 1:
                        pausado = True
                    else:
                        proximo_instante_frame += (
                            quadros_para_avancar * intervalo_frame
                        )

                        agora = time.perf_counter()

                        if proximo_instante_frame < agora:
                            proximo_instante_frame = (
                                agora + intervalo_frame
                            )

            # ---------------------------------------------
            # VERIFICA SE AS JANELAS CONTINUAM ABERTAS
            # ---------------------------------------------

            try:
                if (
                    cv2.getWindowProperty(
                        NOME_JANELA_REPLAY,
                        cv2.WND_PROP_VISIBLE,
                    ) < 1
                ):
                    break
            except cv2.error:
                break

    finally:
        for captura in capturas.values():
            captura.release()

        try:
            cv2.destroyWindow(NOME_JANELA_REPLAY)
        except cv2.error:
            pass


# =========================================================
# GRADE DE CÂMERAS AO VIVO
# =========================================================

def criar_painel_camera(captura, indice_camera):
    largura = LARGURA_PAINEL_CAMERA
    altura_cabecalho = ALTURA_CABECALHO_CAMERA
    altura_imagem = ALTURA_IMAGEM_CAMERA

    painel = np.zeros(
        (
            altura_cabecalho + altura_imagem,
            largura,
            3,
        ),
        dtype=np.uint8,
    )

    cv2.rectangle(
        painel,
        (0, 0),
        (largura, altura_cabecalho),
        (32, 38, 48),
        -1,
    )

    if captura is None:
        titulo = f"CAMERA {indice_camera + 1}"
        status = "NAO DISPONIVEL"

        cv2.putText(
            painel,
            titulo,
            (14, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )

        tamanho_status = cv2.getTextSize(
            status,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            1,
        )[0]

        cv2.putText(
            painel,
            status,
            (largura - tamanho_status[0] - 12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (80, 160, 255),
            1,
            cv2.LINE_AA,
        )

        mensagem = "Conecte a webcam e reinicie o programa."

        tamanho_mensagem = cv2.getTextSize(
            mensagem,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            1,
        )[0]

        cv2.putText(
            painel,
            mensagem,
            (
                max(10, (largura - tamanho_mensagem[0]) // 2),
                altura_cabecalho + altura_imagem // 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 190, 205),
            1,
            cv2.LINE_AA,
        )

        return painel

    frame = captura.obter_frame()

    cv2.putText(
        painel,
        captura.nome_camera.upper(),
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    if frame is None:
        status = "AGUARDANDO IMAGEM"
        cor_status = (0, 190, 255)
    else:
        status = "AO VIVO"
        cor_status = (80, 220, 120)

    tamanho_status = cv2.getTextSize(
        status,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        1,
    )[0]

    cv2.putText(
        painel,
        status,
        (largura - tamanho_status[0] - 12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        cor_status,
        1,
        cv2.LINE_AA,
    )

    if frame is None:
        mensagem = "Aguardando a primeira imagem..."

        tamanho_mensagem = cv2.getTextSize(
            mensagem,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            1,
        )[0]

        cv2.putText(
            painel,
            mensagem,
            (
                max(10, (largura - tamanho_mensagem[0]) // 2),
                altura_cabecalho + altura_imagem // 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 190, 205),
            1,
            cv2.LINE_AA,
        )

        return painel

    altura_frame, largura_frame = frame.shape[:2]

    escala = min(
        largura / largura_frame,
        altura_imagem / altura_frame,
    )

    nova_largura = max(1, int(largura_frame * escala))
    nova_altura = max(1, int(altura_frame * escala))

    frame_redimensionado = cv2.resize(
        frame,
        (nova_largura, nova_altura),
        interpolation=cv2.INTER_AREA,
    )

    x = (largura - nova_largura) // 2
    y = altura_cabecalho + (
        altura_imagem - nova_altura
    ) // 2

    painel[
        y:y + nova_altura,
        x:x + nova_largura,
    ] = frame_redimensionado

    return painel


def montar_grade_ao_vivo(capturas, salvando=False):
    painel_camera_1 = criar_painel_camera(
        capturas.get(0),
        0,
    )

    painel_camera_2 = criar_painel_camera(
        capturas.get(1),
        1,
    )

    espacamento = 12
    margem = 12

    altura_paineis = (
        ALTURA_CABECALHO_CAMERA + ALTURA_IMAGEM_CAMERA
    )

    largura_total = (
        margem * 2
        + LARGURA_PAINEL_CAMERA * 2
        + espacamento
    )

    altura_total = margem * 2 + altura_paineis + 76

    tela = np.zeros(
        (altura_total, largura_total, 3),
        dtype=np.uint8,
    )
    tela[:] = (20, 24, 31)

    x1 = margem
    x2 = margem + LARGURA_PAINEL_CAMERA + espacamento
    y_paineis = margem

    tela[
        y_paineis:y_paineis + altura_paineis,
        x1:x1 + LARGURA_PAINEL_CAMERA,
    ] = painel_camera_1

    tela[
        y_paineis:y_paineis + altura_paineis,
        x2:x2 + LARGURA_PAINEL_CAMERA,
    ] = painel_camera_2

    y_texto = y_paineis + altura_paineis + 28

    texto = (
        "SALVANDO OS DOIS ANGULOS..."
        if salvando
        else "ESPACO: salvar os ultimos 30 segundos das cameras"
    )

    cor = (0, 165, 255) if salvando else (225, 230, 238)

    cv2.putText(
        tela,
        texto,
        (margem, y_texto),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        cor,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        tela,
        "ESC: encerrar o programa",
        (margem, y_texto + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (170, 185, 205),
        1,
        cv2.LINE_AA,
    )

    return tela


# =========================================================
# PROGRAMA PRINCIPAL
# =========================================================

def main():
    salvador = SalvadorReplay()
    capturas = {}

    print("\n" + "=" * 54)
    print("INICIALIZANDO CAMERAS DO VAR FUTEVOLEI")
    print("=" * 54)

    for indice in (0, 1):
        captura_camera = CapturaWebcam(
            indice_camera=indice,
            nome_camera=f"Câmera {indice + 1}",
        )

        print(
            f"Tentando abrir a Câmera {indice + 1} "
            f"(índice {indice})..."
        )

        if captura_camera.iniciar():
            capturas[indice] = captura_camera
            print(f"Câmera {indice + 1} iniciada com sucesso.")
        else:
            print(captura_camera.erro)

    if not capturas:
        print("\nNenhuma webcam foi encontrada.")
        print("Verifique a conexão e tente novamente.")
        return

    cv2.namedWindow(NOME_JANELA_AO_VIVO, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(
        NOME_JANELA_AO_VIVO,
        LARGURA_JANELA_AO_VIVO,
        ALTURA_JANELA_AO_VIVO,
    )

    print("\n" + "=" * 54)
    print("VAR FUTEVOLEI INICIADO")
    print("=" * 54)
    print("As câmeras estão sendo exibidas lado a lado.")
    print("Cada câmera possui seu próprio buffer de 30 segundos.")
    print("ESPAÇO: salvar e abrir o replay dos ângulos disponíveis.")
    print("ESC: encerrar o programa.")
    print("=" * 54 + "\n")

    executando = True

    try:
        while executando:
            resultado_replay = salvador.obter_replay_pronto()

            if resultado_replay is not None:
                # A janela ao vivo não é fechada nem recriada.
                # O visualizador atualiza sua imagem enquanto
                # o replay está aberto.
                visualizar_replay(
                    resultado_replay,
                    capturas_ao_vivo=capturas,
                    nome_janela_ao_vivo=NOME_JANELA_AO_VIVO,
                    salvador=salvador,
                )

                # Ao sair do replay, o loop principal continua.
                continue

            tela = montar_grade_ao_vivo(
                capturas,
                salvador.esta_salvando(),
            )

            cv2.imshow(NOME_JANELA_AO_VIVO, tela)

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == 27:
                executando = False

            elif tecla == 32:
                clipes = {
                    indice: captura.obter_replay()
                    for indice, captura in capturas.items()
                }

                clipes = {
                    indice: clipe
                    for indice, clipe in clipes.items()
                    if clipe
                }

                if not clipes:
                    print(
                        "As câmeras ainda não possuem "
                        "quadros para salvar."
                    )
                else:
                    salvador.iniciar(clipes)

            try:
                if (
                    cv2.getWindowProperty(
                        NOME_JANELA_AO_VIVO,
                        cv2.WND_PROP_VISIBLE,
                    ) < 1
                ):
                    executando = False

            except cv2.error:
                executando = False

    except KeyboardInterrupt:
        print("\nEncerrando o programa...")

    finally:
        for captura_camera in capturas.values():
            captura_camera.parar()

        salvador.aguardar()
        cv2.destroyAllWindows()

        print("VAR Futevolei encerrado.")


if __name__ == "__main__":
    main()
