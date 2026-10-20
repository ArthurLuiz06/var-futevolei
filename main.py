
import cv2


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Erro: não foi possível abrir a webcam.")
        print("Verifique se a câmera está conectada e disponível.")
        camera.release()
        return

    print("Webcam iniciada!")
    print("Pressione ESC para fechar a janela.")

    while True:
        sucesso, frame = camera.read()

        if not sucesso:
            print("Erro ao capturar imagem da webcam.")
            break

        cv2.imshow("VAR Futevolei - Teste da Webcam", frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Teste finalizado.")


if __name__ == "__main__":
    main()
