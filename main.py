import cv2
import mediapipe as mp
import math
import time
import winsound
import threading

waktu_mata_tertutup = None
batas_kantuk = 2

waktu_alarm_terakhir = None
interval_alarm = 10


# Fungsi menghitung jarak antara dua titik
def hitung_jarak(titik1, titik2):
    return math.sqrt(
        (titik2.x - titik1.x) ** 2 +
        (titik2.y - titik1.y) ** 2
    )


# Fungsi menghitung Eye Aspect Ratio
def hitung_ear(wajah, indeks_mata):
    p1 = wajah[indeks_mata[0]]
    p2 = wajah[indeks_mata[1]]
    p3 = wajah[indeks_mata[2]]
    p4 = wajah[indeks_mata[3]]
    p5 = wajah[indeks_mata[4]]
    p6 = wajah[indeks_mata[5]]

    jarak_vertikal1 = hitung_jarak(p2, p6)
    jarak_vertikal2 = hitung_jarak(p3, p5)
    jarak_horizontal = hitung_jarak(p1, p4)

    ear = (
        jarak_vertikal1 + jarak_vertikal2
    ) / (2 * jarak_horizontal)

    return ear

# Fungsi membunyikan alarm dua kali
def bunyikan_alarm():
    for i in range(3):
        winsound.Beep(1000, 500)
        time.sleep(0.2)

# Mengatur MediaPipe Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Lokasi file model
model_path = "face_landmarker.task"

# Pengaturan deteksi wajah
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

# Membuka webcam
kamera = cv2.VideoCapture(0)

if not kamera.isOpened():
    print("Webcam tidak dapat dibuka!")
    exit()

print("Deteksi wajah dimulai!")

# Menghitung waktu video
waktu_mulai = cv2.getTickCount()

# Menjalankan Face Landmarker
with FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        berhasil, frame = kamera.read()

        if not berhasil:
            print("Gagal mengambil gambar!")
            break

        # Membalik gambar agar seperti cermin
        frame = cv2.flip(frame, 1)

        # Mengubah format BGR menjadi RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mengubah gambar menjadi format MediaPipe
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Menghitung waktu video dalam milidetik
        waktu = int(
            (cv2.getTickCount() - waktu_mulai)
            / cv2.getTickFrequency() * 1000
        )

        # Mendeteksi wajah
        hasil = landmarker.detect_for_video(
            mp_image,
            waktu
        )

        # Menggambar titik wajah dan menghitung EAR
        if hasil.face_landmarks:
            for wajah in hasil.face_landmarks:

                tinggi, lebar, _ = frame.shape

                # Titik mata kanan
                mata_kanan = [33, 160, 158, 133, 153, 144]

                # Titik mata kiri
                mata_kiri = [362, 385, 387, 263, 373, 380]

                # Menghitung EAR kedua mata
                ear_kanan = hitung_ear(wajah, mata_kanan)
                ear_kiri = hitung_ear(wajah, mata_kiri)

                # Menghitung rata-rata EAR
                ear_rata_rata = (ear_kanan + ear_kiri) / 2

                # Menentukan batas mata tertutup
                threshold_ear = 0.20

                # Mendeteksi kondisi mata tertutup
                if ear_rata_rata < threshold_ear:

                    if waktu_mata_tertutup is None:
                        waktu_mata_tertutup = time.time()

                    durasi_tertutup = time.time() - waktu_mata_tertutup
                    if durasi_tertutup >= batas_kantuk:
                        status_mata = "TERDETEKSI KANTUK"
                        warna_status = (0, 0, 255)

                        # Alarm berbunyi setiap 20 detik
                        waktu_sekarang = time.time()

                        if (
                            waktu_alarm_terakhir is None
                            or waktu_sekarang - waktu_alarm_terakhir >= interval_alarm
                        ):
                            waktu_alarm_terakhir = waktu_sekarang

                            threading.Thread(
                                target=bunyikan_alarm,
                                daemon=True
                            ).start()

                    else:
                        status_mata = "MATA TERTUTUP"
                        warna_status = (0, 165, 255)
                else:
                    waktu_mata_tertutup = None
                    durasi_tertutup = 0
                    waktu_alarm_terakhir = None

                    status_mata = "MATA TERBUKA"
                    warna_status = (0, 255, 0)

                # Menggambar semua titik wajah
                for titik in wajah:
                    x = int(titik.x * lebar)
                    y = int(titik.y * tinggi)

                    cv2.circle(
                        frame, (x, y), 1, (0, 255, 0), -1
                    )

                # Menandai mata kanan dengan warna biru
                for indeks in mata_kanan:
                    x = int(wajah[indeks].x * lebar)
                    y = int(wajah[indeks].y * tinggi)

                    cv2.circle(
                        frame, (x, y), 4, (255, 0, 0), -1
                    )

                # Menandai mata kiri dengan warna merah
                for indeks in mata_kiri:
                    x = int(wajah[indeks].x * lebar)
                    y = int(wajah[indeks].y * tinggi)

                    cv2.circle(
                        frame, (x, y), 4, (0, 0, 255), -1
                    )

                # Menampilkan nilai EAR
                cv2.putText(
                    frame,
                    f"EAR Kanan: {ear_kanan:.3f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"EAR Kiri: {ear_kiri:.3f}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                frame,
                status_mata,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                warna_status,
                2
            )

        else:
            cv2.putText(
                frame,
                "WAJAH TIDAK TERDETEKSI",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Menampilkan hasil
        cv2.imshow("Deteksi Wajah", frame)

        # Tekan Q untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

kamera.release()
cv2.destroyAllWindows()
