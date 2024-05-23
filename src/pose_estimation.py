import cv2
import mediapipe as mp
import numpy as np

class PoseEstimation:
    def __init__(self, video_path):
        """
        DOCSTRING: PoseEstimation sınıfını başlatır ve MediaPipe Pose modelini yükler.
        """
        # MediaPipe Pose modelini başlat
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(video_path)
        self.count = 0
        self.arms_up_flag = False
        self.Left_Wrist_Points = []  # Sol bilek koordinatlarını tutan liste
        self.Right_Wrist_Points = []  # Sağ bilek koordinatlarını tutan liste

    def __del__(self):
        """
        DOCSTRING: PoseEstimation sınıfı örneği silinirken kaynakları serbest bırakır.
        """
        self.cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def calculate_angle(a, b, c):
        """
        DOCSTRING: Üç nokta arasındaki açıyı hesaplar.
        INPUT: a, b, c (noktalar)
        OUTPUT: Açı (derece)
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arccos(np.dot(b - a, b - c) / (np.linalg.norm(b - a) * np.linalg.norm(b - c)))
        angle = np.degrees(radians)
        return angle

    def draw_lines(self, image):
        """
        DOCSTRING: Ekranın belirli bir bölgesine çizgiler çizer.
        INPUT: image (çerçeve görüntüsü)
        OUTPUT: Yok
        """
        # Çizgilerin piksel değerlerini belirleyin
        self.top_line_y = 250
        self.bottom_line_y = 650
        height, width, _ = image.shape
        cv2.line(image, (0, self.top_line_y), (width, self.top_line_y), (255, 0, 0), thickness=2)
        cv2.line(image, (0, self.bottom_line_y), (width, self.bottom_line_y), (255, 0, 0), thickness=2)

    def check_left_arm(self, landmarks):
        """
        DOCSTRING: Sol kolun hareketlerini kontrol eder ve bilek koordinatlarını kaydeder.
        INPUT: landmarks (pose işaret noktaları)
        OUTPUT: Sol kol açısı
        """
        # Sol omuz, dirsek ve bilek noktalarını alın
        left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Sol bilek koordinatlarını listeye ekle
        self.Left_Wrist_Points.append(left_wrist)
        print(f"Sol bilek koordinatı: ({left_wrist[0]:.3f}, {left_wrist[1]:.3f})")
        # Sol omuz, dirsek ve bilek arasındaki açıyı hesaplayın
        left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)

        return left_angle

    def check_right_arm(self, landmarks):
        """
        DOCSTRING: Sağ kolun hareketlerini kontrol eder ve bilek koordinatlarını kaydeder.
        INPUT: landmarks (pose işaret noktaları)
        OUTPUT: Sağ kol açısı
        """
        # Sağ omuz, dirsek ve bilek noktalarını alın
        right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Sağ bilek koordinatlarını listeye ekle
        self.Right_Wrist_Points.append(right_wrist)
        # Sağ omuz, dirsek ve bilek arasındaki açıyı hesaplayın
        right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        print(f"Sağ bilek koordinatı: ({right_wrist[0]:.3f}, {right_wrist[1]:.3f})")
        print("**********************")
        return right_angle

    def check_posture(self, landmarks, image):
        """
        DOCSTRING: Çalışanın doğru oturma pozisyonunu kontrol eder ve kol hareketini sayar.
        INPUT: landmarks (pose işaret noktaları), image (çerçeve görüntüsü)
        OUTPUT: "UP" veya "DOWN" duruş sonucu
        """
        left_angle = self.check_left_arm(landmarks)
        right_angle = self.check_right_arm(landmarks)

        # Kolların yukarıda ve aşağıda olması için açı aralıkları
        upper_angle_threshold = 60  # Kolların yukarıda olduğunu belirten açı eşiği
        lower_angle_threshold = 100 # Kolların aşağıda olduğunu belirten açı eşiği

        # Sol ve sağ bileklerin y koordinatlarını alın
        left_wrist_y = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0]
        right_wrist_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image.shape[0]

        # Sol ve sağ bileklerin çizgiler arasında olup olmadığını kontrol edin
        if (self.top_line_y < left_wrist_y < self.bottom_line_y) and (self.top_line_y < right_wrist_y < self.bottom_line_y):
            if left_angle < upper_angle_threshold and right_angle < upper_angle_threshold:
                if not self.arms_up_flag:
                    self.arms_up_flag = True  # Kollar yukarıda, bayrağı set et
            elif left_angle > lower_angle_threshold and right_angle > lower_angle_threshold:
                if self.arms_up_flag:
                    self.count += 1
                    self.arms_up_flag = False  # Kollar aşağı indi, bayrağı sıfırla ve sayıyı artır

        return "UP" if self.arms_up_flag == True else "DOWN"

    def run(self):
        """
        DOCSTRING: Canlı kamera görüntüsünden duruş tahmini yapar ve sonucu ekranda gösterir.
        INPUT: Yok
        OUTPUT: Yok
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                # Çizgi çizme işlemini gerçekleştir
                self.draw_lines(image)

                posture = self.check_posture(results.pose_landmarks.landmark, image)
                cv2.putText(image, posture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Count: {self.count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)

            resized = cv2.resize(image, (800, 600))

            cv2.imshow('Pose Estimation', resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    video_path = r"D:\Python\pythonProject\NORBİT\images\(1).mp4"  # Video dosyasının yolu
    pose_estimation = PoseEstimation(video_path)
    pose_estimation.run()
