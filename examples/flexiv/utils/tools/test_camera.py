# 高帧率大于60fps的组合如下（你可以直接从这里选择）：

# 320x240@90.00
# 320x240@120.00
# 640x480@90.00
# 640x480@120.00
# 请求: 1600x1200@60 -> 实际: 1600x1200@60.00 ✔️
#  1280x1024@60.00
# 你可以在代码或需求中直接选择上述任意分辨率+帧率组合使用。
import cv2
import time

def main():
    # 打开摄像头（设备索引通常为0，可能因机器不同而不同）
    cap = cv2.VideoCapture(2)

    # 设置摄像头为MJPEG格式，否则帧率上不去
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # 设置分辨率为1600x1200
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # 设置帧率为60
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("按 q 或 Ctrl+C 退出")

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break

            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time

            # 每隔1秒刷新一次fps数值
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = current_time

            # 可视化: 在左上角绘制fps
            disp = frame.copy()
            cv2.putText(disp, f'FPS: {fps:.2f}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Camera Preview', disp)

            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n退出")
                break

    except KeyboardInterrupt:
        print("\n退出")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
