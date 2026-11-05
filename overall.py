            label = "Collecting camera frames"
            color = (220, 220, 220)

            if len(frame_buffer) == CAMERA_SEQ_LEN:
                clip = torch.stack(list(frame_buffer)).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    probabilities = torch.softmax(camera_model(clip), dim=1)[0]

                p_fall = probabilities[FALL_INDEX].item()
                prediction = int(p_fall >= CAMERA_FALL_THRESHOLD)

                update_sensor(
                    "camera",
                    prediction,
                    detail=f"fall confidence {p_fall:.2f}"
                )

                label = (
                    f"CAMERA FALL {p_fall:.2f}"
                    if prediction == 1
                    else f"CAMERA NORMAL {1 - p_fall:.2f}"
                )
                color = (0, 0, 255) if prediction == 1 else (0, 200, 0)

            with lock:
                event_active = event["active"]

            if event_active:
                label = "ELDERGUARD: EMERGENCY EVENT ACTIVE"
                color = (0, 0, 255)

            cv2.putText(
                frame, label, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )
            cv2.imshow("ElderGuard AI - Camera + Radar", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                log("Stop requested by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


# =========================================================
# 9. MAIN APPLICATION
# =========================================================

def main():
    print_header()
    print(f"Device       : {DEVICE}")
    print(f"Camera input : {VIDEO_SOURCE}")
    print(f"Radar port   : {RADAR_PORT}")
    print("Fall rule    : Either camera OR radar can confirm a fall")

    section("SYSTEM INITIALIZATION")

    log("Loading camera CNN-LSTM model...")
    camera_model = load_camera_model()
    log("Camera CNN-LSTM model loaded successfully.")

    log("Loading radar SVM model...")
    radar_model = load_radar_model()
    log("Radar SVM model loaded successfully.")

    log("Models ready.")

    section("LIVE MONITORING")
    log("Camera status: waiting for frames.")
    log("Radar status : waiting for ESP32 data.")
    log("Press Q in the camera window to stop monitoring.")

    monitor_thread = threading.Thread(
        target=alert_monitor,
        daemon=True
    )
    radar_thread = threading.Thread(
        target=radar_worker,
        args=(radar_model,),
        daemon=True
    )

    monitor_thread.start()
    radar_thread.start()

    try:
        run_camera(camera_model)
    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        radar_thread.join(timeout=2)
        monitor_thread.join(timeout=2)
        log("ElderGuard AI stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_event.set()
        print("\nMonitoring interrupted by user.")
    except Exception as error:
        stop_event.set()
        print(f"\nSYSTEM ERROR: {error}")
        raise
