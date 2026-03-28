import argparse
from webcam_mode import run_webcam

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="webcam", choices=["webcam"])
    args = parser.parse_args()

    if args.mode == "webcam":
        run_webcam(frame_skip=3)

if __name__ == "__main__":
    main()