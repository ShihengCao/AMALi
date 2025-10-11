from src.post_process import parse_log, parse_out
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_post_process.py <app_name>")
        sys.exit(1)
    app_name = sys.argv[1]
    print("complete analysis and start parsing output")
    parse_log(app_name)
    parse_out(app_name)
    print("complete parsing")

if __name__ == "__main__":
    main()