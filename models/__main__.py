import sys
from .main import main

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python __main__.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    main(model_name)
