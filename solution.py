import sys

if __name__ == "__main__":
    if '--generate' in sys.argv:
        from solution import generate as main
    else:
        from solution import main

    main.main()
