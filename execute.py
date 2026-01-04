from main import main

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # safe on Linux, required on some platforms
    main()
