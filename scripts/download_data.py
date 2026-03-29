import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove", action="store_true")
    args = parser.parse_args()
    if args.glove:
        print("Placeholder: download GloVe vectors into data/glove/")


if __name__ == "__main__":
    main()
