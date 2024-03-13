from my_utils import ternary_entropy
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--probability",required=True, type=float, help="half insertion rate")
    args = parser.parse_args()

    print(f"p = {args.probability} ; ternary entropy = {ternary_entropy(args.probability)} bpc")