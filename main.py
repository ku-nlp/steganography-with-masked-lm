from argparse import ArgumentParser
from masked_stego import MaskedStego

def encode(cover_text: str, message: str, mask_interval: int, score_threshold: float):
    print(masked_stego(cover_text, message, mask_interval, score_threshold))


if __name__ == "__main__":
    psr = ArgumentParser()
    psr.add_argument('text', type=str, help='Text to encode or decode message.')
    psr.add_argument('-d', '--decode', action='store_true', help='If this flag is set, decodes from the text.')
    psr.add_argument('-m', '--message', type=str, help='Binary message to encode consisting of 0s or 1s.')
    psr.add_argument('-i', '--mask_interval', type=int, default=3)
    psr.add_argument('-s', '--score_threshold', type=float, default=0.01)
    args = psr.parse_args()

    masked_stego = MaskedStego()

    if args.decode:
        print(masked_stego.decode(args.text, args.mask_interval, args.score_threshold))
    else:
        print(masked_stego(args.text, args.message, args.mask_interval, args.score_threshold))
